/**
 * Branch - Forkable inference handle for covalent generation
 *
 * A Branch owns everything needed for independent generation: a KV cache
 * sequence, sampler chain, logits snapshot, and perplexity tracker.
 *
 * Forking is cheap — the KV prefix is shared in memory (metadata-only operation under unified KV —
 * no KV tensor buffers are copied), so sibling branches read from the same physical KV entries.
 * Only tokens decoded after the fork point are exclusive to each branch.
 * This is the covalent property: branches share a bond (common prefix)
 * while diverging independently.
 *
 * Branches form trees, not just flat lists. Fork from root for best-of-N,
 * fork from children for tree search/beam search, fork from a draft for speculative
 * decoding.
 *
 * The produce/commit protocol separates sampling from state advancement:
 * produce() samples without writing to KV, letting you inspect the result
 * before deciding to commit(). This two-phase split is what makes speculative
 * verification and tree search natural.
 *
 * @example Best-of-N with perplexity selection
 * ```js
 * const root = Branch.create(ctx, tokens.length, { temperature: 0.8 });
 * root.captureLogits();
 *
 * const candidates = Array.from({ length: 5 }, (_, i) => {
 *   const branch = root.fork();
 *   branch.reseedSampler(1000 + i);
 *   return branch;
 * });
 *
 * for (let t = 0; t < 50; t++) {
 *   for (const branch of candidates) {
 *     const { token, isStop } = branch.produce();
 *     if (isStop) continue;
 *     branch.commit(token);
 *   }
 * }
 *
 * const best = candidates.reduce((a, b) => a.perplexity < b.perplexity ? a : b);
 * for (const c of candidates) { if (c !== best) c.prune(); }
 * ```
 */

class Branch {
  /**
   * @param {SessionContext} ctx
   * @param {number} handle
   */
  constructor(ctx, handle) {
    this._ctx = ctx;
    this._handle = handle;
    this._disposed = false;
  }

  /**
   * Create a root branch at the given position
   *
   * The branch takes ownership of the sequence and creates its own sampler
   * chain from the provided params. Call captureLogits() after prefill to
   * freeze the logit distribution before forking.
   *
   * @param {SessionContext} ctx - SessionContext to create branch on
   * @param {number} position - Starting position (typically prompt token count)
   * @param {SamplingParams} [params] - Sampling parameters (temperature, topP, etc.)
   * @param {number} [nBatch] - Per-branch batch size override (defaults to context nBatch).
   *   Controls chunk size for prefill() (decode_and_capture_batch). Has no effect on
   *   single-token commit() which uses a zero-allocation fast path. Useful for tuning
   *   memory/throughput tradeoff on bulk token decode — e.g. smaller nBatch for cheap
   *   exploration branches, larger for the trunk.
   * @param {string} [grammar] - GBNF grammar string for constrained generation.
   *   When provided, sample() returns only grammar-valid tokens. The grammar state
   *   is cloned on fork(), so sibling branches can diverge independently.
   * @returns {Branch} New Branch instance
   */
  static create(ctx, position, params, nBatch, grammar) {
    const handle = ctx._branchCreate(position, params, nBatch, grammar);
    return new Branch(ctx, handle);
  }

  /**
   * Fork this branch to a new sequence
   *
   * The child shares the parent's KV prefix in memory (metadata-only under unified KV, no KV buffer copy).
   * Logits, sampler state, and perplexity tracker are cloned so the child
   * can diverge independently. Fork from any branch — root or intermediate —
   * to build arbitrarily deep trees.
   *
   * Call reseedSampler() on each child for stochastic diversity.
   *
   * @returns {Branch} New forked Branch
   */
  fork() {
    this._ensureNotDisposed();
    const newHandle = this._ctx._branchFork(this._handle);
    return new Branch(this._ctx, newHandle);
  }

  /**
   * Freeze the current logit distribution into this branch
   *
   * Logits are ephemeral — they're overwritten on the next decode() call.
   * Capturing preserves them so this branch (and any forks from it) can
   * sample from the same distribution. Essential before fork().
   */
  captureLogits() {
    this._ensureNotDisposed();
    this._ctx._branchCaptureLogits(this._handle);
  }

  /**
   * Get a copy of this branch's captured logits snapshot
   *
   * Returns n_vocab floats — the raw logit distribution from the last
   * decode_and_capture or captureLogits() call. Use for distributional
   * analysis (KL divergence, entropy, top-k overlap) without crossing
   * the sampling chain.
   *
   * @returns {Float32Array} Copy of the logits snapshot (n_vocab elements)
   * @throws {Error} If no logits have been captured yet
   */
  getLogits() {
    this._ensureNotDisposed();
    return this._ctx._branchGetLogits(this._handle);
  }

  /**
   * Single-token forward pass with logit snapshot
   *
   * Runs one decode step (writing the token's KV entries), advances position,
   * and captures the resulting logits for the next sample() call.
   *
   * @param {number} token - Token to decode
   */
  decodeAndCaptureOne(token) {
    this._ensureNotDisposed();
    this._ctx._branchDecodeAndCaptureOne(this._handle, token);
  }

  /**
   * Bulk-decode tokens into the branch's KV cache and capture logits
   *
   * Feeds an array of tokens through the model. tokens.length is the total
   * count to process; the branch's nBatch (set at Branch.create) controls
   * how many are sent per llama_decode call. For example, 500 tokens with
   * nBatch=64 makes 8 llama_decode calls (7×64 + 1×52). With nBatch=512
   * it makes 1.
   *
   * Advances position by tokens.length and stores the final logits into
   * the branch's internal snapshot. The next produce()/sample() call reads
   * from that snapshot — logits never cross the JS boundary.
   *
   * Does NOT accept tokens into the sampler's repeat-penalty window — use
   * this for external tokens (user input between turns), not model-generated
   * tokens. For model output, use commit() which does accept + decode.
   *
   * This is the branch-level equivalent of ctx.decode().
   *
   * @param {number[]} tokens - Token IDs to decode
   */
  prefill(tokens) {
    this._ensureNotDisposed();
    this._ctx._branchDecodeAndCaptureBatch(this._handle, tokens);
  }

  /**
   * Sample next token from branch's logits snapshot
   *
   * Applies the branch's full sampler chain (top-k, top-p, temperature,
   * repeat/presence penalties) to the captured logits.
   *
   * @returns {number} Sampled token ID
   */
  sample() {
    this._ensureNotDisposed();
    return this._ctx._branchSample(this._handle);
  }

  /**
   * Record token in the sampler's repeat/presence penalty window
   *
   * @param {number} token - Token to accept
   */
  accept(token) {
    this._ensureNotDisposed();
    this._ctx._branchAccept(this._handle, token);
  }

  /**
   * Discard this branch entirely — remove its KV entries and free the handle
   *
   * Use for losers: branches whose generation you want to erase completely.
   * Only removes KV entries divergent from the shared prefix; sibling
   * branches are unaffected.
   */
  prune() {
    if (this._disposed) return;
    this._ctx._branchPrune(this._handle);
    this._disposed = true;
  }

  /**
   * Discard this branch and all its descendants — CASCADE delete
   *
   * Iterative post-order traversal: prunes children first, then this branch.
   * Use when you want to tear down an entire subtree (e.g. abandoned search path).
   */
  pruneSubtree() {
    if (this._disposed) return;
    this._ctx._branchPruneSubtree(this._handle);
    this._disposed = true;
  }

  /**
   * Reseed the sampler's PRNG for diversity after fork()
   *
   * CRITICAL for parallel generation: Without reseeding, all forked branches
   * produce identical outputs because they share the same PRNG state.
   *
   * Only affects stochastic samplers (temperature > 0). Greedy samplers are unchanged.
   *
   * @param {number} seed - New seed for the PRNG
   *
   * @example
   * ```js
   * const root = Branch.create(ctx, pos, { temperature: 0.9 });
   * root.captureLogits();
   *
   * // Fork and reseed for diversity
   * const branches = Array.from({ length: 5 }, (_, i) => {
   *   const branch = root.fork();
   *   branch.reseedSampler(1000 + i);  // Each branch gets unique seed
   *   return branch;
   * });
   * ```
   */
  reseedSampler(seed) {
    this._ensureNotDisposed();
    this._ctx._branchSamplerChainReseed(this._handle, seed);
  }

  /**
   * Apply dynamic logit adjustments for this branch only
   *
   * Unlike logit_bias (which is cloned on fork), steer biases are NOT inherited
   * by child branches. Each branch manages its own steer state independently.
   *
   * Use cases:
   * - tsampler: Block tokens that would create repeated N-grams (per-path history)
   * - Tree search: Block already-explored actions at this node (not inherited by children)
   *
   * Applied during sample() in order: Grammar → Logit Bias → Steer → Sampler Chain
   *
   * @param {Array<{token: number, bias: number}>} biases - Token adjustments.
   *   Use -Infinity to block a token, positive values to boost.
   *
   * @example Block tokens for N-gram deduplication
   * ```js
   * // Client computes blocked tokens based on generated text
   * const blocked = computeNgramBlocks(generatedText);
   * branch.steer(blocked.map(t => ({ token: t, bias: -Infinity })));
   *
   * const { token } = branch.produce();  // Blocked tokens won't be sampled
   * branch.commit(token);
   *
   * branch.clearSteer();  // Reset for next iteration
   * ```
   */
  steer(biases) {
    this._ensureNotDisposed();
    this._ctx._branchSteer(this._handle, biases);
  }

  /**
   * Clear all steer biases from this branch
   *
   * Removes any dynamic logit adjustments set by steer().
   */
  clearSteer() {
    this._ensureNotDisposed();
    this._ctx._branchClearSteer(this._handle);
  }

  /**
   * Sample the next token without advancing state
   *
   * No KV write, no position update. Inspect the result before deciding
   * to commit() — this separation is what enables speculative verification
   * and conditional branching.
   *
   * @returns {{ token: number, text: string, isStop: boolean }}
   */
  produce() {
    this._ensureNotDisposed();
    const token = this.sample();
    return {
      token,
      text: this._ctx.tokenToText(token),
      isStop: this._ctx.isStopToken(token),
    };
  }

  /**
   * Accept and advance — write token to KV and update branch state
   *
   * Accepts the token for repeat-penalty tracking, decodes it (writing to
   * KV cache), and captures the resulting logits for the next produce() call.
   *
   * @param {number} token - Token to commit (from produce())
   */
  commit(token) {
    this._ensureNotDisposed();
    this.accept(token);
    this.decodeAndCaptureOne(token);
  }

  // ===== ACCESSORS =====

  /** @returns {number} Branch's current position (number of tokens decoded) */
  get position() {
    this._ensureNotDisposed();
    return this._ctx._branchGetPosition(this._handle);
  }

  /** @returns {number} Branch's perplexity (exp of mean surprisal) */
  get perplexity() {
    this._ensureNotDisposed();
    return this._ctx._branchGetPerplexity(this._handle);
  }

  /** @returns {number} Internal handle (for debugging) */
  get handle() {
    return this._handle;
  }

  /** @returns {boolean} Whether this branch has been disposed */
  get disposed() {
    return this._disposed;
  }

  /** @returns {number|null} Parent branch handle, or null if root */
  get parent() {
    this._ensureNotDisposed();
    const h = this._ctx._branchParent(this._handle);
    return h === 0 ? null : h;
  }

  /** @returns {number[]} Child branch handles */
  get children() {
    this._ensureNotDisposed();
    return this._ctx._branchChildren(this._handle);
  }

  /** @returns {boolean} True if this branch has no children */
  get isLeaf() {
    this._ensureNotDisposed();
    return this._ctx._branchIsLeaf(this._handle);
  }

  /** @returns {boolean} True if this branch holds a KV lease */
  get isActive() {
    this._ensureNotDisposed();
    return this._ctx._branchIsActive(this._handle);
  }

  // ===== INTERNAL =====

  _ensureNotDisposed() {
    if (this._disposed) {
      throw new Error('Branch has been disposed');
    }
  }
}

module.exports = { Branch };
