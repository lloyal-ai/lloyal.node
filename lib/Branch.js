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
 * await root.prefill(tokens);
 *
 * const results = [];
 * for (let i = 0; i < 5; i++) {
 *   const branch = await root.fork();
 *   branch.reseedSampler(1000 + i);
 *   const tokens = [];
 *   for await (const { token } of branch) tokens.push(token);
 *   results.push({ branch, tokens, ppl: branch.perplexity });
 * }
 *
 * const best = results.reduce((a, b) => a.ppl < b.ppl ? a : b);
 * for (const r of results) { if (r !== best) await r.branch.prune(); }
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
   * chain from the provided params. Call prefill() to decode prompt tokens
   * and capture the logit distribution before forking.
   *
   * @param {SessionContext} ctx - SessionContext to create branch on
   * @param {number} position - Starting position (typically prompt token count)
   * @param {SamplingParams} [params] - Sampling parameters (temperature, topP, etc.)
   * @param {number} [nBatch] - Per-branch batch size override (defaults to context nBatch).
   *   Controls chunk size for prefill(). Has no effect on
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
   * @returns {Promise<Branch>} New forked Branch
   */
  async fork() {
    this._ensureNotDisposed();
    const newHandle = this._ctx._branchFork(this._handle);
    return new Branch(this._ctx, newHandle);
  }

  /**
   * Get a copy of this branch's captured logits snapshot
   *
   * Returns n_vocab floats — the raw logit distribution from the last
   * prefill() or commit() call. Use for distributional analysis
   * (KL divergence, entropy, top-k overlap) without crossing the
   * sampling chain.
   *
   * @returns {Float32Array} Copy of the logits snapshot (n_vocab elements)
   * @throws {Error} If no logits have been captured yet
   */
  getLogits() {
    this._ensureNotDisposed();
    return this._ctx._branchGetLogits(this._handle);
  }

  /**
   * Bulk-decode tokens into the branch's KV cache and capture logits
   *
   * Feeds an array of tokens through the model. tokens.length is the total
   * count to process; the branch's nBatch (set at Branch.create) controls
   * how many are sent per llama_decode call. For example, 500 tokens with
   * nBatch=64 makes 8 llama_decode calls (7x64 + 1x52). With nBatch=512
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
   * The primary way to feed tokens into a branch's KV cache.
   *
   * @param {number[]} tokens - Token IDs to decode
   * @returns {Promise<void>}
   */
  async prefill(tokens) {
    this._ensureNotDisposed();
    await this._ctx._branchPrefill(this._handle, tokens);
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
   *
   * @returns {Promise<void>}
   */
  async prune() {
    if (this._disposed) return;
    this._ctx._branchPrune(this._handle);
    this._disposed = true;
  }

  /**
   * Discard this branch and all its descendants — CASCADE delete
   *
   * Iterative post-order traversal: prunes children first, then this branch.
   * Use when you want to tear down an entire subtree (e.g. abandoned search path).
   *
   * @returns {Promise<void>}
   */
  async pruneSubtree() {
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
   * await root.prefill(promptTokens);
   *
   * // Fork and reseed for diversity
   * const branches = [];
   * for (let i = 0; i < 5; i++) {
   *   const branch = await root.fork();
   *   branch.reseedSampler(1000 + i);  // Each branch gets unique seed
   *   branches.push(branch);
   * }
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
   * Applied during sample() in order: Grammar -> Logit Bias -> Steer -> Sampler Chain
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
   * const { token } = await branch.produce();  // Blocked tokens won't be sampled
   * await branch.commit(token);
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
   * Replace the sampler chain with new parameters (memoized)
   *
   * If the new params match the current chain's params, this is a no-op.
   * Otherwise the old chain is freed and a new one is created.
   *
   * @param {SamplingParams} params - New sampling parameters
   */
  setSamplerParams(params) {
    this._ensureNotDisposed();
    this._ctx._branchSetSamplerParams(this._handle, params);
  }

  /**
   * Replace or remove the grammar constraint
   *
   * Pass a GBNF grammar string to constrain generation, or empty string / null
   * to remove the constraint. The grammar state is cloned on fork().
   *
   * @param {string} [grammarStr] - GBNF grammar string, or empty/null to remove
   */
  setGrammar(grammarStr) {
    this._ensureNotDisposed();
    this._ctx._branchSetGrammar(this._handle, grammarStr || '');
  }

  /**
   * Sample the next token without advancing state (async)
   *
   * No KV write, no position update. Inspect the result before deciding
   * to commit() — this separation is what enables speculative verification
   * and conditional branching.
   *
   * Async contract: local branches resolve immediately; cloud branches
   * may perform an HTTP round-trip. Use produceSync() when you know the
   * branch is local and want zero-overhead sampling.
   *
   * @returns {Promise<{ token: number, text: string, isStop: boolean }>}
   */
  async produce() {
    return this.produceSync();
  }

  /**
   * Sample the next token without advancing state (sync)
   *
   * Same as produce() but synchronous. Use when you know the branch is
   * local and want to avoid the microtick overhead of a promise.
   *
   * @returns {{ token: number, text: string, isStop: boolean }}
   */
  produceSync() {
    this._ensureNotDisposed();
    const token = this.sample();
    return {
      token,
      text: this._ctx.tokenToText(token),
      isStop: this._ctx.isStopToken(token),
    };
  }

  /**
   * Accept and decode — update branch state, then write token to KV
   *
   * Accepts the token into the sampler penalty window (for correct PPL
   * measurement), then decodes (writing to KV cache) and captures the
   * resulting logits for the next produce() call. Accept-first ordering
   * with rollback: if decode throws, sampler/grammar/metrics are restored
   * from clones taken before the accept.
   *
   * @param {number} token - Token to commit (from produce())
   * @returns {Promise<void>}
   */
  async commit(token) {
    this._ensureNotDisposed();
    await this._ctx._storeCommit([this._handle], [token]);
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

  // ===== ASYNC ITERATION =====

  /**
   * Async iterator — generate tokens until EOG
   *
   * Commit-before-yield: every yielded token is already written to KV and
   * accepted into the sampler. Breaking out of the loop is clean — no
   * orphaned uncommitted tokens, perplexity reflects all yielded tokens.
   *
   * For inspect-before-commit (speculative decoding, tree search), use
   * the produce()/commit() protocol directly.
   */
  async *[Symbol.asyncIterator]() {
    while (!this._disposed) {
      const { token, text, isStop } = await this.produce();
      if (isStop) return;
      await this.commit(token);
      yield { token, text };
    }
  }

  // ===== INTERNAL =====

  _ensureNotDisposed() {
    if (this._disposed) {
      throw new Error('Branch has been disposed');
    }
  }
}

module.exports = { Branch };
