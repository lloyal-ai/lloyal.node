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
 * fork from children for MCTS/beam search, fork from a draft for speculative
 * decoding.
 *
 * The produce/commit protocol separates sampling from state advancement:
 * produce() samples without writing to KV, letting you inspect the result
 * before deciding to commit(). This two-phase split is what makes speculative
 * verification and tree search natural.
 *
 * @example Best-of-N with perplexity selection
 * ```js
 * const root = Branch.create(ctx, 0, tokens.length, { temperature: 0.8 });
 * root.captureLogits();
 *
 * const candidates = [1, 2, 3, 4, 5].map((seqId, i) => {
 *   const branch = root.fork(seqId);
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
   * @param {number} seqId - Sequence ID for this branch
   * @param {number} position - Starting position (typically prompt token count)
   * @param {SamplingParams} [params] - Sampling parameters (temperature, topP, etc.)
   * @returns {Branch} New Branch instance
   */
  static create(ctx, seqId, position, params) {
    const handle = ctx._branchCreate(seqId, position, params);
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
   * @param {number} newSeqId - Sequence ID for the forked branch
   * @returns {Branch} New forked Branch
   */
  fork(newSeqId) {
    this._ensureNotDisposed();
    const newHandle = this._ctx._branchFork(this._handle, newSeqId);
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
   * Release the handle but keep KV cache entries intact
   *
   * Use for winners: you're done branching but want to continue generation
   * on this sequence using raw ctx.decode()/ctx.sample() calls. The KV
   * cache entries remain at their current positions.
   */
  destroy() {
    if (this._disposed) return;
    this._ctx._branchDestroy(this._handle);
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
   * const root = Branch.create(ctx, 0, pos, { temperature: 0.9 });
   * root.captureLogits();
   *
   * // Fork and reseed for diversity
   * const branches = [1, 2, 3, 4, 5].map((seqId, i) => {
   *   const branch = root.fork(seqId);
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

  /** @returns {number} Branch's sequence ID */
  get seqId() {
    this._ensureNotDisposed();
    return this._ctx._branchGetSeqId(this._handle);
  }

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

  // ===== INTERNAL =====

  _ensureNotDisposed() {
    if (this._disposed) {
      throw new Error('Branch has been disposed');
    }
  }
}

module.exports = { Branch };
