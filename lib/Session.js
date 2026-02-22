/**
 * Session - Trunk lifecycle + conversation delta helpers
 *
 * Owns the current "trunk" branch — the single conversation thread that
 * persists across agent swarms and follow-up turns. Provides promote()
 * to crown a winner (retainOnly + reassign), and delta helpers that
 * centralize the sep + formatChat + tokenize + prefill pattern.
 *
 * Session does NOT own the SessionContext or BranchStore — the consumer
 * creates those and passes them in. dispose() prunes trunk only.
 */
class Session {
  /**
   * @param {{ ctx: SessionContext, store: BranchStore }} opts
   */
  constructor({ ctx, store }) {
    this._ctx = ctx;
    this._store = store;
    this._trunk = null;
  }

  /** @returns {Branch|null} Current trunk branch */
  get trunk() {
    return this._trunk;
  }

  /** @param {Branch} branch - Assign initial trunk (no promote) */
  set trunk(branch) {
    this._trunk = branch;
  }

  /**
   * Promote a winner to trunk — retainOnly + reassign
   *
   * Safe even if winner is the only branch (resets topology, no-op on KV).
   * @param {Branch} winner
   */
  async promote(winner) {
    await this._store.retainOnly(winner);
    this._trunk = winner;
  }

  /**
   * Dispose trunk only — consumer owns ctx and other resources
   */
  async dispose() {
    if (this._trunk && !this._trunk.disposed) {
      await this._trunk.prune();
    }
    this._trunk = null;
  }

  /**
   * Prefill a user turn into trunk
   *
   * Centralizes: sep + formatChat([system:'', user:content]) + tokenize(false) + prefill
   *
   * @param {string} content - User message content
   * @param {{ tools?: string }} [opts]
   */
  async prefillUser(content, opts = {}) {
    const sep = this._ctx.getTurnSeparator();
    const fmtOpts = opts.tools ? { tools: opts.tools } : {};
    const { prompt } = await this._ctx.formatChat(
      JSON.stringify([{ role: 'system', content: '' }, { role: 'user', content }]),
      fmtOpts
    );
    const delta = await this._ctx.tokenize(prompt, false);
    await this._trunk.prefill([...sep, ...delta]);
  }

  /**
   * Prefill a tool result turn into trunk
   *
   * Centralizes: sep + formatChat([system:'', tool:result]) + tokenize(false) + prefill
   *
   * @param {string} resultStr - JSON-stringified tool result
   * @param {string} callId - Tool call ID
   */
  async prefillToolResult(resultStr, callId) {
    const sep = this._ctx.getTurnSeparator();
    const { prompt } = await this._ctx.formatChat(
      JSON.stringify([
        { role: 'system', content: '' },
        { role: 'tool', content: resultStr, tool_call_id: callId },
      ])
    );
    const delta = await this._ctx.tokenize(prompt, false);
    await this._trunk.prefill([...sep, ...delta]);
  }
}

module.exports = { Session };
