import type { Branch } from './Branch';
import type { BranchStore } from './BranchStore';
import type { SessionContext } from './types';

/**
 * Build token delta for a user turn (sep + formatChat + tokenize)
 *
 * Usable with any branch — not tied to Session's trunk. This is the
 * canonical way to build a user-turn delta for warm prefill.
 *
 * @category Branching
 */
export async function buildUserDelta(
  ctx: SessionContext,
  content: string,
  opts: { tools?: string } = {}
): Promise<number[]> {
  const sep = ctx.getTurnSeparator();
  const fmtOpts = opts.tools ? { tools: opts.tools } : {};
  const { prompt } = await ctx.formatChat(
    JSON.stringify([{ role: 'system', content: '' }, { role: 'user', content }]),
    fmtOpts
  );
  const delta = await ctx.tokenize(prompt, false);
  return [...sep, ...delta];
}

/**
 * Build token delta for a tool result turn (sep + formatChat + tokenize)
 *
 * Usable with any branch — not tied to Session's trunk.
 *
 * @category Branching
 */
export async function buildToolResultDelta(
  ctx: SessionContext,
  resultStr: string,
  callId: string
): Promise<number[]> {
  const sep = ctx.getTurnSeparator();
  const { prompt } = await ctx.formatChat(
    JSON.stringify([
      { role: 'system', content: '' },
      { role: 'tool', content: resultStr, tool_call_id: callId },
    ])
  );
  const delta = await ctx.tokenize(prompt, false);
  return [...sep, ...delta];
}

/**
 * Session - Trunk lifecycle + conversation delta helpers
 *
 * Owns the current "trunk" branch and provides promote() to crown a winner,
 * plus delta helpers that centralize the sep + formatChat + tokenize + prefill
 * pattern for injecting new turns into an ongoing conversation.
 *
 * Session does NOT own the SessionContext or BranchStore — the consumer
 * creates those and passes them in. dispose() prunes trunk only.
 *
 * @example
 * ```typescript
 * const session = new Session({ ctx, store });
 * session.trunk = initialBranch;
 *
 * // After verification, promote the best attempt
 * await session.promote(bestAttempt.branch);
 *
 * // Inject a user turn and generate
 * await session.prefillUser('What about X?');
 * for await (const { text } of session.trunk) {
 *   process.stdout.write(text);
 * }
 *
 * // Cleanup
 * await session.dispose();
 * ctx.dispose();
 * ```
 *
 * @category Branching
 */
export class Session {
  private _ctx: SessionContext;
  private _store: BranchStore;
  private _trunk: Branch | null;

  constructor({ ctx, store }: { ctx: SessionContext; store: BranchStore }) {
    this._ctx = ctx;
    this._store = store;
    this._trunk = null;
  }

  /** Current trunk branch */
  get trunk(): Branch | null {
    return this._trunk;
  }

  /** Assign initial trunk (no promote) */
  set trunk(branch: Branch | null) {
    this._trunk = branch;
  }

  /**
   * Promote a winner to trunk — retainOnly + reassign
   *
   * Safe even if winner is the only branch (resets topology, no-op on KV).
   */
  async promote(winner: Branch): Promise<void> {
    await this._store.retainOnly(winner);
    this._trunk = winner;
  }

  /**
   * Dispose trunk only — consumer owns ctx and other resources
   */
  async dispose(): Promise<void> {
    if (this._trunk && !this._trunk.disposed) {
      await this._trunk.prune();
    }
    this._trunk = null;
  }

  /**
   * Prefill a user turn into trunk
   *
   * @param content - User message content
   * @param opts - Optional tools JSON string
   */
  async prefillUser(content: string, opts: { tools?: string } = {}): Promise<void> {
    const tokens = await buildUserDelta(this._ctx, content, opts);
    await this._trunk!.prefill(tokens);
  }

  /**
   * Prefill a tool result turn into trunk
   *
   * @param resultStr - JSON-stringified tool result
   * @param callId - Tool call ID
   */
  async prefillToolResult(resultStr: string, callId: string): Promise<void> {
    const tokens = await buildToolResultDelta(this._ctx, resultStr, callId);
    await this._trunk!.prefill(tokens);
  }
}
