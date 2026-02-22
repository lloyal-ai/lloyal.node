import type { Branch } from './Branch';
import type {
  AgentState,
  AgentTask,
  ParsedToolCall,
  RunAgentsOptions,
  RunAgentsResult,
  SessionContext,
} from './types';

/**
 * Fork an agent from a parent branch with its own system prompt + task
 *
 * Always prepends getTurnSeparator() for a clean structural break before
 * the agent's system prompt. Returns AgentState ready for store.prefill().
 *
 * @param parent - Branch to fork from
 * @param task - Agent task description
 * @param ctx - SessionContext for formatting and tokenization
 * @returns AgentState with branch and suffixTokens
 *
 * @example
 * ```typescript
 * const agent = await forkAgent(trunk, {
 *   systemPrompt: 'You are a research assistant.',
 *   content: 'What is X?',
 *   tools: toolsJson,
 *   seed: Date.now(),
 * }, ctx);
 * await store.prefill([[agent.branch, agent.suffixTokens]]);
 * ```
 *
 * @category Branching
 */
export async function forkAgent(
  parent: Branch,
  task: AgentTask,
  ctx: SessionContext
): Promise<AgentState> {
  const branch = await parent.fork();
  const messages = [
    { role: 'system', content: task.systemPrompt },
    { role: 'user', content: task.content },
  ];
  const fmtOpts = task.tools ? { tools: task.tools } : {};
  const fmt = await ctx.formatChat(JSON.stringify(messages), fmtOpts);
  const sep = ctx.getTurnSeparator();
  const suffixTokens = [...sep, ...await ctx.tokenize(fmt.prompt, false)];
  if (task.seed != null) branch.reseedSampler(task.seed);
  return {
    branch,
    suffixTokens,
    fmt: {
      format: fmt.format,
      reasoningFormat: fmt.reasoningFormat,
      thinkingForcedOpen: fmt.thinkingForcedOpen,
      parser: fmt.parser,
    },
    rawOutput: '',
    done: false,
    tokenCount: 0,
    toolCallCount: 0,
    turns: 0,
    findings: null,
  };
}

/**
 * Run agents in a batched three-phase tick loop
 *
 * Preserves the mechanical execution wins from BranchStore:
 * shared-prefix KV, batched decode, fire-and-forget tools, idle yield.
 *
 * @param agents - Array of AgentState (from forkAgent or manual construction)
 * @param opts - Configuration including store, ctx, executeTool, and callbacks
 * @returns Aggregate statistics
 *
 * @example
 * ```typescript
 * const result = await runAgents(agents, {
 *   store, ctx,
 *   executeTool: (name, args) => myToolDispatch(name, args),
 *   maxTurns: 6,
 *   onToolCall(ai, name, args) { console.log(`Agent ${ai}: ${name}`); },
 * });
 * ```
 *
 * @category Branching
 */
export async function runAgents(
  agents: AgentState[],
  opts: RunAgentsOptions
): Promise<RunAgentsResult> {
  const { store, ctx, executeTool, maxTurns = 6, onToolCall, onToolResult, onReport } = opts;
  const sep = ctx.getTurnSeparator();

  let steps = 0;
  let totalToolCalls = 0;
  const counters = {
    warmPrefillCalls: 0,
    warmPrefillBranches: 0,
    stalledTicks: 0,
    maxConcurrentTools: 0,
    idleTicks: 0,
  };

  const pendingTools = new Map<number, {
    promise: Promise<{ ai: number; prefillTokens: number[] | null }>;
    name: string;
  }>();

  function dispatchTool(ai: number, w: AgentState, tc: ParsedToolCall): void {
    let toolArgs: Record<string, unknown>;
    try { toolArgs = JSON.parse(tc.arguments); } catch { toolArgs = {}; }
    const callId = tc.id || `call_${w.toolCallCount}`;

    w.toolCallCount++;
    totalToolCalls++;
    w.turns++;

    if (onToolCall) onToolCall(ai, tc.name, tc.arguments);

    const promise = (async () => {
      try {
        const result = await executeTool(tc.name, toolArgs);
        const resultStr = JSON.stringify(result);

        if (onToolResult) onToolResult(ai, tc.name, resultStr);

        const { prompt } = await ctx.formatChat(
          JSON.stringify([
            { role: 'system', content: '' },
            { role: 'tool', content: resultStr, tool_call_id: callId },
          ])
        );
        const delta = await ctx.tokenize(prompt, false);
        return { ai, prefillTokens: [...sep, ...delta] as number[] | null };
      } catch (err) {
        w.done = true;
        w.findings = `Tool error: ${(err as Error).message}`;
        return { ai, prefillTokens: null };
      }
    })();

    pendingTools.set(ai, { promise, name: tc.name });
    counters.maxConcurrentTools = Math.max(counters.maxConcurrentTools, pendingTools.size);
  }

  for (;;) {
    // -- Phase 1: PRODUCE -- sample from active agents
    const entries: [Branch, number][] = [];
    for (let ai = 0; ai < agents.length; ai++) {
      const w = agents[ai];
      if (w.done || pendingTools.has(ai)) continue;

      const { token, text, isStop } = w.branch.produceSync();
      if (isStop) {
        const parsed = ctx.parseChatOutput(w.rawOutput, w.fmt.format, {
          reasoningFormat: w.fmt.reasoningFormat,
          thinkingForcedOpen: w.fmt.thinkingForcedOpen,
          parser: w.fmt.parser,
        });

        const tc = parsed.toolCalls[0];
        if (!tc || w.turns >= maxTurns) {
          w.done = true;
          if (!w.findings && parsed.content) w.findings = parsed.content;
          continue;
        }

        if (tc.name === 'report') {
          try { w.findings = JSON.parse(tc.arguments).findings; } catch { w.findings = tc.arguments; }
          w.done = true;
          w.toolCallCount++;
          totalToolCalls++;
          if (onToolCall) onToolCall(ai, 'report', tc.arguments);
          if (onReport) onReport(ai, w.findings!);
          continue;
        }

        // Fire-and-forget — dispatch tool without blocking the decode loop
        dispatchTool(ai, w, tc);
        w.rawOutput = '';
        continue;
      }

      entries.push([w.branch, token]);
      w.rawOutput += text;
      w.tokenCount++;
    }

    // -- Phase 2: COMMIT -- batch-decode produced tokens
    if (entries.length > 0) {
      await store.commit(entries);
      steps++;
    }

    // -- Phase 3: SETTLE -- non-blocking check for resolved tools
    const prefillPairs: [Branch, number[]][] = [];
    for (const [ai, info] of pendingTools) {
      const result = await Promise.race([info.promise, Promise.resolve(null)]);
      if (result !== null) {
        pendingTools.delete(ai);
        if (result.prefillTokens) {
          prefillPairs.push([agents[result.ai].branch, result.prefillTokens]);
        }
      }
    }

    if (prefillPairs.length > 0) {
      await store.prefill(prefillPairs);
      counters.warmPrefillCalls++;
      counters.warmPrefillBranches += prefillPairs.length;
    }

    // -- Termination + idle yield
    const allDone = agents.every((w) => w.done) && pendingTools.size === 0;
    if (allDone) break;

    if (entries.length === 0 && pendingTools.size > 0) {
      counters.stalledTicks++;
      if (prefillPairs.length === 0) {
        // Nothing produced, nothing settled — yield until a tool resolves
        await Promise.race([...pendingTools.values()].map((i) => i.promise));
        counters.idleTicks++;
      }
    }
  }

  const totalTokens = agents.reduce((s, w) => s + w.tokenCount, 0);
  return { totalTokens, totalToolCalls, steps, counters };
}
