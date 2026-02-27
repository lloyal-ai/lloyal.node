import type { Branch } from './Branch';
import {
  GrammarTriggerType,
  type AgentState,
  type AgentTask,
  type ParsedToolCall,
  type RunAgentsOptions,
  type RunAgentsResult,
  type SessionContext,
} from './types';
import { buildToolResultDelta } from './Session';

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
    agentId: branch.handle,
    branch,
    suffixTokens,
    fmt: {
      format: fmt.format,
      reasoningFormat: fmt.reasoningFormat,
      thinkingForcedOpen: fmt.thinkingForcedOpen,
      parser: fmt.parser,
      grammar: fmt.grammar,
      grammarLazy: fmt.grammarLazy,
      grammarTriggers: fmt.grammarTriggers,
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
 *   onToolCall(agentId, name, args) { console.log(`Agent ${agentId}: ${name}`); },
 * });
 * ```
 *
 * @category Branching
 */
export async function runAgents(
  agents: AgentState[],
  opts: RunAgentsOptions
): Promise<RunAgentsResult> {
  const { store, ctx, executeTool, maxTurns = 100, onProduce, onToolCall, onToolResult, onToolProgress, onReport } = opts;

  let steps = 0;
  let totalToolCalls = 0;
  const counters = {
    warmPrefillCalls: 0,
    warmPrefillBranches: 0,
    stalledTicks: 0,
    maxConcurrentTools: 0,
    idleTicks: 0,
  };

  // Keyed by agentId (= branch handle) — stable across reordering
  const pendingTools = new Map<number, {
    promise: Promise<{ agentId: number; prefillTokens: number[] | null }>;
    name: string;
  }>();

  function dispatchTool(w: AgentState, tc: ParsedToolCall): void {
    let toolArgs: Record<string, unknown>;
    try { toolArgs = JSON.parse(tc.arguments); } catch { toolArgs = {}; }
    const callId = tc.id || `call_${w.toolCallCount}`;

    w.toolCallCount++;
    totalToolCalls++;
    w.turns++;

    if (onToolCall) onToolCall(w.agentId, tc.name, tc.arguments);

    const toolContext = onToolProgress ? {
      onProgress: (p: { filled: number; total: number }) => onToolProgress(w.agentId, tc.name, p),
    } : undefined;

    const promise = (async () => {
      try {
        const result = await executeTool(tc.name, toolArgs, toolContext);
        const resultStr = JSON.stringify(result);

        if (onToolResult) onToolResult(w.agentId, tc.name, resultStr);

        const prefillTokens = await buildToolResultDelta(ctx, resultStr, callId);
        return { agentId: w.agentId, prefillTokens: prefillTokens as number[] | null };
      } catch (err) {
        w.done = true;
        w.findings = `Tool error: ${(err as Error).message}`;
        return { agentId: w.agentId, prefillTokens: null };
      }
    })();

    pendingTools.set(w.agentId, { promise, name: tc.name });
    counters.maxConcurrentTools = Math.max(counters.maxConcurrentTools, pendingTools.size);
  }

  // Build agentId → index lookup for SETTLE phase
  const agentById = new Map(agents.map((w) => [w.agentId, w]));

  // Lazy grammar: unconstrained until trigger fires, then grammar-constrained.
  // Prevents Qwen3 from generating JSON tool calls instead of expected XML.
  //
  // Upstream triggers include tool_start (e.g. "<tool_call>\n<function="),
  // which fires AFTER the model has already committed to XML — useless when
  // the model diverges to JSON. Truncate WORD triggers to scope_start only
  // (e.g. "<tool_call>\n") so the grammar activates at the divergence point
  // and forces the correct format.
  const applyLazyGrammar = (w: AgentState): void => {
    if (w.fmt.grammar && w.fmt.grammarLazy && w.fmt.grammarTriggers.length > 0) {
      const triggers = w.fmt.grammarTriggers.map(t => {
        if (t.type === GrammarTriggerType.WORD) {
          const nlIdx = t.value.indexOf('\n');
          if (nlIdx >= 0 && nlIdx < t.value.length - 1) {
            return { ...t, value: t.value.slice(0, nlIdx + 1) };
          }
        }
        return t;
      });
      w.branch.setGrammarLazy(w.fmt.grammar, triggers);
    }
  };
  for (const w of agents) applyLazyGrammar(w);

  for (;;) {
    // -- Phase 1: PRODUCE -- sample from active agents
    const entries: [Branch, number][] = [];
    for (const w of agents) {
      if (w.done || pendingTools.has(w.agentId)) continue;

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
          // Accept content as findings only if agent did actual research
          if (!w.findings && w.toolCallCount > 0 && parsed.content) {
            w.findings = parsed.content;
            if (onReport) onReport(w.agentId, w.findings);
          }
          continue;
        }

        if (tc.name === 'report') {
          if (w.toolCallCount === 0) {
            // Reject report without prior research — force the agent to use tools first
            const callId = tc.id || `call_${w.toolCallCount}`;
            const errorMsg = 'You must search or read the corpus before reporting. Use search, grep, or read_file first.';
            w.turns++;
            const promise = (async () => {
              const prefillTokens = await buildToolResultDelta(ctx, JSON.stringify({ error: errorMsg }), callId);
              return { agentId: w.agentId, prefillTokens: prefillTokens as number[] | null };
            })();
            pendingTools.set(w.agentId, { promise, name: tc.name });
            w.rawOutput = '';
            continue;
          }
          try { w.findings = JSON.parse(tc.arguments).findings; } catch { w.findings = tc.arguments; }
          w.done = true;
          w.toolCallCount++;
          totalToolCalls++;
          if (onToolCall) onToolCall(w.agentId, 'report', tc.arguments);
          if (onReport) onReport(w.agentId, w.findings!);
          continue;
        }

        // Fire-and-forget — dispatch tool without blocking the decode loop
        dispatchTool(w, tc);
        w.rawOutput = '';
        continue;
      }

      entries.push([w.branch, token]);
      w.rawOutput += text;
      w.tokenCount++;
      if (onProduce) onProduce(w.agentId, text, w.tokenCount);
    }

    // -- Phase 2: COMMIT -- batch-decode produced tokens
    if (entries.length > 0) {
      await store.commit(entries);
      steps++;
    }

    // -- Phase 3: SETTLE -- non-blocking check for resolved tools
    const prefillPairs: [Branch, number[]][] = [];
    for (const [id, info] of pendingTools) {
      const result = await Promise.race([info.promise, Promise.resolve(null)]);
      if (result !== null) {
        pendingTools.delete(id);
        if (result.prefillTokens) {
          const w = agentById.get(result.agentId)!;
          prefillPairs.push([w.branch, result.prefillTokens]);
        }
      }
    }

    if (prefillPairs.length > 0) {
      await store.prefill(prefillPairs);
      counters.warmPrefillCalls++;
      counters.warmPrefillBranches += prefillPairs.length;

      // Reset lazy grammar — previous grammar consumed the tool call and is
      // now in a terminal state. Fresh grammar awaits the next trigger.
      for (const [branch] of prefillPairs) {
        const w = agents.find(a => a.branch === branch);
        if (w) applyLazyGrammar(w);
      }
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
