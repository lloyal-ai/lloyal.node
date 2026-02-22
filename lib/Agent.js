/**
 * Agent - forkAgent + runAgents
 *
 * Two exported functions for the agentic loop pattern:
 * - forkAgent: fork from parent, format task, compute suffix tokens
 * - runAgents: three-phase tick loop (PRODUCE -> COMMIT -> SETTLE)
 *
 * Decoupled from Session — takes ctx directly, operates on agent branches.
 * Consumer wires tool dispatch, callbacks, and Session separately.
 */

/**
 * Fork an agent from a parent branch with its own system prompt + task.
 *
 * Always prepends getTurnSeparator() — forces clean break before agent's
 * system prompt. Returns AgentState ready for store.prefill().
 *
 * @param {Branch} parent - Branch to fork from
 * @param {{ systemPrompt: string, content: string, tools?: string, seed?: number }} task
 * @param {SessionContext} ctx
 * @returns {Promise<AgentState>}
 */
async function forkAgent(parent, task, ctx) {
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
 * Run agents in a batched three-phase tick loop.
 *
 * Mechanics preserved from runAgentSwarm:
 * - Three-phase tick: PRODUCE -> COMMIT -> SETTLE
 * - Fire-and-forget tool dispatch (tools run while other agents generate)
 * - Warm prefill with sep + delta when tools resolve
 * - `report` tool as completion signal (not dispatched to executeTool)
 * - Non-blocking settle via Promise.race
 * - Idle yield when all active agents are pending tools
 *
 * @param {AgentState[]} agents
 * @param {{
 *   store: BranchStore,
 *   ctx: SessionContext,
 *   executeTool: (name: string, args: object) => Promise<any>,
 *   maxTurns?: number,
 *   onToolCall?: (agentIndex: number, toolName: string, args: string) => void,
 *   onToolResult?: (agentIndex: number, toolName: string, resultStr: string) => void,
 *   onReport?: (agentIndex: number, findings: string) => void,
 * }} opts
 * @returns {Promise<{ totalTokens: number, totalToolCalls: number, steps: number, counters: object }>}
 */
async function runAgents(agents, opts) {
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

  // pendingTools: Map<agentIndex, { promise, name }>
  const pendingTools = new Map();

  function dispatchTool(ai, w, tc) {
    let toolArgs;
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

        // Format warm prefill tokens — the assistant's tool-call turn is
        // already in KV from generation; sep closes it.
        const { prompt } = await ctx.formatChat(
          JSON.stringify([
            { role: 'system', content: '' },
            { role: 'tool', content: resultStr, tool_call_id: callId },
          ])
        );
        const delta = await ctx.tokenize(prompt, false);
        return { ai, prefillTokens: [...sep, ...delta] };
      } catch (err) {
        w.done = true;
        w.findings = `Tool error: ${err.message}`;
        return { ai, prefillTokens: null };
      }
    })();

    pendingTools.set(ai, { promise, name: tc.name });
    counters.maxConcurrentTools = Math.max(counters.maxConcurrentTools, pendingTools.size);
  }

  for (;;) {
    // -- Phase 1: PRODUCE -- sample from active agents
    const entries = [];
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
          if (onReport) onReport(ai, w.findings);
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
    const prefillPairs = [];
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

module.exports = { forkAgent, runAgents };
