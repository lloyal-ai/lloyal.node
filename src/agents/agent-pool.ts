import { resource, call, action, useScope, createSignal, spawn, each } from 'effection';
import type { Operation, Scope, Channel } from 'effection';
import type { Branch } from '../Branch';
import { CHAT_FORMAT_CONTENT_ONLY, CHAT_FORMAT_GENERIC, GrammarTriggerType, type GrammarTrigger, type ParsedToolCall, type SessionContext } from '../types';
import type { BranchStore } from '../BranchStore';
import { Ctx, Store, Events } from './context';
import { buildToolResultDelta } from './deltas';
import type { Tool } from './Tool';
import type {
  TraceToken,
  AgentTaskSpec,
  AgentPoolOptions,
  AgentPoolResult,
  AgentResult,
  AgentEvent,
} from './types';

// ── Internal agent state machine ───────────────────────────────
// generating → awaiting_tool → generating  (tool result prefilled)
// generating → done                         (stop + no tool call, or report)
// awaiting_tool → done                      (tool error)

type AgentInternalState = 'generating' | 'awaiting_tool' | 'done';

interface AgentInternal {
  id: number;           // = branch.handle
  branch: Branch;
  state: AgentInternalState;
  fmt: {
    format: number;
    reasoningFormat: number;
    thinkingForcedOpen: boolean;
    parser: string;
    grammar: string;
    grammarLazy: boolean;
    grammarTriggers: GrammarTrigger[];
  };
  rawOutput: string;
  tokenCount: number;
  toolCallCount: number;
  turns: number;
  graceUsed: boolean;
  findings: string | null;
  traceBuffer: TraceToken[];
}

interface SettledTool {
  agentId: number;
  prefillTokens: number[];
  toolName: string;
}

// Report tool schema — auto-injected into agent tools by setupAgent().
// useAgentPool() intercepts report calls (never dispatched to execute()).
const REPORT_SCHEMA = {
  type: 'function' as const,
  function: {
    name: 'report',
    description: 'Submit your final research findings. Call this when you have gathered enough information to answer the question.',
    parameters: {
      type: 'object',
      properties: { findings: { type: 'string', description: 'Your research findings and answer' } },
      required: ['findings'],
    },
  },
};

/** Inject report tool schema if tools are present and report isn't already defined. */
function ensureReportTool(toolsJson: string): string {
  const schemas = JSON.parse(toolsJson) as { type: string; function: { name: string } }[];
  if (schemas.some(s => s.function?.name === 'report')) return toolsJson;
  schemas.push(REPORT_SCHEMA);
  return JSON.stringify(schemas);
}

/**
 * Fork an agent from a parent branch with its own system prompt and task.
 *
 * Formats the agent's messages via `formatChat()`, tokenizes the suffix,
 * and optionally reseeds the sampler for stochastic diversity. When the
 * task has tools, the `report` tool schema is auto-injected if absent.
 */
async function setupAgent(
  parent: Branch,
  task: AgentTaskSpec,
  ctx: SessionContext,
): Promise<{ agent: AgentInternal; suffixTokens: number[] }> {
  const branch = await parent.fork();
  const messages = [
    { role: 'system', content: task.systemPrompt },
    { role: 'user', content: task.content },
  ];
  const tools = task.tools ? ensureReportTool(task.tools) : undefined;
  const fmtOpts = tools ? { tools } : {};
  const fmt = await ctx.formatChat(JSON.stringify(messages), fmtOpts);
  if (tools && (fmt.format === CHAT_FORMAT_CONTENT_ONLY || fmt.format === CHAT_FORMAT_GENERIC)) {
    throw new Error('Model does not support tool calling. Please use a model with native tool support (e.g. Qwen3, Llama 3.x, Mistral).');
  }
  const sep = ctx.getTurnSeparator();
  const suffixTokens = [...sep, ...await ctx.tokenize(fmt.prompt, false)];
  if (task.seed != null) branch.reseedSampler(task.seed);

  return {
    agent: {
      id: branch.handle,
      branch,
      state: 'generating',
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
      tokenCount: 0,
      toolCallCount: 0,
      turns: 0,
      graceUsed: false,
      findings: null,
      traceBuffer: [],
    },
    suffixTokens,
  };
}

/**
 * Concurrent agent generation loop as an Effection resource
 *
 * Runs N agents in parallel using a three-phase tick loop over shared
 * {@link BranchStore} infrastructure. Each agent forks from a parent
 * branch, generates tokens, invokes tools, and reports findings.
 *
 * **Three-phase tick loop:**
 * 1. **PRODUCE** — sample all active agents via `produceSync()` (no async gap)
 * 2. **COMMIT** — single GPU call via `store.commit()` for all produced tokens
 * 3. **SETTLE** — drain settled tool results, batch prefill, reset grammars
 *
 * Tool dispatch uses `scope.run()` for eager start — tool executions run as
 * children of the agent pool scope and are cancelled if the scope exits.
 *
 * **Resource semantics:** `provide()` suspends after all agents complete,
 * keeping branches alive so the caller can fork from them (e.g. for
 * verification). Branches are pruned in the finally block when the
 * scope exits.
 *
 * For automatic branch cleanup on return, use {@link runAgents} instead.
 *
 * @param opts - Pool configuration: tasks, tools, sampling params, max turns
 * @returns Agent pool result with per-agent findings and aggregate statistics
 *
 * @example Shared root with agent pool
 * ```typescript
 * const pool = yield* withSharedRoot(
 *   { systemPrompt: RESEARCH_PROMPT, tools: toolsJson },
 *   function*(root) {
 *     return yield* useAgentPool({
 *       tasks: questions.map(q => ({
 *         systemPrompt: RESEARCH_PROMPT,
 *         content: q,
 *         tools: toolsJson,
 *         parent: root,
 *       })),
 *       tools: toolMap,
 *       maxTurns: 6,
 *     });
 *   },
 * );
 * ```
 *
 * @category Agents
 */
export function useAgentPool(opts: AgentPoolOptions): Operation<AgentPoolResult> {
  return resource(function*(provide) {
    const ctx: SessionContext = yield* Ctx.expect();
    const store: BranchStore = yield* Store.expect();
    const events: Channel<AgentEvent, void> = yield* Events.expect();
    const scope: Scope = yield* useScope();

    // Bridge for onProgress callbacks — Signal is correct here (external callback).
    // A spawned forwarder drains the bridge into the Channel with proper scope context.
    const progressBridge = createSignal<AgentEvent, void>();
    yield* spawn(function*() {
      for (const ev of yield* each(progressBridge)) {
        yield* events.send(ev);
        yield* each.next();
      }
    });
    const { tasks, tools, maxTurns = 100, nCtx = 0, gracePrompt, trace = false } = opts;

    // ── Setup: fork branches, collect suffix tokens ──────────
    const agents: AgentInternal[] = [];
    const prefillSetup: [Branch, number[]][] = [];

    // try/finally wraps everything from agent creation through provide().
    // Agent branches are plain Branch objects (not Effection resources) —
    // their cleanup is manual. Placing it here guarantees any branch that
    // makes it into agents[] is pruned on ANY exit path: normal completion,
    // tick loop error, or scope cancellation.
    try {

    for (const task of tasks) {
      // Per-task parent for tree topology, or first task's parent as shared root
      const parent = task.parent;
      if (!parent) throw new Error('useAgentPool: each task must have a parent branch');

      const { agent, suffixTokens } = yield* call(() => setupAgent(parent, task, ctx));
      agents.push(agent);
      prefillSetup.push([agent.branch, suffixTokens]);
    }

    // Batch prefill all agent suffixes
    yield* call(() => store.prefill(prefillSetup));

    // ── Lazy grammar setup ───────────────────────────────────
    const applyLazyGrammar = (a: AgentInternal): void => {
      if (a.fmt.grammar && a.fmt.grammarLazy && a.fmt.grammarTriggers.length > 0) {
        const triggers = a.fmt.grammarTriggers.map(t => {
          if (t.type === GrammarTriggerType.WORD) {
            const nlIdx = t.value.indexOf('\n');
            if (nlIdx >= 0 && nlIdx < t.value.length - 1) {
              return { ...t, value: t.value.slice(0, nlIdx + 1) };
            }
          }
          return t;
        });
        a.branch.setGrammarLazy(a.fmt.grammar, triggers);
      }
    };
    for (const a of agents) applyLazyGrammar(a);

    // ── Tool dispatch coordination ───────────────────────────
    // Plain JS buffer: spawned tool tasks push synchronously on completion.
    // SETTLE drains with splice(0). Safe because generators are synchronous
    // between yields — spawns can only push at yield points (during COMMIT's
    // yield* call()), and SETTLE runs after COMMIT in the same tick.
    const settledBuffer: SettledTool[] = [];
    const agentById = new Map(agents.map(a => [a.id, a]));

    // Track pending tool count for idle detection
    let pendingToolCount = 0;

    // Resolve function for idle wake — set when all agents stall
    let wakeIdle: (() => void) | null = null;

    let steps = 0;
    let totalToolCalls = 0;
    const counters = {
      warmPrefillCalls: 0,
      warmPrefillBranches: 0,
      stalledTicks: 0,
      maxConcurrentTools: 0,
      idleTicks: 0,
    };

    function* dispatchTool(agent: AgentInternal, tc: ParsedToolCall): Operation<void> {
      let toolArgs: Record<string, unknown>;
      try { toolArgs = JSON.parse(tc.arguments); } catch { toolArgs = {}; }
      const callId = tc.id || `call_${agent.toolCallCount}`;

      agent.toolCallCount++;
      totalToolCalls++;
      agent.turns++;
      agent.state = 'awaiting_tool';

      yield* events.send({ type: 'agent:tool_call', agentId: agent.id, tool: tc.name, args: tc.arguments });

      const tool = tools.get(tc.name);
      pendingToolCount++;
      counters.maxConcurrentTools = Math.max(counters.maxConcurrentTools, pendingToolCount);

      // scope.run() — eager start, child of agent pool scope, cancelled if scope exits.
      // spawn() is lazy (Operation), but we're in a generator — scope.run() is eager.
      scope.run(function*() {
        try {
          const toolContext = {
            onProgress: (p: { filled: number; total: number }) => {
              // Signal bridge — onProgress is an external callback, Signal.send() is correct here.
              progressBridge.send({ type: 'agent:tool_progress', agentId: agent.id, tool: tc.name, filled: p.filled, total: p.total });
            },
          };

          const result: unknown = yield* call(() =>
            tool ? tool.execute(toolArgs, toolContext) : Promise.resolve({ error: `Unknown tool: ${tc.name}` })
          );
          const resultStr = JSON.stringify(result);
          yield* events.send({ type: 'agent:tool_result', agentId: agent.id, tool: tc.name, result: resultStr });

          const prefillTokens: number[] = yield* call(() => buildToolResultDelta(ctx, resultStr, callId));
          settledBuffer.push({ agentId: agent.id, prefillTokens, toolName: tc.name });
        } catch (err) {
          agent.state = 'done';
          agent.findings = `Tool error: ${(err as Error).message}`;
        } finally {
          pendingToolCount--;
          if (wakeIdle) { wakeIdle(); wakeIdle = null; }
        }
      });
    }

    // Context pressure thresholds (in tokens)
    const GRACE_RESERVE = 1024;    // room for grace prompt + report generation
    const CRITICAL_RESERVE = 128;  // absolute minimum — hard stop to prevent crash

    // ── Three-phase tick loop ────────────────────────────────
    for (;;) {
      // -- Phase 1: PRODUCE -- sample from active agents

      // Compute aggregate KV remaining once per tick.
      // All branches (including done, not yet pruned) share one nCtx-sized KV cache.
      // Shared prefix slots are counted once; divergent tails add per-branch.
      let kvRemaining = Infinity;
      if (nCtx > 0) {
        const positions = agents.map(a => a.branch.position);
        const sharedPrefix = Math.min(...positions);
        const totalKV = positions.reduce((s, p) => s + p, 0) - (positions.length - 1) * sharedPrefix;
        kvRemaining = nCtx - totalKV;
      }

      const entries: [Branch, number][] = [];
      for (const a of agents) {
        if (a.state !== 'generating') continue;

        // Critical context pressure — hard stop before produceSync to prevent llama_decode crash
        if (kvRemaining < CRITICAL_RESERVE) {
          a.state = 'done';
          yield* events.send({ type: 'agent:done', agentId: a.id });
          continue;
        }

        const { token, text, isStop } = a.branch.produceSync();
        if (isStop) {
          const parsed = ctx.parseChatOutput(a.rawOutput, a.fmt.format, {
            reasoningFormat: a.fmt.reasoningFormat,
            thinkingForcedOpen: a.fmt.thinkingForcedOpen,
            parser: a.fmt.parser,
          });

          const tc = parsed.toolCalls[0];
          if (!tc) {
            a.state = 'done';
            if (!a.findings && a.toolCallCount > 0 && parsed.content) {
              a.findings = parsed.content;
              yield* events.send({ type: 'agent:report', agentId: a.id, findings: a.findings });
            }
            yield* events.send({ type: 'agent:done', agentId: a.id });
            continue;
          }

          // Grace turn: context pressure or maxTurns reached, agent wants to call a tool
          // that isn't report. Inject gracePrompt so the agent can synthesize findings.
          // If grace already used (or no gracePrompt configured), hard-cut.
          const contextPressure = kvRemaining < GRACE_RESERVE;
          const shouldGrace = (a.turns >= maxTurns || contextPressure) && tc.name !== 'report';

          if (shouldGrace) {
            if (a.graceUsed || !gracePrompt) {
              a.state = 'done';
              yield* events.send({ type: 'agent:done', agentId: a.id });
              continue;
            }
            a.graceUsed = true;
            const callId = tc.id || `call_${a.toolCallCount}`;
            a.turns++;
            a.state = 'awaiting_tool';
            pendingToolCount++;
            scope.run(function*() {
              try {
                const prefillTokens: number[] = yield* call(() =>
                  buildToolResultDelta(ctx, JSON.stringify({ error: gracePrompt }), callId)
                );
                settledBuffer.push({ agentId: a.id, prefillTokens, toolName: tc.name });
              } finally {
                pendingToolCount--;
                if (wakeIdle) { wakeIdle(); wakeIdle = null; }
              }
            });
            a.rawOutput = '';
            continue;
          }

          // Report tool special case — reject if no prior research
          if (tc.name === 'report') {
            if (a.toolCallCount === 0) {
              const callId = tc.id || `call_${a.toolCallCount}`;
              const errorMsg = 'You must search or read the corpus before reporting. Use search, grep, or read_file first.';
              a.turns++;
              a.state = 'awaiting_tool';
              pendingToolCount++;
              scope.run(function*() {
                try {
                  const prefillTokens: number[] = yield* call(() =>
                    buildToolResultDelta(ctx, JSON.stringify({ error: errorMsg }), callId)
                  );
                  settledBuffer.push({ agentId: a.id, prefillTokens, toolName: tc.name });
                } finally {
                  pendingToolCount--;
                  if (wakeIdle) { wakeIdle(); wakeIdle = null; }
                }
              });
              a.rawOutput = '';
              continue;
            }
            try { a.findings = JSON.parse(tc.arguments).findings; } catch { a.findings = tc.arguments; }
            a.state = 'done';
            a.toolCallCount++;
            totalToolCalls++;
            yield* events.send({ type: 'agent:tool_call', agentId: a.id, tool: 'report', args: tc.arguments });
            yield* events.send({ type: 'agent:report', agentId: a.id, findings: a.findings! });
            yield* events.send({ type: 'agent:done', agentId: a.id });
            continue;
          }

          // Fire-and-forget — dispatch tool without blocking the decode loop
          yield* dispatchTool(a, tc);
          a.rawOutput = '';
          continue;
        }

        entries.push([a.branch, token]);
        a.rawOutput += text;
        a.tokenCount++;
        if (trace) {
          const entropy = a.branch.modelEntropy();
          const surprisal = a.branch.modelSurprisal(token);
          a.traceBuffer.push({ text, entropy, surprisal });
          yield* events.send({
            type: 'agent:produce', agentId: a.id, text, tokenCount: a.tokenCount,
            entropy, surprisal,
          });
        } else {
          yield* events.send({ type: 'agent:produce', agentId: a.id, text, tokenCount: a.tokenCount });
        }
      }

      // -- Phase 2: COMMIT -- batch-decode produced tokens
      if (entries.length > 0) {
        yield* call(() => store.commit(entries));
        steps++;
      }

      // -- Phase 3: SETTLE -- drain settled tool buffer, batch prefill
      const settled = settledBuffer.splice(0);
      if (settled.length > 0) {
        const prefillPairs: [Branch, number[]][] = [];
        const settledAgents: AgentInternal[] = [];

        for (const item of settled) {
          const a = agentById.get(item.agentId);
          if (!a || a.state === 'done') continue;
          prefillPairs.push([a.branch, item.prefillTokens]);
          settledAgents.push(a);
        }

        if (prefillPairs.length > 0) {
          yield* call(() => store.prefill(prefillPairs));
          counters.warmPrefillCalls++;
          counters.warmPrefillBranches += prefillPairs.length;

          // Only NOW transition state + reset grammar
          for (const a of settledAgents) {
            a.state = 'generating';
            a.rawOutput = '';
            applyLazyGrammar(a);
          }
        }
      }

      // -- Termination + idle yield
      const allDone = agents.every(a => a.state === 'done') && pendingToolCount === 0;
      if (allDone) break;

      if (entries.length === 0 && pendingToolCount > 0) {
        counters.stalledTicks++;
        if (settled.length === 0) {
          // Nothing produced, nothing settled — yield until a tool resolves
          yield* action<void>((resolve) => {
            wakeIdle = resolve;
            return () => { wakeIdle = null; };
          });
          counters.idleTicks++;
        }
      }
    }

    // ── Provide result — suspends, branches stay alive ───────
    const result: AgentPoolResult = {
      agents: agents.map(a => ({
        agentId: a.id,
        branch: a.branch,
        findings: a.findings,
        toolCallCount: a.toolCallCount,
        tokenCount: a.tokenCount,
        ppl: a.branch.perplexity,
        samplingPpl: a.branch.samplingPerplexity,
        trace: trace ? a.traceBuffer : undefined,
      })),
      totalTokens: agents.reduce((s, a) => s + a.tokenCount, 0),
      totalToolCalls,
      steps,
      counters,
    };

    yield* provide(result);

    } finally {
      // Structured cleanup: prune all agent branches when scope exits.
      // Covers setup errors, tick loop errors, and normal scope teardown
      // (provide() suspends via yield* suspend(), halting jumps to finally).
      for (const a of agents) {
        yield* call(() => a.branch.prune());
      }
    }
  });
}
