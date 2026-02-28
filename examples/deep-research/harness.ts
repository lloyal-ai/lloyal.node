import * as fs from 'node:fs';
import * as path from 'node:path';
import { call } from 'effection';
import type { Operation, Signal } from 'effection';
import { Branch, Session } from '../../dist/index.js';
import type { SessionContext } from '../../dist/index.js';
import {
  Ctx,
  generate, runAgents, diverge, withSharedRoot,
} from '../../dist/agents/index.js';
import type { Tool, AgentPoolResult, DivergeResult, AgentEvent } from '../../dist/agents/index.js';

/** Load a task prompt file. Convention: system prompt above `---`, user content below. */
function loadTask(name: string): { system: string; user: string } {
  const raw = fs.readFileSync(path.resolve(__dirname, `tasks/${name}.md`), 'utf8').trim();
  const sep = raw.indexOf('\n---\n');
  if (sep === -1) return { system: raw, user: '' };
  return { system: raw.slice(0, sep).trim(), user: raw.slice(sep + 5).trim() };
}

const PLAN = loadTask('plan');
const RESEARCH = loadTask('research');
const VERIFY = loadTask('verify');
const EVAL = loadTask('eval');

// ── Harness events ───────────────────────────────────────────────
// Phase-level events sent by the harness. Display subscribes to these
// alongside AgentEvent (agent-level events from useAgentPool).

export interface PhaseStats {
  label: string;
  tokens: number;
  detail: string;
  timeMs: number;
}

export type PhaseEvent =
  | { type: 'query'; query: string; warm: boolean }
  | { type: 'plan'; questions: string[]; tokenCount: number; timeMs: number }
  | { type: 'research:start'; agentCount: number }
  | { type: 'research:done'; pool: AgentPoolResult; sharedPrefixLength: number; timeMs: number }
  | { type: 'verify:start'; count: number }
  | { type: 'verify:done'; result: DivergeResult; timeMs: number }
  | { type: 'eval:done'; converged: boolean | null; tokenCount: number; timeMs: number }
  | { type: 'answer'; text: string }
  | { type: 'response:start' }
  | { type: 'response:text'; text: string }
  | { type: 'response:done'; tokenCount: number; timeMs: number }
  | { type: 'stats'; phases: PhaseStats[]; kvLine?: string; ctxPct: number; ctxPos: number; ctxTotal: number }
  | { type: 'complete'; data: Record<string, unknown> };

export type HarnessEvent = AgentEvent | PhaseEvent;

// ── Options ──────────────────────────────────────────────────────

export interface HarnessOptions {
  session: Session;
  toolMap: Map<string, Tool>;
  toolsJson: string;
  agentCount: number;
  verifyCount: number;
  maxTurns: number;
  nCtx: number;
  trace: boolean;
  events: Signal<HarnessEvent, void>;
}

// ── Plan ─────────────────────────────────────────────────────────

function* planPhase(
  query: string,
  agentCount: number,
  parent?: Branch,
): Operation<{ questions: string[]; tokenCount: number }> {
  const ctx: SessionContext = yield* Ctx.expect();

  const schema = {
    type: 'object',
    properties: {
      questions: {
        type: 'array',
        items: { type: 'string' },
        minItems: 2,
        maxItems: agentCount,
      },
    },
    required: ['questions'],
  };
  const grammar: string = yield* call(() => ctx.jsonSchemaToGrammar(JSON.stringify(schema)));

  const userContent = PLAN.user
    .replace('{{count}}', String(agentCount))
    .replace('{{query}}', query);

  const messages = [
    { role: 'system', content: PLAN.system },
    { role: 'user', content: userContent },
  ];
  const { prompt }: { prompt: string } = yield* call(() => ctx.formatChat(JSON.stringify(messages)));

  let output: string;
  let tokenCount: number;

  if (parent) {
    // Warm: fork from trunk — planner inherits conversation KV
    const lead: Branch = yield* call(() => parent.fork());
    try {
      lead.setGrammar(grammar);
      const sep = ctx.getTurnSeparator();
      const delta: number[] = yield* call(() => ctx.tokenize(prompt, false));
      yield* call(() => lead.prefill([...sep, ...delta]));

      ({ output, tokenCount } = yield* call(async () => {
        let o = '';
        let tc = 0;
        for await (const { text } of lead) { o += text; tc++; }
        return { output: o, tokenCount: tc };
      }));
    } finally {
      if (!lead.disposed) yield* call(() => lead.prune());
    }
  } else {
    // Cold: fresh branch via generate()
    const result = yield* generate({ prompt, grammar, params: { temperature: 0.3 } });
    output = result.output;
    tokenCount = result.tokenCount;
  }

  let questions: string[];
  try {
    questions = JSON.parse(output).questions.slice(0, agentCount);
    if (!questions.length) throw new Error('empty');
  } catch {
    questions = Array.from({ length: agentCount }, (_, i) => `${query} (aspect ${i + 1})`);
  }

  return { questions, tokenCount };
}

// ── Verify ───────────────────────────────────────────────────────

function* verifyPhase(opts: {
  findings: string;
  query: string;
  count: number;
}): Operation<DivergeResult> {
  const ctx: SessionContext = yield* Ctx.expect();

  const userContent = VERIFY.user
    .replace('{{findings}}', opts.findings)
    .replace('{{query}}', opts.query);

  const messages = [
    { role: 'system', content: VERIFY.system },
    { role: 'user', content: userContent },
  ];
  const { prompt }: { prompt: string } = yield* call(() => ctx.formatChat(JSON.stringify(messages)));

  return yield* diverge({
    prompt,
    attempts: opts.count,
    params: { temperature: 0.7 },
  });
}

// ── Eval ─────────────────────────────────────────────────────────

function* evalPhase(
  attempts: { output: string }[],
): Operation<{ converged: boolean | null; tokenCount: number }> {
  const ctx: SessionContext = yield* Ctx.expect();

  const responsesText = attempts
    .map((a, i) => `Response ${i + 1}: ${a.output.trim()}`)
    .join('\n\n');

  const userContent = EVAL.user.replace('{{responses}}', responsesText);

  const messages = [
    { role: 'system', content: EVAL.system },
    { role: 'user', content: userContent },
  ];

  const evalSchema = {
    type: 'object',
    properties: { converged: { type: 'boolean' } },
    required: ['converged'],
  };
  const grammar: string = yield* call(() => ctx.jsonSchemaToGrammar(JSON.stringify(evalSchema)));
  const { prompt }: { prompt: string } = yield* call(() => ctx.formatChat(JSON.stringify(messages)));

  const result = yield* generate({
    prompt,
    grammar,
    params: { temperature: 0 },
    parse: (output: string) => {
      try { return JSON.parse(output).converged as boolean; }
      catch { return null; }
    },
  });

  return { converged: result.parsed as boolean | null, tokenCount: result.tokenCount };
}

// ── handleQuery — the orchestrator ───────────────────────────────
// Composes phases, sends HarnessEvent for display, touches no log().

export function* handleQuery(query: string, opts: HarnessOptions): Operation<void> {
  const { session, toolMap, toolsJson, agentCount, verifyCount, maxTurns, nCtx, trace, events } = opts;
  const warm = !!session.trunk;
  const t0 = performance.now();

  events.send({ type: 'query', query, warm });

  // ── Plan
  let t = performance.now();
  const { questions, tokenCount: planTokens } = yield* planPhase(
    query, agentCount, warm ? session.trunk! : undefined,
  );
  const planMs = performance.now() - t;
  events.send({ type: 'plan', questions, tokenCount: planTokens, timeMs: planMs });

  // ── Research
  events.send({ type: 'research:start', agentCount: questions.length });
  t = performance.now();

  let pool: AgentPoolResult;
  let sharedPrefixLength: number;

  const agentTasks = (parent: Branch, seed?: number) => questions.map((q, i) => ({
    systemPrompt: RESEARCH.system,
    content: q,
    tools: toolsJson,
    parent,
    seed: seed != null ? seed + i : undefined,
  }));

  if (!warm) {
    // Cold: withSharedRoot handles root create → prefill → cleanup
    const { result, prefixLen } = yield* withSharedRoot(
      { systemPrompt: RESEARCH.system, tools: toolsJson },
      function*(root, prefixLen) {
        const result = yield* runAgents({ tasks: agentTasks(root), tools: toolMap, maxTurns, trace });
        return { result, prefixLen };
      },
    );
    pool = result;
    sharedPrefixLength = prefixLen;
  } else {
    // Warm: fork from conversation trunk
    pool = yield* runAgents({
      tasks: agentTasks(session.trunk!, Date.now()),
      tools: toolMap,
      maxTurns, trace,
    });
    sharedPrefixLength = 0;
  }

  const researchMs = performance.now() - t;
  events.send({ type: 'research:done', pool, sharedPrefixLength, timeMs: researchMs });

  // ── Post-research diverges based on cold/warm
  const phases: PhaseStats[] = [
    { label: 'Plan', tokens: planTokens, detail: '', timeMs: planMs },
    {
      label: 'Research', tokens: pool.totalTokens,
      detail: `(${pool.agents.map(a => a.tokenCount).join(' + ')})  ${pool.totalToolCalls} tools`,
      timeMs: researchMs,
    },
  ];

  if (!warm) {
    // ── Verify
    const findingsText = pool.agents
      .map((a, i) => `Q: ${questions[i]}\nA: ${(a.findings || '').trim()}`)
      .join('\n\n');

    events.send({ type: 'verify:start', count: verifyCount });
    t = performance.now();
    const verifyResult = yield* verifyPhase({ findings: findingsText, query, count: verifyCount });
    const verifyMs = performance.now() - t;
    events.send({ type: 'verify:done', result: verifyResult, timeMs: verifyMs });

    // ── Eval
    t = performance.now();
    const { converged, tokenCount: evalTokens } = yield* evalPhase(verifyResult.attempts);
    const evalMs = performance.now() - t;
    events.send({ type: 'eval:done', converged, tokenCount: evalTokens, timeMs: evalMs });

    // ── Answer
    events.send({ type: 'answer', text: verifyResult.bestOutput });

    phases.push(
      {
        label: 'Verify', tokens: verifyResult.totalTokens,
        detail: `(${verifyResult.attempts.map(a => a.tokenCount).join(' + ')})`,
        timeMs: verifyMs,
      },
      { label: 'Eval', tokens: evalTokens, detail: `converged: ${converged ? 'yes' : 'no'}`, timeMs: evalMs },
    );

    yield* call(() => session.promote(verifyResult.best));

    const kvSaved = sharedPrefixLength * (questions.length - 1)
      + verifyResult.prefixLength * (verifyResult.attempts.length - 1);

    events.send({
      type: 'stats', phases,
      kvLine: `KV shared    ${sharedPrefixLength} \u00d7 ${questions.length - 1} + ${verifyResult.prefixLength} \u00d7 ${verifyResult.attempts.length - 1} = ${kvSaved.toLocaleString()} tok saved`,
      ctxPct: Math.round(100 * (session.trunk?.position ?? 0) / nCtx),
      ctxPos: session.trunk?.position ?? 0,
      ctxTotal: nCtx,
    });

    events.send({
      type: 'complete',
      data: {
        planTokens,
        agentTokens: pool.totalTokens, researchSteps: pool.steps,
        agentPpl: pool.agents.map(a => a.ppl),
        verifyTokens: verifyResult.totalTokens, verifySteps: verifyResult.steps,
        evalTokens, converged,
        totalToolCalls: pool.totalToolCalls,
        prefixTokens: verifyResult.prefixLength,
        sharedPrefixTokens: sharedPrefixLength,
        agentCount: questions.length, attemptCount: verifyResult.attempts.length,
        wallTimeMs: Math.round(performance.now() - t0),
        planMs: Math.round(planMs), researchMs: Math.round(researchMs),
        verifyMs: Math.round(verifyMs), evalMs: Math.round(evalMs),
        ...pool.counters,
      },
    });

  } else {
    // ── Grounded response from trunk
    const agentFindings = pool.agents
      .map((a: { findings: string | null }, i: number) =>
        a.findings ? `[Agent ${i}] ${a.findings.trim()}` : null)
      .filter(Boolean)
      .join('\n\n');

    yield* call(() => session.prefillUser(agentFindings
      ? `Research findings:\n${agentFindings}\n\nUser question: ${query}\n\nAnswer based on the research findings above.`
      : query));

    events.send({ type: 'response:start' });
    t = performance.now();
    let responseTokens = 0;
    const trunk = session.trunk!;
    for (;;) {
      const { token, text, isStop } = trunk.produceSync();
      if (isStop) break;
      yield* call(() => trunk.commit(token));
      responseTokens++;
      events.send({ type: 'response:text', text } as HarnessEvent);
    }
    const responseMs = performance.now() - t;
    events.send({ type: 'response:done', tokenCount: responseTokens, timeMs: responseMs });

    phases.push({ label: 'Response', tokens: responseTokens, detail: '', timeMs: responseMs });

    events.send({
      type: 'stats', phases,
      ctxPct: Math.round(100 * (session.trunk?.position ?? 0) / nCtx),
      ctxPos: session.trunk?.position ?? 0,
      ctxTotal: nCtx,
    });
  }
}
