import * as fs from 'node:fs';
import { each } from 'effection';
import type { Operation, Signal } from 'effection';
import type { HarnessEvent, PhaseStats } from './harness.js';
import type { AgentPoolResult } from '../../dist/agents/index.js';

// ── Mode + color ─────────────────────────────────────────────────

let _jsonlMode = false;

export function setJsonlMode(on: boolean): void { _jsonlMode = on; }

const isTTY = process.stdout.isTTY;

export const c = isTTY ? {
  bold: '\x1b[1m', dim: '\x1b[2m', reset: '\x1b[0m',
  green: '\x1b[32m', cyan: '\x1b[36m', yellow: '\x1b[33m', red: '\x1b[31m',
} : { bold: '', dim: '', reset: '', green: '', cyan: '', yellow: '', red: '' };

// ── Primitives ───────────────────────────────────────────────────

let _statusText = '';

export function status(text: string): void {
  if (_jsonlMode || !isTTY) return;
  _statusText = text;
  process.stdout.write('\r\x1b[K' + text);
}

export function statusClear(): void {
  if (!_statusText) return;
  _statusText = '';
  process.stdout.write('\r\x1b[K');
}

export const log = (...a: unknown[]): void => {
  if (_jsonlMode) return;
  statusClear();
  console.log(...a);
};

function emit(event: string, data: Record<string, unknown>): void {
  if (_jsonlMode) console.log(JSON.stringify({ event, ...data }));
}

export const sec = (a: number, b: number): string => ((b - a) / 1000).toFixed(1);
export const pad = (s: unknown, n: number): string => String(s).padStart(n);
export const fmtSize = (bytes: number): string => bytes > 1e9
  ? (bytes / 1e9).toFixed(1) + ' GB'
  : (bytes / 1e6).toFixed(0) + ' MB';

// ── Display subscriber ──────────────────────────────────────────
// Spawned once in main.ts. Handles both AgentEvent (agent-level,
// from useAgentPool) and PhaseEvent (harness-level).

export interface DisplayOptions {
  model: string;
  reranker: string;
  agentCount: number;
  verifyCount: number;
  chunkCount: number;
}

export function* displaySubscriber(
  events: Signal<HarnessEvent, void>,
  opts: DisplayOptions,
): Operation<void> {
  // Agent label tracking — scoped to subscriber lifetime
  const agentLabel = new Map<number, string>();
  let nextLabel = 0;
  const agentText = new Map<number, string>();
  const agentStatus = new Map<number, { state: string; tokenCount: number; detail: string }>();

  function label(agentId: number): string {
    let l = agentLabel.get(agentId);
    if (!l) { l = `A${nextLabel++}`; agentLabel.set(agentId, l); }
    return l;
  }

  function resetLabels(): void {
    nextLabel = 0; agentLabel.clear(); agentStatus.clear(); agentText.clear();
  }

  function renderStatus(): void {
    const active = [...agentStatus.entries()].filter(([, s]) => s.state !== 'done');
    if (active.length === 0) return;

    const generating = active.filter(([, s]) => s.state === 'gen');

    if (generating.length === 1 && active.length === 1) {
      const [id] = generating[0];
      const raw = (agentText.get(id) ?? '').replace(/\n/g, ' ').trimStart();
      const cols = process.stdout.columns || 80;
      const maxLen = cols - 12;
      const text = raw.length > maxLen ? raw.slice(raw.length - maxLen) : raw;
      status(`    ${c.dim}\u25c6${c.reset} ${c.yellow}${label(id)}${c.reset} ${text}`);
      return;
    }

    const parts = active.map(([id, s]) => {
      const lbl = `${c.yellow}${label(id)}${c.reset}`;
      if (s.state === 'gen') return `${lbl}: ${s.tokenCount} tok`;
      const detail = s.detail ? ` ${s.detail}` : '';
      return `${lbl}: ${c.cyan}${s.state}${c.reset}${detail}`;
    });
    status(`    ${c.dim}\u25c6${c.reset} ${parts.join('  ')}`);
  }

  function renderStats(phases: PhaseStats[], kvLine?: string, ctxPct?: number, ctxPos?: number, ctxTotal?: number): void {
    const totalTokens = phases.reduce((s, p) => s + p.tokens, 0);
    const totalMs = phases.reduce((s, p) => s + p.timeMs, 0);

    log(`\n  ${c.dim}${'\u2501'.repeat(58)}${c.reset}`);
    for (const p of phases) {
      const left = `${p.label.padEnd(10)} ${pad(p.tokens, 5)} tok`;
      const detail = p.detail ? `  ${p.detail}` : '';
      const right = p.timeMs > 0 ? `${pad((p.timeMs / 1000).toFixed(1), 6)}s` : '';
      log(`  ${c.dim}${left}${detail}${' '.repeat(Math.max(1, 58 - left.length - detail.length - right.length))}${right}${c.reset}`);
    }
    log(`  ${c.dim}${'\u2501'.repeat(58)}${c.reset}`);
    log(`  ${c.bold}Total${c.reset}      ${c.bold}${pad(totalTokens, 5)}${c.reset} tok         ${c.bold}${pad((totalMs / 1000).toFixed(1), 6)}s${c.reset}`);
    if (kvLine) log(`  ${c.dim}${kvLine}${c.reset}`);
    if (ctxPct != null && ctxPos != null && ctxTotal != null) {
      const ctxStr = `ctx: ${ctxPct}% (${ctxPos.toLocaleString()}/${ctxTotal.toLocaleString()})`;
      log(`  ${c.dim}${'\u2501'.repeat(58)}${c.reset}`);
      log(`  ${c.dim}${' '.repeat(58 - ctxStr.length)}${ctxStr}${c.reset}`);
    }
    log();
  }

  // ── Trace persistence ────────────────────────────────────────
  // Per-token trace data lives on AgentResult.trace (populated by
  // useAgentPool when trace: true). We just write it to disk here.
  let traceQuery = '';

  function flushTrace(pool: AgentPoolResult): void {
    if (!pool.agents.some(a => a.trace?.length)) return;
    const filename = `trace-${Date.now()}.json`;
    fs.writeFileSync(filename, JSON.stringify({
      query: traceQuery,
      timestamp: new Date().toISOString(),
      agents: pool.agents.map(a => ({
        agentId: a.agentId, label: label(a.agentId),
        ppl: a.ppl, samplingPpl: a.samplingPpl,
        tokenCount: a.tokenCount, toolCallCount: a.toolCallCount,
        findings: a.findings, trace: a.trace ?? [],
      })),
    }, null, 2));
    log(`  ${c.dim}Trace written to ${filename}${c.reset}`);
  }

  for (const ev of yield* each(events)) {
    switch (ev.type) {
      // ── Agent-level events (from useAgentPool) ──────────
      case 'agent:produce': {
        agentText.set(ev.agentId, (agentText.get(ev.agentId) ?? '') + ev.text);
        agentStatus.set(ev.agentId, { state: 'gen', tokenCount: ev.tokenCount, detail: '' });
        renderStatus();
        break;
      }
      case 'agent:tool_call': {
        agentText.delete(ev.agentId);
        agentStatus.set(ev.agentId, { state: ev.tool, tokenCount: 0, detail: '' });
        emit('tool_call', { agentId: ev.agentId, toolName: ev.tool, arguments: ev.args });
        let toolArgs: Record<string, string>;
        try { toolArgs = JSON.parse(ev.args); } catch { toolArgs = {}; }
        const argSummary = ev.tool === 'search'
          ? `"${toolArgs.query || ''}"`
          : ev.tool === 'grep'
          ? `/${toolArgs.pattern || ''}/`
          : ev.tool === 'report' ? ''
          : `${toolArgs.filename}` + (toolArgs.startLine ? ` L${toolArgs.startLine}-${toolArgs.endLine}` : '');
        log(`    ${c.dim}\u251c${c.reset} ${c.yellow}${label(ev.agentId)}${c.reset} ${c.cyan}${ev.tool}${c.reset}${argSummary ? `(${argSummary})` : ''}`);
        break;
      }
      case 'agent:tool_result': {
        emit('tool_result', {
          agentId: ev.agentId, toolName: ev.tool,
          result: ev.result.length > 200 ? ev.result.slice(0, 200) + '...' : ev.result,
        });
        let preview = '';
        if (ev.tool === 'read_file') {
          try {
            const firstLine = (JSON.parse(ev.result) as { content: string }).content.split('\n').find((l: string) => l.trim());
            if (firstLine) preview = ` \u00b7 ${firstLine.trim().slice(0, 60)}${firstLine.trim().length > 60 ? '\u2026' : ''}`;
          } catch { /* non-fatal */ }
        } else if (ev.tool === 'search') {
          try {
            const top = (JSON.parse(ev.result) as { heading: string }[])[0];
            if (top?.heading) preview = ` \u00b7 ${top.heading}`;
          } catch { /* non-fatal */ }
        } else if (ev.tool === 'grep') {
          try {
            const r = JSON.parse(ev.result) as { totalMatches: number; matchingLines: number };
            preview = ` \u00b7 ${r.totalMatches} matches in ${r.matchingLines} lines`;
          } catch { /* non-fatal */ }
        }
        log(`    ${c.dim}\u251c${c.reset} ${c.yellow}${label(ev.agentId)}${c.reset} ${c.dim}\u2190 ${ev.tool} ${ev.result.length}b${preview}${c.reset}`);
        break;
      }
      case 'agent:tool_progress': {
        agentStatus.set(ev.agentId, { state: ev.tool, tokenCount: 0, detail: `${ev.filled}/${ev.total}` });
        renderStatus();
        break;
      }
      case 'agent:report': {
        agentStatus.set(ev.agentId, { state: 'done', tokenCount: 0, detail: '' });
        const cols = process.stdout.columns || 80;
        const lbl = `${c.yellow}${label(ev.agentId)}${c.reset}`;
        const prefix = `    ${c.dim}\u2502${c.reset}   `;
        const wrap = cols - 8;

        log(`    ${c.dim}\u2502${c.reset}`);
        log(`    ${c.dim}\u251c\u2500\u2500${c.reset} ${lbl} ${c.bold}findings${c.reset}`);

        for (const para of ev.findings.split('\n')) {
          if (!para.trim()) { log(prefix); continue; }
          const words = para.split(/\s+/);
          let line = '';
          for (const word of words) {
            if (line && line.length + 1 + word.length > wrap) {
              log(`${prefix}${c.dim}${line}${c.reset}`);
              line = word;
            } else {
              line = line ? `${line} ${word}` : word;
            }
          }
          if (line) log(`${prefix}${c.dim}${line}${c.reset}`);
        }
        log(`    ${c.dim}\u2502${c.reset}`);
        break;
      }
      case 'agent:done': break;

      // ── Phase events (from harness) ─────────────────────
      case 'query': {
        traceQuery = ev.query;
        if (!ev.warm) {
          emit('start', {
            model: opts.model, reranker: opts.reranker, query: ev.query,
            agentCount: opts.agentCount, verifyCount: opts.verifyCount, chunks: opts.chunkCount,
          });
          log();
          log(`  ${c.dim}Query${c.reset}`);
          log(`  ${c.bold}${ev.query}${c.reset}`);
        }
        break;
      }
      case 'plan': {
        emit('plan', { questions: ev.questions, planTokens: ev.tokenCount });
        log(`\n  ${c.green}\u25cf${c.reset} ${c.bold}Plan${c.reset} ${c.dim}${ev.tokenCount} tok \u00b7 ${(ev.timeMs / 1000).toFixed(1)}s${c.reset}`);
        ev.questions.forEach((q: string, i: number) => log(`    ${c.dim}${i + 1}.${c.reset} ${q}`));
        break;
      }
      case 'research:start': {
        log(`\n  ${c.green}\u25cf${c.reset} ${c.bold}Research${c.reset} ${c.dim}${ev.agentCount} agents${c.reset}`);
        resetLabels();
        break;
      }
      case 'research:done': {
        statusClear();
        ev.pool.agents.forEach((a, i) => {
          const tree = i === ev.pool.agents.length - 1 ? '\u2514' : '\u251c';
          emit('agent_done', {
            index: i, findings: (a.findings || '').slice(0, 500),
            toolCalls: a.toolCallCount, tokenCount: a.tokenCount,
            ppl: a.ppl, samplingPpl: a.samplingPpl,
          });
          const raw = (agentText.get(a.agentId) ?? '').replace(/\n/g, ' ').trim();
          if (raw) log(`    ${c.dim}\u251c${c.reset} ${c.yellow}${label(a.agentId)}${c.reset} ${c.dim}\u25b8 ${raw.slice(0, 120)}${raw.length > 120 ? '\u2026' : ''}${c.reset}`);
          const pplStr = Number.isFinite(a.ppl) ? ` \u00b7 ppl ${a.ppl.toFixed(2)}` : '';
          log(`    ${c.dim}${tree}${c.reset} ${c.yellow}${label(a.agentId)}${c.reset} ${c.green}done${c.reset} ${c.dim}${a.tokenCount} tok \u00b7 ${a.toolCallCount} tools${pplStr}${c.reset}`);
        });
        log(`    ${c.dim}${ev.pool.totalTokens} tok \u00b7 ${ev.pool.totalToolCalls} tools \u00b7 ${(ev.timeMs / 1000).toFixed(1)}s${c.reset}`);
        flushTrace(ev.pool);
        break;
      }
      case 'verify:start': {
        log(`\n  ${c.green}\u25cf${c.reset} ${c.bold}Verify${c.reset} ${c.dim}${ev.count} attempts${c.reset}`);
        break;
      }
      case 'verify:done': {
        ev.result.attempts.forEach((a, i) => {
          const tree = i === ev.result.attempts.length - 1 ? '\u2514' : '\u251c';
          emit('attempt_done', { index: i, output: a.output.trim().slice(0, 500), tokenCount: a.tokenCount, ppl: a.ppl });
          log(`    ${c.dim}${tree} ${a.tokenCount} tok \u00b7 ppl ${a.ppl.toFixed(2)}${c.reset}`);
        });
        log(`    ${c.dim}${ev.result.totalTokens} tok \u00b7 ${(ev.timeMs / 1000).toFixed(1)}s${c.reset}`);
        break;
      }
      case 'eval:done': {
        emit('convergence', { converged: ev.converged, evalTokens: ev.tokenCount });
        const verdict = ev.converged === true ? `${c.green}yes${c.reset}`
          : ev.converged === false ? `${c.red}no${c.reset}`
          : `${c.yellow}unknown${c.reset}`;
        log(`\n  ${c.green}\u25cf${c.reset} ${c.bold}Eval${c.reset} ${c.dim}${ev.tokenCount} tok \u00b7 ${(ev.timeMs / 1000).toFixed(1)}s${c.reset}`);
        log(`    Converged: ${verdict}`);
        break;
      }
      case 'answer': {
        log(`\n  ${c.dim}${'\u2500'.repeat(58)}${c.reset}\n`);
        const prose = ev.text.trim()
          .replace(/\*\*(.+?)\*\*/g, `${c.bold}$1${c.reset}`)
          .split('\n').map((l: string) => `  ${l}`).join('\n');
        log(prose);
        break;
      }
      case 'response:start': {
        process.stdout.write(`  ${c.dim}<${c.reset} `);
        break;
      }
      case 'response:text': {
        process.stdout.write(ev.text);
        break;
      }
      case 'response:done': {
        console.log('\n');
        break;
      }
      case 'stats': {
        renderStats(ev.phases, ev.kvLine, ev.ctxPct, ev.ctxPos, ev.ctxTotal);
        break;
      }
      case 'complete': {
        emit('complete', ev.data);
        break;
      }
    }
    yield* each.next();
  }
}
