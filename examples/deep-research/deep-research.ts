#!/usr/bin/env node
/**
 * Deep Research with Tool-Calling Agents via BranchStore
 *
 * Demonstrates composable fork patterns in a multi-agent research pipeline:
 *
 * - PLAN:     Branch.create() + grammar — constrained single generation
 * - RESEARCH: fork() + prefill() divergent suffixes — parallel tool-calling agents
 * - VERIFY:   fork() + reseed() — stochastic divergence for convergence checking
 * - EVAL:     Branch.create() + grammar — model-as-judge
 *
 * Cold run composes: plan → research → verify → eval
 * Warm follow-up composes: research(parent: trunk) → session.prefillUser → generate
 *
 * Usage:
 *   node deep-research.ts [model-path] --corpus <path> [--query <text>] [options]
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as readline from 'node:readline';
import { createContext, BranchStore, Session } from '../../dist/index.js';
import { c, log, emit, setJsonlMode, status, statusClear, pad, fmtSize } from './display.js';
import { loadResources, chunkResources } from './resources/files.js';
import { createReranker } from './reranker.js';
import { createTools } from './tools/index.js';
import { plan } from './tasks/plan.js';
import { research } from './tasks/research.js';
import { verify } from './tasks/verify.js';
import { evaluate } from './tasks/eval.js';

// ================================================================
// CLI ARGS
// ================================================================

const DEFAULT_MODEL = path.resolve(
  __dirname,
  '../../models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
);
const DEFAULT_RERANKER = path.resolve(
  __dirname,
  '../../models/qwen3-reranker-0.6b-q4_k_m.gguf'
);

const args = process.argv.slice(2);
const jsonlMode = args.includes('--jsonl');
const verbose = args.includes('--verbose');

function argVal(flag: string): string | null {
  const i = args.indexOf(flag);
  return i !== -1 ? args[i + 1] : null;
}
const flagIndices = new Set(
  ['--reranker', '--corpus', '--query'].flatMap((f) => {
    const i = args.indexOf(f);
    return i !== -1 ? [i, i + 1] : [];
  })
);

const rerankModelPath = argVal('--reranker') || DEFAULT_RERANKER;
const corpusDir = argVal('--corpus');
const initialQuery = argVal('--query');
const modelPath = args.find((a, i) =>
  !a.startsWith('--') && !flagIndices.has(i)
) || DEFAULT_MODEL;

if (!corpusDir) {
  process.stdout.write(
    `Usage: node deep-research.ts [model-path] --corpus <path> [--query <text>] [--reranker <path>]\n` +
    `Missing: --corpus\n`
  );
  process.exit(1);
}

if (jsonlMode) setJsonlMode(true);

// Suppress native llama.cpp logs for clean output
if (!verbose && !jsonlMode) {
  try {
    fs.closeSync(2);
    fs.openSync(process.platform === 'win32' ? '\\\\.\\NUL' : '/dev/null', 'w');
  } catch { /* non-fatal */ }
}

const AGENT_COUNT = 3;
const VERIFY_COUNT = 3;
const MAX_TOOL_TURNS = 6;

// ================================================================
// MAIN
// ================================================================

async function main(): Promise<void> {
  // Resources
  const resources = loadResources(corpusDir!);
  const chunks = chunkResources(resources);

  const modelName = path.basename(modelPath).replace(/-Q\w+\.gguf$/, '');
  const rerankName = path.basename(rerankModelPath).replace(/-q\w+\.gguf$/i, '');
  const modelSize = fmtSize(fs.statSync(modelPath).size);
  const rerankSize = fmtSize(fs.statSync(rerankModelPath).size);

  log();
  log(`${c.bold}  Deep Research${c.reset} ${c.dim}— BranchStore Tool-Calling Agents${c.reset}`);
  log();

  log(`  ${c.green}●${c.reset} Loading ${c.bold}${modelName}${c.reset} ${c.dim}(${modelSize}, KV: Q4_0)${c.reset}`);

  const nCtx = parseInt(process.env.LLAMA_CTX_SIZE || '16384', 10);
  const ctx = await createContext({
    modelPath, nCtx,
    nSeqMax: Math.max(AGENT_COUNT, VERIFY_COUNT) + 1,
    typeK: 'q4_0', typeV: 'q4_0',
  });

  log(`  ${c.green}●${c.reset} Loading ${c.bold}${rerankName}${c.reset} ${c.dim}(${rerankSize}, reranker)${c.reset}`);

  const reranker = await createReranker(rerankModelPath, { nSeqMax: 8, nCtx: 4096 });
  await reranker.tokenizeChunks(chunks);

  const corpusIsFile = resources.length === 1 && fs.statSync(corpusDir!).isFile();
  const corpusLabel = corpusIsFile
    ? path.basename(corpusDir!)
    : `${path.basename(corpusDir!)}/ — ${resources.length} files`;
  log(`  ${c.dim}  Corpus: ${corpusLabel} → ${chunks.length} chunks${c.reset}`);

  const { toolsJson, executeTool } = createTools({ resources, chunks, reranker });
  const store = new BranchStore(ctx);
  const session = new Session({ ctx, store });

  // ── Agent labels + status line ──────────────────────────
  const agentLabel = new Map<number, string>();
  let nextLabel = 0;
  function label(agentId: number): string {
    let l = agentLabel.get(agentId);
    if (!l) { l = `A${nextLabel++}`; agentLabel.set(agentId, l); }
    return l;
  }
  const agentText = new Map<number, string>();    // accumulated raw text per agent
  function resetLabels(): void { nextLabel = 0; agentLabel.clear(); agentStatus.clear(); agentText.clear(); }

  const agentStatus = new Map<number, { state: string; tokenCount: number; detail: string }>();

  function renderStatus(): void {
    const active = [...agentStatus.entries()].filter(([, s]) => s.state !== 'done');
    if (active.length === 0) return;

    const generating = active.filter(([, s]) => s.state === 'gen');

    // Single agent generating → stream text on status line (rewritable — clears
    // when tool call fires or agent finishes)
    if (generating.length === 1 && active.length === 1) {
      const [id] = generating[0];
      const raw = (agentText.get(id) ?? '').replace(/\n/g, ' ').trimStart();
      const cols = process.stdout.columns || 80;
      const maxLen = cols - 12;  // "    ◆ A0 " prefix ≈ 9 visible chars + margin
      const text = raw.length > maxLen ? raw.slice(raw.length - maxLen) : raw;
      status(`    ${c.dim}\u25c6${c.reset} ${c.yellow}${label(id)}${c.reset} ${text}`);
      return;
    }

    // Multi-agent: compact counters
    const parts = active.map(([id, s]) => {
      const lbl = `${c.yellow}${label(id)}${c.reset}`;
      if (s.state === 'gen') return `${lbl}: ${s.tokenCount} tok`;
      const detail = s.detail ? ` ${s.detail}` : '';
      return `${lbl}: ${c.cyan}${s.state}${c.reset}${detail}`;
    });
    status(`    ${c.dim}\u25c6${c.reset} ${parts.join('  ')}`);
  }

  // ── Callbacks — shared across cold + warm paths ────────
  const onProduce = (agentId: number, text: string, tokenCount: number): void => {
    agentText.set(agentId, (agentText.get(agentId) ?? '') + text);
    agentStatus.set(agentId, { state: 'gen', tokenCount, detail: '' });
    renderStatus();
  };

  const onToolProgress = (agentId: number, toolName: string, p: { filled: number; total: number }): void => {
    agentStatus.set(agentId, { state: toolName, tokenCount: 0, detail: `${p.filled}/${p.total}` });
    renderStatus();
  };

  const onToolCall = (agentId: number, toolName: string, argsStr: string): void => {
    agentText.delete(agentId);  // this generation led to a parsed tool call — clear
    agentStatus.set(agentId, { state: toolName, tokenCount: 0, detail: '' });
    emit('tool_call', { agentId, toolName, arguments: argsStr });
    let toolArgs: Record<string, string>;
    try { toolArgs = JSON.parse(argsStr); } catch { toolArgs = {}; }
    const argSummary = toolName === 'search'
      ? `"${toolArgs.query || ''}"`
      : toolName === 'grep'
      ? `/${toolArgs.pattern || ''}/`
      : toolName === 'report' ? ''
      : `${toolArgs.filename}` + (toolArgs.startLine ? ` L${toolArgs.startLine}-${toolArgs.endLine}` : '');
    log(`    ${c.dim}\u251c${c.reset} ${c.yellow}${label(agentId)}${c.reset} ${c.cyan}${toolName}${c.reset}${argSummary ? `(${argSummary})` : ''}`);
  };

  const onToolResult = (agentId: number, toolName: string, resultStr: string): void => {
    emit('tool_result', {
      agentId, toolName,
      result: resultStr.length > 200 ? resultStr.slice(0, 200) + '...' : resultStr,
    });
    let preview = '';
    if (toolName === 'read_file') {
      try {
        const firstLine = (JSON.parse(resultStr) as { content: string }).content.split('\n').find(l => l.trim());
        if (firstLine) preview = ` · ${firstLine.trim().slice(0, 60)}${firstLine.trim().length > 60 ? '\u2026' : ''}`;
      } catch { /* non-fatal */ }
    } else if (toolName === 'search') {
      try {
        const top = (JSON.parse(resultStr) as { heading: string }[])[0];
        if (top?.heading) preview = ` · ${top.heading}`;
      } catch { /* non-fatal */ }
    } else if (toolName === 'grep') {
      try {
        const r = JSON.parse(resultStr) as { totalMatches: number; matchingLines: number };
        preview = ` · ${r.totalMatches} matches in ${r.matchingLines} lines`;
      } catch { /* non-fatal */ }
    }
    log(`    ${c.dim}\u251c${c.reset} ${c.yellow}${label(agentId)}${c.reset} ${c.dim}\u2190 ${toolName} ${resultStr.length}b${preview}${c.reset}`);
  };

  const onReport = (agentId: number, findings: string): void => {
    agentStatus.set(agentId, { state: 'done', tokenCount: 0, detail: '' });
    const cols = process.stdout.columns || 80;
    const lbl = `${c.yellow}${label(agentId)}${c.reset}`;
    const prefix = `    ${c.dim}\u2502${c.reset}   `;
    // visible width: "    │   " = 8 chars
    const wrap = cols - 8;

    log(`    ${c.dim}\u2502${c.reset}`);
    log(`    ${c.dim}\u251c\u2500\u2500${c.reset} ${lbl} ${c.bold}findings${c.reset}`);

    // Word-wrap findings, preserve paragraph breaks
    for (const para of findings.split('\n')) {
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
  };

  // ================================================================
  // handleQuery — the orchestrator
  //
  // No session yet → cold: plan → research → verify → eval
  // Session exists → warm: research(parent: trunk) → prefillUser → generate
  // ================================================================

  async function handleQuery(query: string): Promise<void> {
    const t0 = performance.now();
    const warm = !!session.trunk;

    if (!warm) {
      emit('start', {
        model: path.basename(modelPath), reranker: path.basename(rerankModelPath),
        query, agentCount: AGENT_COUNT, verifyCount: VERIFY_COUNT, chunks: chunks.length,
      });
      log();
      log(`  ${c.dim}Query${c.reset}`);
      log(`  ${c.bold}${query}${c.reset}`);
    }

    // ─── plan ─────────────────────────────────────────────
    let t = performance.now();
    const { questions, tokenCount: planTokens } = await plan(ctx, {
      query, agentCount: AGENT_COUNT,
      ...(warm && { parent: session.trunk! }),
    });
    const planMs = performance.now() - t;

    if (!warm) emit('plan', { questions, planTokens });
    log(`\n  ${c.green}●${c.reset} ${c.bold}Plan${c.reset} ${c.dim}${planTokens} tok · ${(planMs / 1000).toFixed(1)}s${c.reset}`);
    questions.forEach((q, i) => log(`    ${c.dim}${i + 1}.${c.reset} ${q}`));

    // ─── research ─────────────────────────────────────────
    t = performance.now();
    log(`\n  ${c.green}●${c.reset} ${c.bold}Research${c.reset} ${c.dim}${questions.length} agents${c.reset}`);

    resetLabels();
    const researchResult = await research(ctx, store, {
      questions, toolsJson, executeTool,
      maxTurns: MAX_TOOL_TURNS,
      onProduce, onToolCall, onToolResult, onToolProgress, onReport,
      ...(warm && { parent: session.trunk!, seed: Date.now() }),
    });
    statusClear();
    const researchMs = performance.now() - t;

    researchResult.agents.forEach((a, i) => {
      const tree = i === researchResult.agents.length - 1 ? '└' : '├';
      emit('agent_done', { index: i, question: questions[i], findings: (a.findings || '').slice(0, 500), toolCalls: a.toolCallCount, tokenCount: a.tokenCount });
      // Show remaining accumulated text — unparsed tool calls, reasoning, etc.
      // (agentText is cleared by onToolCall on successful parse, so only failed-parse text remains)
      const raw = (agentText.get(a.agentId) ?? '').replace(/\n/g, ' ').trim();
      if (raw) log(`    ${c.dim}├${c.reset} ${c.yellow}${label(a.agentId)}${c.reset} ${c.dim}▸ ${raw.slice(0, 120)}${raw.length > 120 ? '…' : ''}${c.reset}`);
      log(`    ${c.dim}${tree}${c.reset} ${c.yellow}${label(a.agentId)}${c.reset} ${c.green}done${c.reset} ${c.dim}${a.tokenCount} tok · ${a.toolCallCount} tools${c.reset}`);
    });
    log(`    ${c.dim}${researchResult.totalTokens} tok · ${researchResult.totalToolCalls} tools · ${(researchMs / 1000).toFixed(1)}s${c.reset}`);

    // ─── post-research: verify+eval (cold) or generate (warm) ─
    const phases: { label: string; tokens: number; detail: string; timeMs: number }[] = [
      { label: 'Plan', tokens: planTokens, detail: '', timeMs: planMs },
      {
        label: 'Research', tokens: researchResult.totalTokens,
        detail: `(${researchResult.agents.map(a => a.tokenCount).join(' + ')})  ${pad(researchResult.totalToolCalls, 2)} tools`,
        timeMs: researchMs,
      },
    ];
    let kvLine: string | null = null;

    if (!warm) {
      // ── verify ──────────────────────────────────────────
      t = performance.now();
      const findingsText = researchResult.agents
        .map((a, i) => `Q: ${questions[i]}\nA: ${(a.findings || '').trim()}`)
        .join('\n\n');

      log(`\n  ${c.green}●${c.reset} ${c.bold}Verify${c.reset} ${c.dim}${VERIFY_COUNT} attempts${c.reset}`);
      const verifyResult = await verify(ctx, store, { findings: findingsText, query, count: VERIFY_COUNT });
      const verifyMs = performance.now() - t;

      verifyResult.attempts.forEach((a, i) => {
        const tree = i === verifyResult.attempts.length - 1 ? '└' : '├';
        emit('attempt_done', { index: i, output: a.output.trim().slice(0, 500), tokenCount: a.tokenCount, ppl: a.ppl });
        log(`    ${c.dim}${tree} ${a.tokenCount} tok · ppl ${a.ppl.toFixed(2)}${c.reset}`);
      });
      log(`    ${c.dim}${verifyResult.totalTokens} tok · ${(verifyMs / 1000).toFixed(1)}s${c.reset}`);

      // ── eval ────────────────────────────────────────────
      t = performance.now();
      const { converged, tokenCount: evalTokens } = await evaluate(ctx, { attempts: verifyResult.attempts });
      const evalMs = performance.now() - t;

      emit('convergence', { converged, evalTokens });
      const verdict = converged === true ? `${c.green}yes${c.reset}` : converged === false ? `${c.red}no${c.reset}` : `${c.yellow}unknown${c.reset}`;
      log(`\n  ${c.green}●${c.reset} ${c.bold}Eval${c.reset} ${c.dim}${evalTokens} tok · ${(evalMs / 1000).toFixed(1)}s${c.reset}`);
      log(`    Converged: ${verdict}`);

      // ── answer ──────────────────────────────────────────
      log(`\n  ${c.dim}${'─'.repeat(58)}${c.reset}\n`);
      const prose = verifyResult.bestOutput.trim()
        .replace(/\*\*(.+?)\*\*/g, `${c.bold}$1${c.reset}`)
        .split('\n').map((l) => `  ${l}`).join('\n');
      log(prose);

      phases.push(
        { label: 'Verify', tokens: verifyResult.totalTokens, detail: `(${verifyResult.attempts.map(a => a.tokenCount).join(' + ')})`, timeMs: verifyMs },
        { label: 'Eval', tokens: evalTokens, detail: `converged: ${converged ? 'yes' : 'no'}`, timeMs: evalMs },
      );

      const kvSaved = researchResult.sharedPrefixLength * (questions.length - 1)
        + verifyResult.prefixLength * (verifyResult.attempts.length - 1);
      kvLine = `  ${c.dim}KV shared    ${researchResult.sharedPrefixLength} × ${questions.length - 1} + ${verifyResult.prefixLength} × ${verifyResult.attempts.length - 1} = ${kvSaved.toLocaleString()} tok saved${c.reset}`;

      emit('complete', {
        planTokens, agentTokens: researchResult.totalTokens,
        researchSteps: researchResult.steps,
        verifyTokens: verifyResult.totalTokens, verifySteps: verifyResult.steps,
        evalTokens, converged,
        totalToolCalls: researchResult.totalToolCalls,
        prefixTokens: verifyResult.prefixLength,
        sharedPrefixTokens: researchResult.sharedPrefixLength,
        agentCount: questions.length, attemptCount: verifyResult.attempts.length,
        wallTimeMs: Math.round(performance.now() - t0),
        planMs: Math.round(planMs), researchMs: Math.round(researchMs),
        verifyMs: Math.round(verifyMs), evalMs: Math.round(evalMs),
        ...researchResult.counters,
      });

      await session.promote(verifyResult.bestBranch);
    } else {
      // ── grounded response ───────────────────────────────
      const agentFindings = researchResult.agents
        .map((a, i) => a.findings ? `[Agent ${i}] ${a.findings.trim()}` : null)
        .filter(Boolean)
        .join('\n\n');

      await session.prefillUser(agentFindings
        ? `Research findings:\n${agentFindings}\n\nUser question: ${query}\n\nAnswer based on the research findings above.`
        : query);

      t = performance.now();
      let responseTokens = 0;
      process.stdout.write(`  ${c.dim}<${c.reset} `);
      for await (const { text } of session.trunk!) {
        process.stdout.write(text);
        responseTokens++;
      }
      console.log('\n');

      phases.push({ label: 'Response', tokens: responseTokens, detail: '', timeMs: performance.now() - t });
    }

    // ─── stats table ──────────────────────────────────────
    const tEnd = performance.now();
    const totalTokens = phases.reduce((s, p) => s + p.tokens, 0);

    log(`\n  ${c.dim}${'━'.repeat(58)}${c.reset}`);
    for (const p of phases) {
      const left = `${p.label.padEnd(10)} ${pad(p.tokens, 5)} tok`;
      const detail = p.detail ? `  ${p.detail}` : '';
      const right = `${pad((p.timeMs / 1000).toFixed(1), 6)}s`;
      log(`  ${c.dim}${left}${detail}${' '.repeat(Math.max(1, 58 - left.length - detail.length - right.length))}${right}${c.reset}`);
    }
    log(`  ${c.dim}${'━'.repeat(58)}${c.reset}`);
    log(`  ${c.bold}Total${c.reset}      ${c.bold}${pad(totalTokens, 5)}${c.reset} tok  ${c.dim}${questions.length} agents · ${researchResult.totalToolCalls} tools${c.reset}         ${c.bold}${pad(((tEnd - t0) / 1000).toFixed(1), 6)}s${c.reset}`);
    if (kvLine) log(kvLine);
    const trunkPos = session.trunk ? session.trunk.position : 0;
    const ctxPct = Math.round(100 * trunkPos / nCtx);
    const ctxStr = `ctx: ${ctxPct}% (${trunkPos.toLocaleString()}/${nCtx.toLocaleString()})`;
    log(`  ${c.dim}${'━'.repeat(58)}${c.reset}`);
    log(`  ${c.dim}${' '.repeat(58 - ctxStr.length)}${ctxStr}${c.reset}`);
    log();
  }

  // ================================================================
  // REPL — single input loop drives both cold and warm paths
  // ================================================================

  // --query with --jsonl: run cold pipeline, emit results, exit
  if (jsonlMode && initialQuery) {
    await handleQuery(initialQuery);
    await session.dispose();
    reranker.dispose();
    ctx.dispose();
    return;
  }

  // --query provided interactively: use as first input
  if (initialQuery) {
    await handleQuery(initialQuery);
  }

  log(`  ${c.dim}${session.trunk ? 'Ask a follow-up question' : 'Enter your research question'} or /quit to exit${c.reset}`);
  log();

  await new Promise<void>((resolve) => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    let exiting = false;
    let generating = false;
    let eofWhileGenerating = false;

    async function exit(): Promise<void> {
      if (exiting) return;
      exiting = true;
      rl.close();
      await session.dispose();
      reranker.dispose();
      ctx.dispose();
      resolve();
    }

    const ask = (): void => {
      if (exiting) return;
      rl.question(`  ${c.dim}>${c.reset} `, handleInput);
    };

    async function handleInput(input: string): Promise<void> {
      try {
        const trimmed = input.trim();
        if (!trimmed || trimmed === '/quit') { await exit(); return; }

        generating = true;
        await handleQuery(trimmed);
        generating = false;

        if (eofWhileGenerating) { await exit(); } else { ask(); }
      } catch (err) {
        log(`  ${c.red}Error: ${(err as Error).message}${c.reset}`);
        generating = false;
        ask();
      }
    }

    rl.on('close', () => {
      if (generating) { eofWhileGenerating = true; } else { exit(); }
    });
    ask();
  });
}

main().catch((err: unknown) => {
  process.stdout.write(`Error: ${(err as Error).message}\n${(err as Error).stack}\n`);
  process.exit(1);
});
