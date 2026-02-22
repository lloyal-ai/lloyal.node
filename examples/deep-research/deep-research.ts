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
import { c, log, emit, setJsonlMode, pad, fmtSize } from './display.js';
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
    nSeqMax: AGENT_COUNT + 1,
    typeK: 'q4_0', typeV: 'q4_0',
  });

  log(`  ${c.green}●${c.reset} Loading ${c.bold}${rerankName}${c.reset} ${c.dim}(${rerankSize}, reranker)${c.reset}`);

  const reranker = await createReranker(rerankModelPath, { nSeqMax: AGENT_COUNT });
  await reranker.tokenizeChunks(chunks);

  const corpusIsFile = resources.length === 1 && fs.statSync(corpusDir!).isFile();
  const corpusLabel = corpusIsFile
    ? path.basename(corpusDir!)
    : `${path.basename(corpusDir!)}/ — ${resources.length} files`;
  log(`  ${c.dim}  Corpus: ${corpusLabel} → ${chunks.length} chunks${c.reset}`);

  const { toolsJson, executeTool } = createTools({ resources, chunks, reranker });
  const store = new BranchStore(ctx);
  const session = new Session({ ctx, store });

  // Tool call display — shared across cold + warm paths
  const onToolCall = (ai: number, toolName: string, argsStr: string): void => {
    emit('tool_call', { agentIndex: ai, toolName, arguments: argsStr });
    let toolArgs: Record<string, string>;
    try { toolArgs = JSON.parse(argsStr); } catch { toolArgs = {}; }
    const argSummary = toolName === 'search'
      ? `"${toolArgs.query || ''}"`
      : toolName === 'report' ? ''
      : `${toolArgs.filename}` + (toolArgs.startLine ? ` L${toolArgs.startLine}-${toolArgs.endLine}` : '');
    log(`    ${c.dim}├${c.reset} ${c.yellow}${ai}${c.reset} ${c.cyan}${toolName}${c.reset}${argSummary ? `(${argSummary})` : ''}`);
  };
  const onToolResult = (ai: number, toolName: string, resultStr: string): void => {
    emit('tool_result', {
      agentIndex: ai, toolName,
      result: resultStr.length > 200 ? resultStr.slice(0, 200) + '...' : resultStr,
    });
    log(`    ${c.dim}├${c.reset} ${c.yellow}${ai}${c.reset} ${c.dim}← ${toolName} ${resultStr.length}b${c.reset}`);
  };

  // ================================================================
  // handleQuery — the orchestrator
  //
  // No session yet → cold: plan → research → verify → eval
  // Session exists → warm: research(parent: trunk) → prefillUser → generate
  // ================================================================

  async function handleQuery(query: string): Promise<void> {
    if (!session.trunk) {
      // ─── cold: plan → research → verify → eval ─────────
      const t0 = performance.now();

      emit('start', {
        model: path.basename(modelPath), reranker: path.basename(rerankModelPath),
        query, agentCount: AGENT_COUNT, verifyCount: VERIFY_COUNT, chunks: chunks.length,
      });

      log();
      log(`  ${c.dim}Query${c.reset}`);
      log(`  ${c.bold}${query}${c.reset}`);

      // ─── query → questions ────────────────────────────
      let t = performance.now();

      const { questions, tokenCount: planTokens } = await plan(ctx, {
        query, agentCount: AGENT_COUNT,
      });

      emit('plan', { questions, planTokens });
      const planMs = performance.now() - t;
      log(`\n  ${c.green}●${c.reset} ${c.bold}Plan${c.reset} ${c.dim}${planTokens} tok · ${(planMs / 1000).toFixed(1)}s${c.reset}`);
      questions.forEach((q, i) => log(`    ${c.dim}${i + 1}.${c.reset} ${q}`));

      // ─── questions → findings ─────────────────────────
      t = performance.now();
      log(`\n  ${c.green}●${c.reset} ${c.bold}Research${c.reset} ${c.dim}${questions.length} agents${c.reset}`);

      const researchResult = await research(ctx, store, {
        questions, toolsJson, executeTool,
        maxTurns: MAX_TOOL_TURNS, onToolCall, onToolResult,
      });

      const researchMs = performance.now() - t;
      researchResult.agents.forEach((a, i) => {
        const tree = i === researchResult.agents.length - 1 ? '└' : '├';
        emit('agent_done', { index: i, question: questions[i], findings: (a.findings || '').slice(0, 500), toolCalls: a.toolCallCount, tokenCount: a.tokenCount });
        log(`    ${c.dim}${tree}${c.reset} ${c.yellow}${i}${c.reset} ${c.green}done${c.reset} ${c.dim}${a.tokenCount} tok · ${a.toolCallCount} tools${c.reset}`);
      });
      log(`    ${c.dim}${researchResult.totalTokens} tok · ${researchResult.totalToolCalls} tools · ${(researchMs / 1000).toFixed(1)}s${c.reset}`);

      // ─── findings → attempts ──────────────────────────
      t = performance.now();

      const findingsText = researchResult.agents
        .map((a, i) => `Q: ${questions[i]}\nA: ${(a.findings || '').trim()}`)
        .join('\n\n');

      log(`\n  ${c.green}●${c.reset} ${c.bold}Verify${c.reset} ${c.dim}${VERIFY_COUNT} attempts${c.reset}`);

      const verifyResult = await verify(ctx, store, {
        findings: findingsText, query, count: VERIFY_COUNT,
      });

      const verifyMs = performance.now() - t;
      verifyResult.attempts.forEach((a, i) => {
        const tree = i === verifyResult.attempts.length - 1 ? '└' : '├';
        emit('attempt_done', { index: i, output: a.output.trim().slice(0, 500), tokenCount: a.tokenCount, ppl: a.ppl });
        log(`    ${c.dim}${tree} ${a.tokenCount} tok · ppl ${a.ppl.toFixed(2)}${c.reset}`);
      });
      log(`    ${c.dim}${verifyResult.totalTokens} tok · ${(verifyMs / 1000).toFixed(1)}s${c.reset}`);

      // ─── attempts → convergence ───────────────────────
      t = performance.now();

      const { converged, tokenCount: evalTokens } = await evaluate(ctx, {
        attempts: verifyResult.attempts,
      });

      const evalMs = performance.now() - t;
      emit('convergence', { converged, evalTokens });
      const verdict = converged === true ? `${c.green}yes${c.reset}` : converged === false ? `${c.red}no${c.reset}` : `${c.yellow}unknown${c.reset}`;
      log(`\n  ${c.green}●${c.reset} ${c.bold}Eval${c.reset} ${c.dim}${evalTokens} tok · ${(evalMs / 1000).toFixed(1)}s${c.reset}`);
      log(`    Converged: ${verdict}`);

      // ─── result ───────────────────────────────────────
      const tEnd = performance.now();
      const totalTokens = planTokens + researchResult.totalTokens + verifyResult.totalTokens + evalTokens;

      log(`\n  ${c.dim}${'─'.repeat(58)}${c.reset}\n`);
      const prose = verifyResult.bestOutput.trim()
        .replace(/\*\*(.+?)\*\*/g, `${c.bold}$1${c.reset}`)
        .split('\n').map((l) => `  ${l}`).join('\n');
      log(prose);

      emit('complete', {
        planTokens, agentTokens: researchResult.totalTokens,
        researchSteps: researchResult.steps,
        verifyTokens: verifyResult.totalTokens, verifySteps: verifyResult.steps,
        evalTokens, converged,
        totalToolCalls: researchResult.totalToolCalls,
        prefixTokens: verifyResult.prefixLength,
        sharedPrefixTokens: researchResult.sharedPrefixLength,
        agentCount: questions.length, attemptCount: verifyResult.attempts.length,
        wallTimeMs: Math.round(tEnd - t0),
        planMs: Math.round(planMs), researchMs: Math.round(researchMs),
        verifyMs: Math.round(verifyMs), evalMs: Math.round(evalMs),
        ...researchResult.counters,
      });

      log(`\n  ${c.dim}${'━'.repeat(58)}${c.reset}`);
      log(`  ${c.dim}Plan       ${pad(planTokens, 5)} tok${' '.repeat(30)}${pad((planMs / 1000).toFixed(1), 6)}s${c.reset}`);
      log(`  ${c.dim}Research   ${pad(researchResult.totalTokens, 5)} tok  (${researchResult.agents.map((a) => a.tokenCount).join(' + ')})  ${pad(researchResult.totalToolCalls, 2)} tools  ${pad((researchMs / 1000).toFixed(1), 6)}s${c.reset}`);
      log(`  ${c.dim}Verify     ${pad(verifyResult.totalTokens, 5)} tok  (${verifyResult.attempts.map((a) => a.tokenCount).join(' + ')})${' '.repeat(11)}${pad((verifyMs / 1000).toFixed(1), 6)}s${c.reset}`);
      log(`  ${c.dim}Eval       ${pad(evalTokens, 5)} tok  converged: ${converged ? 'yes' : 'no'}${' '.repeat(11)}${pad((evalMs / 1000).toFixed(1), 6)}s${c.reset}`);
      const kvSaved = researchResult.sharedPrefixLength * (questions.length - 1) + verifyResult.prefixLength * (verifyResult.attempts.length - 1);
      log(`  ${c.dim}${'━'.repeat(58)}${c.reset}`);
      log(`  ${c.bold}Total${c.reset}      ${c.bold}${pad(totalTokens, 5)}${c.reset} tok  ${c.dim}${questions.length} agents · ${researchResult.totalToolCalls} tools${c.reset}         ${c.bold}${pad(((tEnd - t0) / 1000).toFixed(1), 6)}s${c.reset}`);
      log(`  ${c.dim}KV shared    ${researchResult.sharedPrefixLength} × ${questions.length - 1} + ${verifyResult.prefixLength} × ${verifyResult.attempts.length - 1} = ${kvSaved.toLocaleString()} tok saved${c.reset}`);
      log();

      await session.promote(verifyResult.bestBranch);
    } else {
      // ─── warm: plan → research → findings → grounded response ─
      const { questions, tokenCount: planTokens } = await plan(ctx, {
        query, agentCount: AGENT_COUNT,
        parent: session.trunk!,
      });

      log(`\n  ${c.green}●${c.reset} ${c.bold}Plan${c.reset} ${c.dim}${planTokens} tok${c.reset}`);
      questions.forEach((q, i) => log(`    ${c.dim}${i + 1}.${c.reset} ${q}`));

      log(`\n  ${c.green}●${c.reset} ${c.bold}Research${c.reset} ${c.dim}${questions.length} agents${c.reset}`);

      const followUp = await research(ctx, store, {
        questions,
        parent: session.trunk!,
        seed: Date.now(),
        toolsJson, executeTool,
        maxTurns: MAX_TOOL_TURNS, onToolCall, onToolResult,
      });

      followUp.agents.forEach((a, i) => {
        const tree = i === followUp.agents.length - 1 ? '└' : '├';
        log(`    ${c.dim}${tree}${c.reset} ${c.yellow}${i}${c.reset} ${c.green}done${c.reset} ${c.dim}${a.tokenCount} tok · ${a.toolCallCount} tools${c.reset}`);
      });
      log(`    ${c.dim}${followUp.totalToolCalls} tools · ${followUp.totalTokens} tok${c.reset}`);

      const agentFindings = followUp.agents
        .map((a, i) => a.findings ? `[Agent ${i}] ${a.findings.trim()}` : null)
        .filter(Boolean)
        .join('\n\n');

      const groundedContent = agentFindings
        ? `Research findings:\n${agentFindings}\n\nUser question: ${query}\n\nAnswer based on the research findings above.`
        : query;
      await session.prefillUser(groundedContent);

      process.stdout.write(`  ${c.dim}<${c.reset} `);
      for await (const { text } of session.trunk!) {
        process.stdout.write(text);
      }
      console.log('\n');
    }
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
