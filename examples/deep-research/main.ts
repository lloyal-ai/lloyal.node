#!/usr/bin/env node
/**
 * Deep Research — CLI entry point
 *
 * Wiring only: setup, display subscriber, signal-based REPL.
 * Orchestration lives in harness.ts. Rendering lives in display.ts.
 *
 * Usage:
 *   npx tsx examples/deep-research/main.ts [model-path] --corpus <path> [--query <text>] [options]
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as readline from 'node:readline';
import { main, ensure, createSignal, spawn, each, call, action } from 'effection';
import { createContext } from '../../dist/index.js';
import type { SessionContext } from '../../dist/index.js';
import { initAgents } from '../../dist/agents/index.js';
import { c, log, setJsonlMode, fmtSize } from './display.js';
import { displaySubscriber } from './display.js';
import { loadResources, chunkResources } from './resources/files.js';
import { createReranker } from './reranker.js';
import { createTools } from './tools/index.js';
import { handleQuery } from './harness.js';
import type { HarnessEvent, HarnessOptions } from './harness.js';

// ── CLI args ─────────────────────────────────────────────────────

const DEFAULT_MODEL = path.resolve(__dirname, '../../models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf');
const DEFAULT_RERANKER = path.resolve(__dirname, '../../models/qwen3-reranker-0.6b-q4_k_m.gguf');

const args = process.argv.slice(2);
const jsonlMode = args.includes('--jsonl');
const verbose = args.includes('--verbose');
const trace = args.includes('--trace');

function argVal(flag: string): string | null {
  const i = args.indexOf(flag);
  return i !== -1 ? args[i + 1] : null;
}
const flagIndices = new Set(
  ['--reranker', '--corpus', '--query'].flatMap((f) => {
    const i = args.indexOf(f);
    return i !== -1 ? [i, i + 1] : [];
  }),
);

const rerankModelPath = argVal('--reranker') || DEFAULT_RERANKER;
const corpusDir = argVal('--corpus');
const initialQuery = argVal('--query');
const modelPath = args.find((a, i) => !a.startsWith('--') && !flagIndices.has(i)) || DEFAULT_MODEL;

if (!corpusDir) {
  process.stdout.write(
    `Usage: npx tsx examples/deep-research/main.ts [model-path] --corpus <path> [--query <text>] [--reranker <path>]\nMissing: --corpus\n`,
  );
  process.exit(1);
}

if (jsonlMode) setJsonlMode(true);
if (!verbose && !jsonlMode) {
  try { fs.closeSync(2); fs.openSync(process.platform === 'win32' ? '\\\\.\\NUL' : '/dev/null', 'w'); } catch { /* non-fatal */ }
}

const AGENT_COUNT = 3;
const VERIFY_COUNT = 3;
const MAX_TOOL_TURNS = 6;

// ── Main ─────────────────────────────────────────────────────────

main(function*() {
  const resources = loadResources(corpusDir!);
  const chunks = chunkResources(resources);

  const modelName = path.basename(modelPath).replace(/-Q\w+\.gguf$/, '');
  const rerankName = path.basename(rerankModelPath).replace(/-q\w+\.gguf$/i, '');

  log();
  log(`${c.bold}  Deep Research${c.reset} ${c.dim}\u2014 Structured Concurrency Runtime${c.reset}`);
  log();
  log(`  ${c.green}\u25cf${c.reset} Loading ${c.bold}${modelName}${c.reset} ${c.dim}(${fmtSize(fs.statSync(modelPath).size)}, KV: Q4_0)${c.reset}`);

  const nCtx = parseInt(process.env.LLAMA_CTX_SIZE || '16384', 10);
  const ctx: SessionContext = yield* call(() => createContext({
    modelPath, nCtx,
    nSeqMax: Math.max(AGENT_COUNT, VERIFY_COUNT) + 1,
    typeK: 'q4_0', typeV: 'q4_0',
  }));

  log(`  ${c.green}\u25cf${c.reset} Loading ${c.bold}${rerankName}${c.reset} ${c.dim}(${fmtSize(fs.statSync(rerankModelPath).size)}, reranker)${c.reset}`);

  const reranker = yield* call(() => createReranker(rerankModelPath, { nSeqMax: 8, nCtx: 4096 }));
  yield* ensure(() => { reranker.dispose(); });
  yield* call(() => reranker.tokenizeChunks(chunks));

  const corpusIsFile = resources.length === 1 && fs.statSync(corpusDir!).isFile();
  const corpusLabel = corpusIsFile
    ? path.basename(corpusDir!)
    : `${path.basename(corpusDir!)}/ \u2014 ${resources.length} files`;
  log(`  ${c.dim}  Corpus: ${corpusLabel} \u2192 ${chunks.length} chunks${c.reset}`);

  const { toolMap, toolsJson } = createTools({ resources, chunks, reranker });
  const { session, events } = yield* initAgents<HarnessEvent>(ctx);

  // Display subscriber — all rendering lives here
  yield* spawn(function*() {
    yield* displaySubscriber(events, {
      model: path.basename(modelPath),
      reranker: path.basename(rerankModelPath),
      agentCount: AGENT_COUNT,
      verifyCount: VERIFY_COUNT,
      chunkCount: chunks.length,
    });
  });

  const harnessOpts: HarnessOptions = {
    session, toolMap, toolsJson, events,
    agentCount: AGENT_COUNT, verifyCount: VERIFY_COUNT,
    maxTurns: MAX_TOOL_TURNS, nCtx, trace,
  };

  // Initial query
  if (initialQuery) {
    yield* handleQuery(initialQuery, harnessOpts);
    if (jsonlMode) return;  // scope exit triggers initAgents + ensure cleanup
  }

  // REPL — signal bridges readline into Effection scope
  log(`  ${c.dim}${session.trunk ? 'Ask a follow-up question' : 'Enter your research question'} or /quit to exit${c.reset}`);
  log();

  const inputSignal = createSignal<string, void>();
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  rl.setPrompt(`  ${c.dim}>${c.reset} `);

  yield* spawn(function*() {
    yield* action<void>((resolve) => {
      rl.on('line', (line: string) => inputSignal.send(line.trim()));
      rl.on('close', () => { inputSignal.close(); resolve(); });
      return () => rl.close();
    });
  });

  rl.prompt();
  for (const input of yield* each(inputSignal)) {
    if (!input || input === '/quit') break;
    try {
      yield* handleQuery(input, harnessOpts);
    } catch (err) {
      log(`  ${c.red}Error: ${(err as Error).message}${c.reset}`);
    }
    yield* each.next();
    try { rl.prompt(); } catch { break; }
  }

  // scope exit triggers initAgents + ensure cleanup
}).catch((err: unknown) => {
  process.stdout.write(`Error: ${(err as Error).message}\n${(err as Error).stack}\n`);
  process.exit(1);
});
