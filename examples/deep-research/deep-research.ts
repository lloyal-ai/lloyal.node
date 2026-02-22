#!/usr/bin/env node
/**
 * Deep Research with Tool-Calling Agents via BranchStore
 *
 * Demonstrates three fork patterns in a multi-agent research pipeline:
 *
 * 1. PLAN:     Branch.create() + grammar — single constrained generation
 * 2. RESEARCH: fork() + prefill() divergent suffixes — content-based divergence
 *              from shared prefix, with tool-calling agentic loop
 * 3. VERIFY:   fork() + reseed() same prompt — stochastic divergence for
 *              convergence checking, then model-as-judge eval fork
 *
 * Search uses a Qwen3-Reranker-0.6B cross-encoder for semantic relevance
 * scoring over a local corpus of markdown files. Both models (generative +
 * reranker) are loaded simultaneously — Qwen3 family shares vocabulary.
 *
 * The key performance insight: BranchStore.commit() packs N branches into
 * ONE llama_decode() call. N agents generate in lockstep with O(1) GPU
 * dispatches per step, regardless of branch count.
 *
 * Usage:
 *   node deep-research.ts <model-path> --corpus <path> --query <text> [options]
 *
 * Required:
 *   <model-path>     Path to generative model (e.g. Qwen3-4B-Instruct)
 *   --corpus  path   Directory of .md files (or single .md file) to research
 *   --query   text   Research question
 *
 * Options:
 *   --reranker path  Reranker model path (default: qwen3-reranker-0.6b)
 *   --jsonl          JSONL output for testing
 *   --verbose        Show native llama.cpp logs
 *
 * Example:
 *   node deep-research.ts ./models/Qwen3-4B.gguf \
 *     --corpus ~/docs --query "How does the auth system work?"
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as readline from 'node:readline';
import {
  createContext, Branch, BranchStore, Session, forkAgent, runAgents,
} from '../../dist/index.js';
import type { SessionContext, AgentState } from '../../dist/index.js';

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
const QUERY = argVal('--query');
const modelPath = args.find((a, i) =>
  !a.startsWith('--') && !flagIndices.has(i)
) || DEFAULT_MODEL;

if (!corpusDir || !QUERY) {
  const missing = [
    !corpusDir && '--corpus',
    !QUERY && '--query',
  ].filter(Boolean);
  process.stdout.write(
    `Usage: node deep-research.ts [model-path] --corpus <path> --query <text> [--reranker <path>]\n` +
    `Missing: ${missing.join(', ')}\n`
  );
  process.exit(1);
}

// ================================================================
// Suppress native llama.cpp logs (C-level stderr) for clean output.
// The native binary hasn't loaded yet (lazy on first createContext),
// so redirecting fd 2 here catches all ggml/llama init logs.
// Use --verbose to see them.
// ================================================================
if (!verbose && !jsonlMode) {
  try {
    fs.closeSync(2);
    fs.openSync(process.platform === 'win32' ? '\\\\.\\NUL' : '/dev/null', 'w');
  } catch { /* non-fatal — logs will show */ }
}

// ================================================================
// DISPLAY — ANSI formatting for terminal output
// ================================================================

const isTTY = process.stdout.isTTY;
const c = isTTY ? {
  bold: '\x1b[1m', dim: '\x1b[2m', reset: '\x1b[0m',
  green: '\x1b[32m', cyan: '\x1b[36m', yellow: '\x1b[33m', red: '\x1b[31m',
} : { bold: '', dim: '', reset: '', green: '', cyan: '', yellow: '', red: '' };

const log = (...a: unknown[]): void => { if (!jsonlMode) console.log(...a); };

function emit(event: string, data: Record<string, unknown>): void {
  if (jsonlMode) console.log(JSON.stringify({ event, ...data }));
}

// ================================================================
// CONSTANTS
// ================================================================

const AGENT_COUNT = 3;
const VERIFY_COUNT = 3;
const MAX_TOOL_TURNS = 6;

// ================================================================
// CORPUS — load and chunk at ## boundaries
// ================================================================

interface CorpusFile { name: string; content: string }
interface Chunk { file: string; heading: string; text: string; tokens: number[] }
interface SubChunk { heading: string; text: string }

function loadCorpus(): CorpusFile[] {
  if (!fs.existsSync(corpusDir!)) {
    process.stdout.write(`Error: corpus not found: ${corpusDir}\n`);
    process.exit(1);
  }
  const stat = fs.statSync(corpusDir!);
  if (stat.isFile()) {
    return [{ name: path.basename(corpusDir!), content: fs.readFileSync(corpusDir!, 'utf8') }];
  }
  const files = fs.readdirSync(corpusDir!).filter((f) => f.endsWith('.md'));
  if (!files.length) {
    process.stdout.write(`Error: no .md files in: ${corpusDir}\n`);
    process.exit(1);
  }
  return files.map((f) => ({
    name: f,
    content: fs.readFileSync(path.join(corpusDir!, f), 'utf8'),
  }));
}

// Max chars per chunk — conservative estimate at ~3 chars/token for code-heavy
// content, leaving room for reranker template overhead (~130 tokens).
// With reranker nCtx=8192: budget ≈ 8000 tokens × 3 = 24000 chars.
const CHUNK_CHAR_LIMIT = 24000;

function chunkCorpus(files: CorpusFile[]): Chunk[] {
  const out: Chunk[] = [];
  for (const file of files) {
    for (const section of file.content.split(/(?=^## )/m)) {
      const heading = (section.match(/^##?\s+(.+)/m) || [, file.name])[1]!;
      const trimmed = section.trim();
      if (trimmed.length <= CHUNK_CHAR_LIMIT) {
        out.push({ file: file.name, heading, text: trimmed, tokens: [] });
        continue;
      }
      // Sub-split oversized sections: ### → paragraph → hard truncate
      for (const sub of subChunk(trimmed, heading)) {
        out.push({ file: file.name, heading: sub.heading, text: sub.text, tokens: [] });
      }
    }
  }
  return out;
}

function subChunk(text: string, parentHeading: string): SubChunk[] {
  // Try splitting at ### boundaries first
  const subSections = text.split(/(?=^### )/m);
  if (subSections.length > 1) {
    const results: SubChunk[] = [];
    for (const sub of subSections) {
      const subHeading = (sub.match(/^###?\s+(.+)/m) || [, parentHeading])[1]!;
      const trimmed = sub.trim();
      if (trimmed.length <= CHUNK_CHAR_LIMIT) {
        results.push({ heading: subHeading, text: trimmed });
      } else {
        // Still too large — fall through to paragraph splitting
        results.push(...splitByParagraph(trimmed, subHeading));
      }
    }
    return results;
  }
  // No ### headings — split by paragraphs
  return splitByParagraph(text, parentHeading);
}

function splitByParagraph(text: string, heading: string): SubChunk[] {
  const paragraphs = text.split(/\n\n+/);
  const results: SubChunk[] = [];
  let current = '';
  let partIndex = 0;

  for (const para of paragraphs) {
    if (current.length + para.length + 2 > CHUNK_CHAR_LIMIT && current.length > 0) {
      results.push({ heading: `${heading} (${++partIndex})`, text: current.trim() });
      current = '';
    }
    // Single paragraph exceeds limit — hard truncate
    if (para.length > CHUNK_CHAR_LIMIT) {
      if (current.length > 0) {
        results.push({ heading: `${heading} (${++partIndex})`, text: current.trim() });
        current = '';
      }
      results.push({ heading: `${heading} (${++partIndex})`, text: para.slice(0, CHUNK_CHAR_LIMIT) });
      continue;
    }
    current += (current ? '\n\n' : '') + para;
  }
  if (current.trim()) {
    results.push({ heading: `${heading} (${partIndex > 0 ? ++partIndex : ''})`.replace(/ \(\)$/, ''), text: current.trim() });
  }
  return results;
}

const corpus = loadCorpus();
const chunks: Chunk[] = chunkCorpus(corpus);

// ================================================================
// RERANKER — Qwen3-Reranker cross-encoder scoring via Branch API
// ================================================================

// Prompt template from Qwen3-Reranker model card: system (yes/no judge) +
// user (<Instruct> + <Query> + <Document>) + empty think block prefix.
const RERANK_PREFIX =
  '<|im_start|>system\n' +
  'Judge whether the Document meets the requirements based on the Query ' +
  'and the Instruct provided. Note that the answer can only be "yes" or "no".' +
  '<|im_end|>\n<|im_start|>user\n' +
  '<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n\n' +
  '<Query>: ';
const RERANK_MID = '\n\n<Document>: ';
const RERANK_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n';

let rerankCtx: SessionContext | null = null;
let yesId = 0;
let noId = 0;

// Pre-tokenized template segments — populated after reranker loads.
let rerankPrefixTokens: number[] | null = null; // RERANK_PREFIX (with BOS)
let rerankMidTokens: number[] | null = null;    // RERANK_MID
let rerankSuffixTokens: number[] | null = null; // RERANK_SUFFIX

function rerankScore(logits: Float32Array): number {
  const max = Math.max(logits[yesId], logits[noId]);
  const yesExp = Math.exp(logits[yesId] - max);
  const noExp = Math.exp(logits[noId] - max);
  return yesExp / (yesExp + noExp);
}

// ================================================================
// TOOLS — reranker-backed search + snippet extraction
// ================================================================

interface ScoredChunk { file: string; heading: string; score: number }

async function toolSearch(query: string): Promise<ScoredChunk[]> {
  const queryTokens = await rerankCtx!.tokenize(query, false);
  const scored: ScoredChunk[] = [];
  for (const chunk of chunks) {
    // Pre-tokenized segments — no string concat, no per-chunk tokenize().
    // Boundary safety: all joints are at special tokens or newlines,
    // which are explicit token boundaries in Qwen3's BPE vocabulary.
    const tokens = [
      ...rerankPrefixTokens!, ...queryTokens,
      ...rerankMidTokens!, ...chunk.tokens,
      ...rerankSuffixTokens!,
    ];
    // Fresh branch per chunk — position must start at 0 each time.
    const branch = Branch.create(rerankCtx!, 0, { temperature: 0 });
    await branch.prefill(tokens);
    const score = rerankScore(branch.getLogits());
    await branch.prune();
    scored.push({ file: chunk.file, heading: chunk.heading, score: Math.round(score * 1000) / 1000 });
  }
  return scored.sort((a, b) => b.score - a.score).slice(0, 5);
}

interface ReadFileResult {
  file: string;
  content?: string;
  snippets?: string[];
  error?: string;
}

function toolReadFile(filename: string, query: string): ReadFileResult | { error: string } {
  const file = corpus.find((f) => f.name === filename);
  if (!file) {
    return { error: `File not found: ${filename}. Available: ${corpus.map((f) => f.name).join(', ')}` };
  }
  if (!query) return { file: file.name, content: file.content.slice(0, 800) };
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  const lines = file.content.split('\n');
  const snippets: string[] = [];
  const seen = new Set<number>();
  for (let i = 0; i < lines.length; i++) {
    if (!terms.some((t) => lines[i].toLowerCase().includes(t))) continue;
    const start = Math.max(0, i - 1);
    const end = Math.min(lines.length, i + 4);
    if (seen.has(start)) continue;
    seen.add(start);
    snippets.push(lines.slice(start, end).join('\n'));
    if (snippets.length >= 3) break;
  }
  return snippets.length > 0
    ? { file: file.name, snippets }
    : { file: file.name, snippets: ['No matches for: ' + query] };
}

async function executeTool(name: string, toolArgs: Record<string, unknown>): Promise<unknown> {
  switch (name) {
    case 'search':
      return toolSearch((toolArgs.query as string) || '');
    case 'read_file':
      return toolReadFile(
        (toolArgs.filename as string) || (toolArgs.path as string) || '',
        (toolArgs.query as string) || ''
      );
    case 'report':
      return { acknowledged: true };
    default:
      return { error: `Unknown tool: ${name}` };
  }
}

const TOOLS = [
  {
    type: 'function',
    function: {
      name: 'search',
      description: 'Search the knowledge base for relevant content. Returns sections ranked by semantic relevance.',
      parameters: {
        type: 'object',
        properties: { query: { type: 'string', description: 'Search query' } },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'read_file',
      description: 'Extract relevant snippets from a specific file. Use query to target specific content.',
      parameters: {
        type: 'object',
        properties: {
          filename: { type: 'string', description: 'Filename from search results (e.g. "api-security.md")' },
          query: { type: 'string', description: 'What to extract from the file' },
        },
        required: ['filename'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'report',
      description: 'Submit your final research findings. Call this when you have gathered enough information to answer the question.',
      parameters: {
        type: 'object',
        properties: { findings: { type: 'string', description: 'Your research findings and answer' } },
        required: ['findings'],
      },
    },
  },
];

const TOOLS_JSON = JSON.stringify(TOOLS);

const AGENT_SYSTEM_PROMPT =
  'You are a research assistant with access to a knowledge base. ' +
  'Use the search and read_file tools to find information, then call report with your findings. ' +
  'Be thorough: search first, read relevant files, then report. ' +
  'Available files: ' + corpus.map((f) => f.name).join(', ');

// ================================================================
// HELPERS
// ================================================================

const sec = (a: number, b: number): string => ((b - a) / 1000).toFixed(1);
const pad = (s: unknown, n: number): string => String(s).padStart(n);
const fmtSize = (bytes: number): string => bytes > 1e9
  ? (bytes / 1e9).toFixed(1) + ' GB'
  : (bytes / 1e6).toFixed(0) + ' MB';

// ================================================================
// MAIN
// ================================================================

interface Attempt {
  branch: InstanceType<typeof Branch>;
  output: string;
  done: boolean;
  tokenCount: number;
  ppl: number;
}

async function main(): Promise<void> {
  const t0 = performance.now();

  const modelName = path.basename(modelPath).replace(/-Q\w+\.gguf$/, '');
  const rerankName = path.basename(rerankModelPath).replace(/-q\w+\.gguf$/i, '');
  const modelSize = fmtSize(fs.statSync(modelPath).size);
  const rerankSize = fmtSize(fs.statSync(rerankModelPath).size);

  log();
  log(`${c.bold}  Deep Research${c.reset} ${c.dim}— BranchStore Tool-Calling Agents${c.reset}`);
  log();

  emit('start', {
    model: path.basename(modelPath),
    reranker: path.basename(rerankModelPath),
    query: QUERY!,
    agentCount: AGENT_COUNT,
    verifyCount: VERIFY_COUNT,
    chunks: chunks.length,
  });

  log(`  ${c.green}●${c.reset} Loading ${c.bold}${modelName}${c.reset} ${c.dim}(${modelSize}, KV: Q4_0)${c.reset}`);

  // Load generative model
  const nCtx = parseInt(process.env.LLAMA_CTX_SIZE || '16384', 10);
  const ctx = await createContext({
    modelPath,
    nCtx,
    nSeqMax: AGENT_COUNT + 1,
    typeK: 'q4_0',
    typeV: 'q4_0',
  });

  log(`  ${c.green}●${c.reset} Loading ${c.bold}${rerankName}${c.reset} ${c.dim}(${rerankSize}, reranker)${c.reset}`);

  // Load reranker (small — ~300 MB alongside the 4B generative model)
  rerankCtx = await createContext({
    modelPath: rerankModelPath,
    nCtx: 8192,
    nSeqMax: AGENT_COUNT,
  });

  // Pre-tokenize reranker template segments + chunk texts.
  // Done once — saves N_chunks × tokenize() calls per search.
  [yesId] = await rerankCtx.tokenize('yes', false);
  [noId] = await rerankCtx.tokenize('no', false);
  rerankPrefixTokens = await rerankCtx.tokenize(RERANK_PREFIX, true);
  rerankMidTokens = await rerankCtx.tokenize(RERANK_MID, false);
  rerankSuffixTokens = await rerankCtx.tokenize(RERANK_SUFFIX, false);
  for (const chunk of chunks) {
    chunk.tokens = await rerankCtx.tokenize(chunk.text, false);
  }

  const corpusIsFile = corpus.length === 1 && fs.statSync(corpusDir!).isFile();
  const corpusLabel = corpusIsFile
    ? path.basename(corpusDir!)
    : `${path.basename(corpusDir!)}/ — ${corpus.length} files`;
  log(`  ${c.dim}  Corpus: ${corpusLabel} → ${chunks.length} chunks${c.reset}`);

  const store = new BranchStore(ctx);

  log();
  log(`  ${c.dim}Query${c.reset}`);
  log(`  ${c.bold}${QUERY}${c.reset}`);

  // ================================================================
  // PHASE 1: PLAN — Branch.create() + grammar
  // ================================================================
  const tPlan = performance.now();

  const planSchema = {
    type: 'object',
    properties: {
      questions: {
        type: 'array',
        items: { type: 'string' },
        minItems: 2,
        maxItems: AGENT_COUNT,
      },
    },
    required: ['questions'],
  };
  const planGrammar = await ctx.jsonSchemaToGrammar(JSON.stringify(planSchema));

  const planMessages = [
    { role: 'system', content: 'You break research queries into sub-questions. Output JSON only.' },
    { role: 'user', content: `Break this into ${AGENT_COUNT} independent sub-questions for parallel research: "${QUERY}"` },
  ];
  const { prompt: planPrompt } = await ctx.formatChat(JSON.stringify(planMessages));
  const planTokens = await ctx.tokenize(planPrompt);

  const lead = Branch.create(ctx, 0, { temperature: 0.3 }, undefined, planGrammar);
  await lead.prefill(planTokens);

  let planOutput = '';
  let planTokenCount = 0;
  for await (const { text } of lead) {
    planOutput += text;
    planTokenCount++;
  }
  await lead.prune();

  let questions: string[];
  try {
    const plan = JSON.parse(planOutput);
    questions = plan.questions.slice(0, AGENT_COUNT);
    if (!questions.length) throw new Error('empty questions');
  } catch {
    questions = Array.from({ length: AGENT_COUNT }, (_, i) => `${QUERY} (aspect ${i + 1})`);
  }

  emit('plan', { questions, planTokens: planTokenCount });

  // ================================================================
  // PHASE 2: RESEARCH — fork() + prefill() divergent suffixes + tools
  // ================================================================
  const tResearch = performance.now();

  log();
  log(`  ${c.green}●${c.reset} ${c.bold}Plan${c.reset} ${c.dim}${planTokenCount} tok · ${sec(tPlan, tResearch)}s${c.reset}`);
  questions.forEach((q, i) => log(`    ${c.dim}${i + 1}.${c.reset} ${q}`));

  // Shared prefix: system prompt + tool definitions, NO assistant prompt
  const sharedMessages = [{ role: 'system', content: AGENT_SYSTEM_PROMPT }];
  const sharedFmt = await ctx.formatChat(
    JSON.stringify(sharedMessages),
    { tools: TOOLS_JSON, addGenerationPrompt: false }
  );
  const sharedTokens = await ctx.tokenize(sharedFmt.prompt);

  // Root branch — prefill shared prefix once
  const agentRoot = Branch.create(ctx, 0, { temperature: 0.5 });
  await agentRoot.prefill(sharedTokens);

  // Fork N agents, compute divergent suffixes via token slicing
  const agents: AgentState[] = [];
  for (const q of questions) {
    const branch = await agentRoot.fork();

    const fullMessages = [
      { role: 'system', content: AGENT_SYSTEM_PROMPT },
      { role: 'user', content: q },
    ];
    const fmt = await ctx.formatChat(JSON.stringify(fullMessages), { tools: TOOLS_JSON });
    const fullTokens = await ctx.tokenize(fmt.prompt);
    const suffixTokens = fullTokens.slice(sharedTokens.length);

    agents.push({
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
    });
  }
  // agentRoot pruned after agents are done (can't prune parent with live children)

  // Batched prefill — only the unique suffixes
  await store.prefill(agents.map((w) => [w.branch, w.suffixTokens]));

  emit('research_start', {
    agentCount: agents.length,
    sharedPrefixTokens: sharedTokens.length,
  });

  log();
  log(`  ${c.green}●${c.reset} ${c.bold}Research${c.reset} ${c.dim}${agents.length} agents · shared prefix ${sharedTokens.length} tok${c.reset}`);

  // Reranker mutex — serializes llama_decode calls on rerankCtx.
  // Fire-and-forget tool dispatch means multiple agents can dispatch search
  // concurrently; _branchPrefill runs on the libuv thread pool, so concurrent
  // calls race llama_decode on the same llama_context. BranchStore serializes
  // via batched decode (one llama_decode per commit/prefill), but individual
  // Branch.prefill calls on rerankCtx bypass that.
  let rerankLock = Promise.resolve();
  function withRerankLock<T>(fn: () => Promise<T>): Promise<T> {
    const prev = rerankLock;
    let release: () => void;
    rerankLock = new Promise((r) => { release = r; });
    return prev.then(fn).finally(release!);
  }

  const executeToolLocked = (name: string, args: Record<string, unknown>): Promise<unknown> =>
    name === 'search'
      ? withRerankLock(() => executeTool(name, args))
      : executeTool(name, args);

  const { totalTokens: totalAgentTokens, totalToolCalls, steps: researchSteps, counters } =
    await runAgents(agents, {
      store, ctx,
      executeTool: executeToolLocked,
      maxTurns: MAX_TOOL_TURNS,
      onToolCall(ai: number, toolName: string, args: string) {
        emit('tool_call', { agentIndex: ai, toolName, arguments: args });
        let toolArgs: Record<string, string>;
        try { toolArgs = JSON.parse(args); } catch { toolArgs = {}; }
        const argSummary = toolName === 'search'
          ? `"${toolArgs.query || ''}"`
          : toolName === 'report' ? ''
          : toolArgs.filename + (toolArgs.query ? `, "${toolArgs.query}"` : '');
        log(`    ${c.dim}├${c.reset} ${c.yellow}${ai}${c.reset} ${c.cyan}${toolName}${c.reset}${argSummary ? `(${argSummary})` : ''}`);
      },
      onToolResult(ai: number, toolName: string, resultStr: string) {
        emit('tool_result', {
          agentIndex: ai, toolName,
          result: resultStr.length > 200 ? resultStr.slice(0, 200) + '...' : resultStr,
        });
        log(`    ${c.dim}├${c.reset} ${c.yellow}${ai}${c.reset} ${c.dim}← ${toolName} ${resultStr.length}b${c.reset}`);
      },
    });

  for (let i = 0; i < agents.length; i++) {
    const w = agents[i];
    const isLast = i === agents.length - 1;
    const branchChar = isLast ? '└' : '├';

    emit('agent_done', {
      index: i,
      question: questions[i],
      findings: (w.findings || '').slice(0, 500),
      toolCalls: w.toolCallCount,
      turns: w.turns,
      tokenCount: w.tokenCount,
    });

    log(`    ${c.dim}${branchChar}${c.reset} ${c.yellow}${i}${c.reset} ${c.green}done${c.reset} ${c.dim}${w.tokenCount} tok · ${w.toolCallCount} tools${c.reset}`);

    await w.branch.prune();
  }
  await agentRoot.prune();

  // ================================================================
  // PHASE 3: VERIFY — fork() + reseed() + eval fork
  // ================================================================
  const tVerify = performance.now();

  log(`    ${c.dim}${totalAgentTokens} tok · ${totalToolCalls} tools · ${sec(tResearch, tVerify)}s${c.reset}`);

  const findingsText = agents
    .map((w, i) => `Q: ${questions[i]}\nA: ${(w.findings || '').trim()}`)
    .join('\n\n');

  const synthMessages = [
    { role: 'system', content: 'Synthesize the research findings into a coherent, concise summary.' },
    { role: 'user', content: `Research findings:\n\n${findingsText}\n\nSynthesize these into a brief summary answering: "${QUERY}"` },
  ];
  const { prompt: synthPrompt } = await ctx.formatChat(JSON.stringify(synthMessages));
  const synthTokens = await ctx.tokenize(synthPrompt);

  const synthRoot = Branch.create(ctx, 0, { temperature: 0.7 });
  await synthRoot.prefill(synthTokens);

  emit('verify_start', {
    attemptCount: VERIFY_COUNT,
    prefixTokens: synthTokens.length,
  });

  log();
  log(`  ${c.green}●${c.reset} ${c.bold}Verify${c.reset} ${c.dim}${VERIFY_COUNT} attempts · shared prefix ${synthTokens.length} tok${c.reset}`);

  const attempts: Attempt[] = [];
  for (let i = 0; i < VERIFY_COUNT; i++) {
    const branch = await synthRoot.fork();
    branch.reseedSampler(2000 + i);
    attempts.push({ branch, output: '', done: false, tokenCount: 0, ppl: Infinity });
  }
  // synthRoot pruned after attempts are done (can't prune parent with live children)

  let verifySteps = 0;
  for (;;) {
    const entries: [InstanceType<typeof Branch>, number][] = [];
    for (const a of attempts) {
      if (a.done) continue;
      const { token, text, isStop } = a.branch.produceSync();
      if (isStop) {
        const p = a.branch.perplexity;
        a.ppl = Number.isFinite(p) ? p : Infinity;
        a.done = true;
        continue;
      }
      entries.push([a.branch, token]);
      a.output += text;
      a.tokenCount++;
    }
    if (entries.length === 0) break;
    await store.commit(entries);
    verifySteps++;
  }

  const totalVerifyTokens = attempts.reduce((s, a) => s + a.tokenCount, 0);
  for (let i = 0; i < attempts.length; i++) {
    const isLast = i === attempts.length - 1;
    const branchChar = isLast ? '└' : '├';

    emit('attempt_done', {
      index: i,
      output: attempts[i].output.trim().slice(0, 500),
      tokenCount: attempts[i].tokenCount,
      ppl: attempts[i].ppl,
    });

    log(`    ${c.dim}${branchChar} ${attempts[i].tokenCount} tok · ppl ${attempts[i].ppl.toFixed(2)}${c.reset}`);
  }

  // Pick lowest perplexity synthesis (most coherent) — same as best-of-n.mjs
  // Selected before pruning so we can keep the best branch alive for follow-up.
  const bestAttempt = attempts.reduce((a, b) => a.ppl <= b.ppl ? a : b);

  for (const a of attempts) { if (a !== bestAttempt) await a.branch.prune(); }
  // synthRoot stays alive until interactive loop ends — forked children share
  // physical KV entries with the parent via seq_id tags.

  // Eval fork — model-as-judge
  const tEval = performance.now();

  log(`    ${c.dim}${totalVerifyTokens} tok · ${sec(tVerify, tEval)}s${c.reset}`);

  const responsesText = attempts
    .map((a, i) => `Response ${i + 1}: ${a.output.trim()}`)
    .join('\n\n');

  const evalMessages = [
    {
      role: 'system',
      content: 'You are a consistency checker. Compare the responses and determine if they convey the same core meaning. Output JSON only.',
    },
    {
      role: 'user',
      content: `Do these responses agree on the key points?\n\n${responsesText}`,
    },
  ];

  const evalSchema = {
    type: 'object',
    properties: { converged: { type: 'boolean' } },
    required: ['converged'],
  };
  const evalGrammar = await ctx.jsonSchemaToGrammar(JSON.stringify(evalSchema));

  const { prompt: evalPrompt } = await ctx.formatChat(JSON.stringify(evalMessages));
  const evalTokens = await ctx.tokenize(evalPrompt);

  const evalBranch = Branch.create(ctx, 0, { temperature: 0 }, undefined, evalGrammar);
  await evalBranch.prefill(evalTokens);

  let evalOutput = '';
  let evalTokenCount = 0;
  for await (const { text } of evalBranch) {
    evalOutput += text;
    evalTokenCount++;
  }
  await evalBranch.prune();

  let converged: boolean | null;
  try {
    converged = JSON.parse(evalOutput).converged;
  } catch {
    converged = null;
  }

  emit('convergence', { evalOutput, evalTokens: evalTokenCount, converged });

  // ================================================================
  // COMPLETE
  // ================================================================
  const tEnd = performance.now();

  const verdict = converged === true ? `${c.green}yes${c.reset}` : converged === false ? `${c.red}no${c.reset}` : `${c.yellow}unknown${c.reset}`;
  log();
  log(`  ${c.green}●${c.reset} ${c.bold}Eval${c.reset} ${c.dim}${evalTokenCount} tok · ${sec(tEval, tEnd)}s${c.reset}`);
  log(`    Converged: ${verdict}`);

  log();
  log(`  ${c.dim}${'─'.repeat(58)}${c.reset}`);
  log();
  const prose = bestAttempt.output.trim()
    .replace(/\*\*(.+?)\*\*/g, `${c.bold}$1${c.reset}`)
    .split('\n').map((l) => `  ${l}`).join('\n');
  log(prose);
  log();

  const totalTokens = planTokenCount + totalAgentTokens + totalVerifyTokens + evalTokenCount;

  emit('complete', {
    planTokens: planTokenCount,
    agentTokens: totalAgentTokens,
    researchSteps,
    verifyTokens: totalVerifyTokens,
    verifySteps,
    evalTokens: evalTokenCount,
    converged,
    totalToolCalls,
    prefixTokens: synthTokens.length,
    sharedPrefixTokens: sharedTokens.length,
    agentCount: questions.length,
    attemptCount: attempts.length,
    wallTimeMs: Math.round(tEnd - t0),
    planMs: Math.round(tResearch - tPlan),
    researchMs: Math.round(tVerify - tResearch),
    verifyMs: Math.round(tEval - tVerify),
    evalMs: Math.round(tEnd - tEval),
    ...counters,
  });

  log();
  log(`  ${c.dim}${'━'.repeat(58)}${c.reset}`);
  log(`  ${c.dim}Plan       ${pad(planTokenCount, 5)} tok${' '.repeat(30)}${pad(sec(tPlan, tResearch), 6)}s${c.reset}`);
  log(`  ${c.dim}Research   ${pad(totalAgentTokens, 5)} tok  (${agents.map((w) => w.tokenCount).join(' + ')})  ${pad(totalToolCalls, 2)} tools  ${pad(sec(tResearch, tVerify), 6)}s${c.reset}`);
  log(`  ${c.dim}Verify     ${pad(totalVerifyTokens, 5)} tok  (${attempts.map((a) => a.tokenCount).join(' + ')})${' '.repeat(11)}${pad(sec(tVerify, tEval), 6)}s${c.reset}`);
  log(`  ${c.dim}Eval       ${pad(evalTokenCount, 5)} tok  converged: ${converged ? 'yes' : 'no'}${' '.repeat(11)}${pad(sec(tEval, tEnd), 6)}s${c.reset}`);
  const kvSaved = sharedTokens.length * (agents.length - 1) + synthTokens.length * (attempts.length - 1);
  log(`  ${c.dim}${'━'.repeat(58)}${c.reset}`);
  log(`  ${c.bold}Total${c.reset}      ${c.bold}${pad(totalTokens, 5)}${c.reset} tok  ${c.dim}${agents.length} agents · ${totalToolCalls} tools${c.reset}         ${c.bold}${pad(sec(t0, tEnd), 6)}s${c.reset}`);
  log(`  ${c.dim}KV shared    ${sharedTokens.length} × ${agents.length - 1} + ${synthTokens.length} × ${attempts.length - 1} = ${kvSaved.toLocaleString()} tok saved${c.reset}`);
  log();

  if (jsonlMode) {
    await bestAttempt.branch.prune();
    await synthRoot.prune();
    rerankCtx!.dispose();
    ctx.dispose();
    return;
  }

  // ================================================================
  // INTERACTIVE — readline follow-up loop with agent-swarm research
  // ================================================================

  // Session manages trunk lifecycle — promote crowns winner, freeing
  // AGENT_COUNT seq_ids for follow-up research agents.
  const session = new Session({ ctx, store });
  await session.promote(bestAttempt.branch);

  log(`  ${c.dim}Ask a follow-up question or /quit to exit${c.reset}`);
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
      rerankCtx!.dispose();
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
      if (!trimmed || trimmed === '/quit') {
        await exit();
        return;
      }

      generating = true;

      // Fork AGENT_COUNT research agents from the conversation trunk.
      // Each agent inherits full conversation KV (back-references resolve
      // naturally), gets reseeded for search diversity.
      log(`  ${c.dim}  researching...${c.reset}`);

      const followUpAgents: AgentState[] = [];
      for (let i = 0; i < AGENT_COUNT; i++) {
        const agent = await forkAgent(session.trunk!, {
          systemPrompt: AGENT_SYSTEM_PROMPT,
          content: trimmed,
          tools: TOOLS_JSON,
          seed: Date.now() + i,
        }, ctx);
        followUpAgents.push(agent);
      }

      // Batch prefill all agents' divergent suffixes
      await store.prefill(followUpAgents.map((a) => [a.branch, a.suffixTokens]));

      // Run parallel research with batched decode
      const swarmResult = await runAgents(followUpAgents, {
        store, ctx,
        executeTool: executeToolLocked,
        maxTurns: MAX_TOOL_TURNS,
        onToolCall(ai: number, toolName: string, args: string) {
          emit('tool_call', { agentIndex: ai, toolName, arguments: args });
          let toolArgs: Record<string, string>;
          try { toolArgs = JSON.parse(args); } catch { toolArgs = {}; }
          const argSummary = toolName === 'search'
            ? `"${toolArgs.query || ''}"`
            : toolName === 'report' ? ''
            : toolArgs.filename + (toolArgs.query ? `, "${toolArgs.query}"` : '');
          log(`    ${c.dim}├${c.reset} ${c.yellow}${ai}${c.reset} ${c.cyan}${toolName}${c.reset}${argSummary ? `(${argSummary})` : ''}`);
        },
        onToolResult(ai: number, toolName: string, resultStr: string) {
          emit('tool_result', { agentIndex: ai, toolName,
            result: resultStr.length > 200 ? resultStr.slice(0, 200) + '...' : resultStr });
          log(`    ${c.dim}├${c.reset} ${c.yellow}${ai}${c.reset} ${c.dim}← ${toolName} ${resultStr.length}b${c.reset}`);
        },
      });

      log(`  ${c.dim}  ${swarmResult.totalToolCalls} tools · ${swarmResult.totalTokens} tok${c.reset}`);

      // Collect findings from all agents
      const agentFindings = followUpAgents
        .map((a, i) => a.findings ? `[Agent ${i}] ${a.findings.trim()}` : null)
        .filter(Boolean)
        .join('\n\n');

      // Prune all agent branches — their findings are captured
      for (const a of followUpAgents) await a.branch.prune();

      // Format findings + question as user turn, prefill into trunk via Session
      const groundedContent = agentFindings
        ? `Research findings:\n${agentFindings}\n\nUser question: ${trimmed}\n\nAnswer based on the research findings above.`
        : trimmed;

      await session.prefillUser(groundedContent);

      // Generate grounded response
      process.stdout.write(`  ${c.dim}<${c.reset} `);
      for await (const { text } of session.trunk!) {
        process.stdout.write(text);
      }
      console.log('\n');

      generating = false;

      if (eofWhileGenerating) {
        await exit();
      } else {
        ask();
      }
      } catch (err) {
        log(`  ${c.red}Error: ${(err as Error).message}${c.reset}`);
        generating = false;
        ask();
      }
    }

    rl.on('close', () => {
      if (generating) {
        eofWhileGenerating = true;
      } else {
        exit();
      }
    });
    ask();
  });
}

main().catch((err: unknown) => {
  // stderr is redirected in quiet mode — use stdout for errors
  process.stdout.write(`Error: ${(err as Error).message}\n${(err as Error).stack}\n`);
  process.exit(1);
});
