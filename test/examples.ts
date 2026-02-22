/**
 * Examples Integration Test
 *
 * Runs examples with --jsonl flag and validates structured output.
 * Each example emits JSONL events that we parse and assert on.
 *
 * Usage:
 *   npx tsx test/examples.ts           # Run all examples
 *   npx tsx test/examples.ts entropy   # Run specific example
 *
 * Environment variables:
 *   LLAMA_TEST_MODEL  - Path to chat/instruct model (default: SmolLM2)
 *   EMBED_MODEL_PATH - Path to embedding model (default: nomic-embed)
 */

import { spawn, ChildProcess } from 'node:child_process';
import * as path from 'node:path';
import * as fs from 'node:fs';

interface ExampleEvent {
  event: string;
  [key: string]: any; // eslint-disable-line @typescript-eslint/no-explicit-any -- dynamic JSONL fields
}

interface ExampleConfig {
  path: string;
  timeout: number;
  modelPath?: string;
  extraArgs?: string[];
  skip?: boolean;
  skipReason?: string;
  validate: (events: ExampleEvent[]) => void;
}

interface TestResult {
  name: string;
  passed?: boolean;
  skipped?: boolean;
  skipReason?: string;
  elapsed?: number;
  eventCount?: number;
  metrics?: Record<string, unknown>;
  error?: string;
}

const MODEL_PATH: string = process.env.LLAMA_TEST_MODEL
  ? path.resolve(process.env.LLAMA_TEST_MODEL)
  : path.join(__dirname, '../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf');

const EMBED_MODEL_PATH: string = process.env.EMBED_MODEL_PATH
  ? path.resolve(process.env.EMBED_MODEL_PATH)
  : path.join(__dirname, '../liblloyal/tests/fixtures/nomic-embed-text-v1.5.Q4_K_M.gguf');

const QWEN3_PATH: string = process.env.QWEN3_MODEL
  ? path.resolve(process.env.QWEN3_MODEL)
  : path.join(__dirname, '../models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf');

const RERANKER_PATH: string = process.env.RERANKER_MODEL
  ? path.resolve(process.env.RERANKER_MODEL)
  : path.join(__dirname, '../models/qwen3-reranker-0.6b-q4_k_m.gguf');


if (!fs.existsSync(MODEL_PATH)) {
  console.error('❌ Test model not found!');
  console.error(`   Expected: ${MODEL_PATH}`);
  console.error('   Run: npm run download-models');
  process.exit(1);
}

function runExample(scriptPath: string, timeout: number = 600000, extraArgs: string[] = [], modelPathOverride: string | null = null): Promise<ExampleEvent[]> {
  return new Promise((resolve: (value: ExampleEvent[]) => void, reject: (reason: Error) => void) => {
    const events: ExampleEvent[] = [];
    let stderr: string = '';

    const modelArg: string = modelPathOverride || MODEL_PATH;

    const child: ChildProcess = spawn('npx', ['tsx', scriptPath, modelArg, '--jsonl', ...extraArgs], {
      cwd: path.dirname(scriptPath),
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    child.stdout!.on('data', (data: Buffer) => {
      const lines: string[] = data.toString().split('\n');
      for (const line of lines) {
        if (line.startsWith('{')) {
          try {
            const event: ExampleEvent = JSON.parse(line);
            events.push(event);
          } catch {
            // Ignore malformed JSON
          }
        }
      }
    });

    child.stderr!.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    const timeoutId: NodeJS.Timeout = setTimeout(() => {
      child.kill('SIGTERM');
      reject(new Error('TIMEOUT'));
    }, timeout);

    child.on('close', (code: number | null) => {
      clearTimeout(timeoutId);
      if (code === 0) {
        resolve(events);
      } else {
        reject(new Error(`Exit code ${code}\n${stderr.slice(-500)}`));
      }
    });

    child.on('error', (err: Error) => {
      clearTimeout(timeoutId);
      reject(err);
    });
  });
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(`Assertion failed: ${message}`);
  }
}

const EXAMPLES: Record<string, ExampleConfig> = {
  entropy: {
    path: 'entropy/entropy.ts',
    timeout: 120000,
    validate(events: ExampleEvent[]): void {
      const start: ExampleEvent | undefined = events.find(e => e.event === 'start');
      assert(start, 'should have start event');
      assert(start.model, 'start should have model');

      const comparisons: ExampleEvent[] = events.filter(e => e.event === 'comparison');
      assert(comparisons.length === 3, `should have 3 comparisons, got ${comparisons.length}`);

      for (const c of comparisons) {
        assert(c.fixed && c.edt, 'comparison should have fixed and edt results');
        assert(c.fixed.tokenCount > 0, 'fixed should generate tokens');
        assert(c.edt.tokenCount > 0, 'edt should generate tokens');
        assert(typeof c.edt.avgTemp === 'number', 'edt should have avgTemp');
      }

      const complete: ExampleEvent | undefined = events.find(e => e.event === 'complete');
      assert(complete, 'should have complete event');
      assert(complete.comparisons === 3, 'should complete 3 comparisons');
    },
  },

  embed: {
    path: 'embed/embed.ts',
    timeout: 60000,
    modelPath: EMBED_MODEL_PATH,
    skip: !fs.existsSync(EMBED_MODEL_PATH),
    skipReason: 'nomic-embed-text model not found',
    validate(events: ExampleEvent[]): void {
      const start: ExampleEvent | undefined = events.find(e => e.event === 'start');
      assert(start, 'should have start event');
      assert(start.embeddingDim > 0, 'should have embedding dimension');
      assert(start.hasPooling === true, 'should have pooling enabled');

      const embeddings: ExampleEvent[] = events.filter(e => e.event === 'embedding');
      assert(embeddings.length === 4, 'should embed 4 texts');

      for (const e of embeddings) {
        assert(e.dimension > 0, 'embedding should have dimension');
        assert(e.elapsed >= 0, 'embedding should have elapsed time');
      }

      const similarities: ExampleEvent[] = events.filter(e => e.event === 'similarity');
      assert(similarities.length === 6, 'should have 6 similarity pairs (4 choose 2)');

      for (const s of similarities) {
        assert(s.similarity >= -1 && s.similarity <= 1, 'similarity should be in [-1, 1]');
      }

      const search: ExampleEvent | undefined = events.find(e => e.event === 'search');
      assert(search, 'should have search event');
      assert(search.results.length === 4, 'search should rank all texts');

      const complete: ExampleEvent | undefined = events.find(e => e.event === 'complete');
      assert(complete, 'should have complete event');
    },
  },

  'deep-research': {
    path: 'deep-research/deep-research.ts',
    timeout: 300000,
    modelPath: QWEN3_PATH,
    extraArgs: [
      '--reranker', RERANKER_PATH,
      '--corpus', process.env.DEEP_RESEARCH_CORPUS || '',
      '--query', process.env.DEEP_RESEARCH_QUERY || '',
    ],
    skip: !fs.existsSync(QWEN3_PATH) || !fs.existsSync(RERANKER_PATH)
      || !process.env.DEEP_RESEARCH_CORPUS || !process.env.DEEP_RESEARCH_QUERY,
    skipReason: 'Requires QWEN3_MODEL, RERANKER_MODEL, DEEP_RESEARCH_CORPUS, and DEEP_RESEARCH_QUERY env vars',
    validate(events: ExampleEvent[]): void {
      const start: ExampleEvent | undefined = events.find(e => e.event === 'start');
      assert(start, 'should have start event');
      assert(start.agentCount === 3, 'should have 3 agents');
      assert(start.chunks > 0, 'should have corpus chunks');

      const plan: ExampleEvent | undefined = events.find(e => e.event === 'plan');
      assert(plan, 'should have plan event');
      assert(plan.questions.length >= 2, 'should plan at least 2 sub-questions');

      const researchStart: ExampleEvent | undefined = events.find(e => e.event === 'research_start');
      assert(researchStart, 'should have research_start event');
      assert(researchStart.sharedPrefixTokens > 0, 'should have shared prefix');

      const toolCalls: ExampleEvent[] = events.filter(e => e.event === 'tool_call');
      assert(toolCalls.length > 0, 'should make at least one tool call');

      const agentsDone: ExampleEvent[] = events.filter(e => e.event === 'agent_done');
      assert(agentsDone.length === 3, 'all 3 agents should finish');
      for (const a of agentsDone) {
        assert(a.tokenCount > 0, `agent ${a.index} should generate tokens`);
      }

      const complete: ExampleEvent | undefined = events.find(e => e.event === 'complete');
      assert(complete, 'should have complete event');
      assert(complete.totalToolCalls > 0, 'should have tool calls');
      assert(complete.wallTimeMs > 0, 'should have wall time');
      assert(complete.converged !== undefined, 'should have convergence result');
    },
  },
};

async function runTest(name: string, config: ExampleConfig): Promise<TestResult> {
  const fullPath: string = path.join(__dirname, '../examples', config.path);

  if (config.skip) {
    console.log(`⏭️  ${name}: SKIPPED`);
    console.log(`   Reason: ${config.skipReason}`);
    return { name, skipped: true, skipReason: config.skipReason };
  }

  console.log(`\n📜 ${name}:`);
  const startTime: number = Date.now();

  try {
    const modelPathToUse: string = config.modelPath || MODEL_PATH;
    const extraArgs: string[] = config.extraArgs || [];

    const events: ExampleEvent[] = await runExample(fullPath, config.timeout, extraArgs, modelPathToUse);
    config.validate(events);

    const elapsed: string = ((Date.now() - startTime) / 1000).toFixed(1);

    console.log(`   ✅ PASSED (${elapsed}s)`);
    console.log(`   Events: ${events.length} total`);

    const complete: ExampleEvent | undefined = events.find(e => e.event === 'complete');
    if (complete) {
      const metrics: string[] = [];
      if (complete.generatedTokens) metrics.push(`tokens: ${complete.generatedTokens}`);
      if (complete.outputTokens) metrics.push(`tokens: ${complete.outputTokens}`);
      if (complete.finalPpl) metrics.push(`ppl: ${complete.finalPpl.toFixed(2)}`);
      if (complete.reseeds !== undefined) metrics.push(`reseeds: ${complete.reseeds}`);
      if (complete.acceptRate !== undefined) metrics.push(`accept: ${(complete.acceptRate * 100).toFixed(0)}%`);
      if (complete.validJsonCount !== undefined) metrics.push(`valid: ${complete.validJsonCount}/${complete.branchCount}`);
      if (complete.bestPpl) metrics.push(`bestPpl: ${complete.bestPpl.toFixed(2)}`);
      if (complete.embeddings) metrics.push(`embeddings: ${complete.embeddings}`);
      if (metrics.length > 0) {
        console.log(`   Metrics: ${metrics.join(', ')}`);
      }
    }

    return {
      name,
      passed: true,
      elapsed: parseFloat(elapsed),
      eventCount: events.length,
      metrics: complete || {}
    };

  } catch (err) {
    const elapsed: string = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`   ❌ FAILED (${elapsed}s)`);
    console.log(`   Error: ${(err as Error).message}`);
    return { name, passed: false, elapsed: parseFloat(elapsed), error: (err as Error).message };
  }
}

async function main(): Promise<void> {
  const filterName: string | undefined = process.argv[2];

  console.log('=== Examples Integration Test ===');
  console.log(`Model: ${path.basename(MODEL_PATH)}`);

  const toRun: Record<string, ExampleConfig> = filterName
    ? { [filterName]: EXAMPLES[filterName] }
    : EXAMPLES;

  if (filterName && !EXAMPLES[filterName]) {
    console.error(`Unknown example: ${filterName}`);
    console.error(`Available: ${Object.keys(EXAMPLES).join(', ')}`);
    process.exit(1);
  }

  const results: TestResult[] = [];

  for (const [name, config] of Object.entries(toRun)) {
    const result: TestResult = await runTest(name, config);
    results.push(result);
  }

  console.log('\n' + '═'.repeat(60));
  console.log('EXAMPLES TEST SUMMARY');
  console.log('═'.repeat(60));
  console.log(`Model: ${path.basename(MODEL_PATH)}`);
  console.log();

  const passed: number = results.filter(r => r.passed).length;
  const failed: number = results.filter(r => !r.passed && !r.skipped).length;
  const skipped: number = results.filter(r => r.skipped).length;
  const totalTime: string = results.reduce((sum: number, r: TestResult) => sum + (r.elapsed || 0), 0).toFixed(1);

  console.log('Results:');
  for (const r of results) {
    const status: string = r.skipped ? '⏭️ ' : (r.passed ? '✅' : '❌');
    const time: string = r.elapsed ? ` (${r.elapsed}s)` : '';
    const detail: string = r.skipped ? ` - ${r.skipReason}` : (r.error ? ` - ${r.error.slice(0, 50)}` : '');
    console.log(`  ${status} ${r.name}${time}${detail}`);
  }

  console.log();
  console.log(`Total: ${passed} passed, ${failed} failed, ${skipped} skipped in ${totalTime}s`);

  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err: unknown) => {
  console.error('Fatal:', err);
  process.exit(1);
});
