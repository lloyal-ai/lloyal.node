/**
 * Examples Integration Test
 *
 * Runs examples with --jsonl flag and validates structured output.
 * Each example emits JSONL events that we parse and assert on.
 *
 * Usage:
 *   node test/examples.js           # Run all examples
 *   node test/examples.js entropy   # Run specific example
 *
 * Environment variables:
 *   LLAMA_TEST_MODEL  - Path to chat/instruct model (default: SmolLM2)
 *   EMBED_MODEL_PATH - Path to embedding model (default: nomic-embed)
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Model paths - use env var or default (resolve to absolute path)
const MODEL_PATH = process.env.LLAMA_TEST_MODEL
  ? path.resolve(process.env.LLAMA_TEST_MODEL)
  : path.join(__dirname, '../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf');

// Embedding model (separate from chat model, resolve to absolute path)
const EMBED_MODEL_PATH = process.env.EMBED_MODEL_PATH
  ? path.resolve(process.env.EMBED_MODEL_PATH)
  : path.join(__dirname, '../liblloyal/tests/fixtures/nomic-embed-text-v1.5.Q4_K_M.gguf');


if (!fs.existsSync(MODEL_PATH)) {
  console.error('❌ Test model not found!');
  console.error(`   Expected: ${MODEL_PATH}`);
  console.error('   Run: npm run download-models');
  process.exit(1);
}

/**
 * Run an example with --jsonl and collect events
 */
function runExample(scriptPath, timeout = 600000, extraArgs = [], modelPathOverride = null) {
  return new Promise((resolve, reject) => {
    const events = [];
    let stderr = '';

    const modelArg = modelPathOverride || MODEL_PATH;

    const child = spawn('node', [scriptPath, modelArg, '--jsonl', ...extraArgs], {
      cwd: path.dirname(scriptPath),
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    child.stdout.on('data', (data) => {
      const lines = data.toString().split('\n');
      for (const line of lines) {
        if (line.startsWith('{')) {
          try {
            const event = JSON.parse(line);
            events.push(event);
          } catch {
            // Ignore malformed JSON
          }
        }
      }
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    const timeoutId = setTimeout(() => {
      child.kill('SIGTERM');
      reject(new Error('TIMEOUT'));
    }, timeout);

    child.on('close', (code) => {
      clearTimeout(timeoutId);
      if (code === 0) {
        resolve(events);
      } else {
        reject(new Error(`Exit code ${code}\n${stderr.slice(-500)}`));
      }
    });

    child.on('error', (err) => {
      clearTimeout(timeoutId);
      reject(err);
    });
  });
}

/**
 * Assert helper
 */
function assert(condition, message) {
  if (!condition) {
    throw new Error(`Assertion failed: ${message}`);
  }
}

/**
 * Example test definitions
 */
const EXAMPLES = {
  entropy: {
    path: 'entropy/entropy.mjs',
    timeout: 120000,
    validate(events) {
      const start = events.find(e => e.event === 'start');
      assert(start, 'should have start event');
      assert(start.model, 'start should have model');

      const comparisons = events.filter(e => e.event === 'comparison');
      assert(comparisons.length === 3, `should have 3 comparisons, got ${comparisons.length}`);

      for (const c of comparisons) {
        assert(c.fixed && c.edt, 'comparison should have fixed and edt results');
        assert(c.fixed.tokenCount > 0, 'fixed should generate tokens');
        assert(c.edt.tokenCount > 0, 'edt should generate tokens');
        assert(typeof c.edt.avgTemp === 'number', 'edt should have avgTemp');
      }

      const complete = events.find(e => e.event === 'complete');
      assert(complete, 'should have complete event');
      assert(complete.comparisons === 3, 'should complete 3 comparisons');
    },
  },

  speculative: {
    path: 'speculative/speculative.mjs',
    timeout: 120000,
    validate(events) {
      const start = events.find(e => e.event === 'start');
      assert(start, 'should have start event');
      assert(start.draftCount > 0, 'should have draftCount');

      const iterations = events.filter(e => e.event === 'iteration');
      assert(iterations.length > 0, 'should have iterations');

      for (const iter of iterations) {
        assert(iter.drafted > 0, 'iteration should have drafted tokens');
        assert(iter.accepted >= 0, 'iteration should have accepted count');
      }

      const complete = events.find(e => e.event === 'complete');
      assert(complete, 'should have complete event');
      assert(complete.outputTokens > 0, 'should generate tokens');
      assert(complete.acceptRate >= 0 && complete.acceptRate <= 1, 'acceptRate should be 0-1');
    },
  },

  grammar: {
    path: 'grammar/grammar.mjs',
    timeout: 120000,
    validate(events) {
      const start = events.find(e => e.event === 'start');
      assert(start, 'should have start event');

      const branchPoint = events.find(e => e.event === 'branch_point');
      assert(branchPoint, 'should have branch_point event');
      assert(branchPoint.prefix.includes('"city"'), 'should branch at city field');

      const branchCompletes = events.filter(e => e.event === 'branch_complete');
      assert(branchCompletes.length === 3, 'should complete 3 branches');

      const complete = events.find(e => e.event === 'complete');
      assert(complete, 'should have complete event');
      assert(complete.validJsonCount > 0, 'should produce valid JSON');
    },
  },

  'best-of-n': {
    path: 'best-of-n/best-of-n.mjs',
    timeout: 180000,
    validate(events) {
      const start = events.find(e => e.event === 'start');
      assert(start, 'should have start event');
      assert(start.n === 5, 'should have n=5 candidates');

      const baseline = events.find(e => e.event === 'baseline');
      assert(baseline, 'should have baseline');
      assert(baseline.ppl > 0, 'baseline should have positive ppl');

      const candidates = events.filter(e => e.event === 'candidate');
      assert(candidates.length === 5, 'should have 5 candidates');

      for (const c of candidates) {
        assert(c.ppl >= 1, 'candidate ppl should be >= 1');
        assert(c.tokenCount > 0, 'candidate should have tokens');
      }

      const complete = events.find(e => e.event === 'complete');
      assert(complete, 'should have complete event');
      assert(complete.bestPpl > 0, 'should have bestPpl');
    },
  },

  streaming: {
    path: 'streaming/streaming.mjs',
    timeout: 120000,
    extraArgs: ['--max-tokens=500'],
    validate(events) {
      const start = events.find(e => e.event === 'start');
      assert(start, 'should have start event');

      const tokens = events.filter(e => e.event === 'token');
      assert(tokens.length > 50, 'should generate tokens');

      for (const t of tokens.slice(0, 10)) {
        assert(typeof t.surprisal === 'number', 'token should have surprisal');
      }

      const complete = events.find(e => e.event === 'complete');
      assert(complete, 'should have complete event');
      assert(complete.generatedTokens > 0, 'should generate tokens');
      assert(complete.finalPpl > 0, 'should have finalPpl');
    },
  },

  'streaming-tsampler': {
    path: 'streaming/streaming-tsampler.mjs',
    timeout: 120000,
    extraArgs: ['--max-tokens=500'],
    validate(events) {
      const start = events.find(e => e.event === 'start');
      assert(start, 'should have start event');
      assert(start.ngramSize > 0, 'should have ngramSize');

      const tokens = events.filter(e => e.event === 'token');
      assert(tokens.length > 0, 'should generate tokens');

      const complete = events.find(e => e.event === 'complete');
      assert(complete, 'should have complete event');
      assert(complete.generatedTokens > 0, 'should generate tokens');
      assert(typeof complete.blockedCount === 'number', 'should track blocked count');
      assert(complete.uniqueNgrams > 0, 'should track unique ngrams');
    },
  },

  'streaming-summary': {
    path: 'streaming/streaming-summary.mjs',
    timeout: 180000,
    extraArgs: ['--max-tokens=500'],
    validate(events) {
      const start = events.find(e => e.event === 'start');
      assert(start, 'should have start event');
      assert(start.summaryMode === 'self', 'should default to self-summary mode');

      const tokens = events.filter(e => e.event === 'token');
      assert(tokens.length > 50, 'should generate tokens');

      for (const t of tokens.slice(0, 10)) {
        assert(t.source === 'main', 'token should have source=main');
        assert(typeof t.surprisal === 'number', 'token should have surprisal');
      }

      const complete = events.find(e => e.event === 'complete');
      assert(complete, 'should have complete event');
      assert(complete.generatedTokens > 0, 'should generate tokens');
      assert(complete.finalPpl > 0, 'should have finalPpl');
    },
  },

  embed: {
    path: 'embed/embed.mjs',
    timeout: 60000,
    modelPath: EMBED_MODEL_PATH,
    skip: !fs.existsSync(EMBED_MODEL_PATH),
    skipReason: 'nomic-embed-text model not found',
    validate(events) {
      const start = events.find(e => e.event === 'start');
      assert(start, 'should have start event');
      assert(start.embeddingDim > 0, 'should have embedding dimension');
      assert(start.hasPooling === true, 'should have pooling enabled');

      const embeddings = events.filter(e => e.event === 'embedding');
      assert(embeddings.length === 4, 'should embed 4 texts');

      for (const e of embeddings) {
        assert(e.dimension > 0, 'embedding should have dimension');
        assert(e.elapsed >= 0, 'embedding should have elapsed time');
      }

      const similarities = events.filter(e => e.event === 'similarity');
      assert(similarities.length === 6, 'should have 6 similarity pairs (4 choose 2)');

      for (const s of similarities) {
        assert(s.similarity >= -1 && s.similarity <= 1, 'similarity should be in [-1, 1]');
      }

      const search = events.find(e => e.event === 'search');
      assert(search, 'should have search event');
      assert(search.results.length === 4, 'search should rank all texts');

      const complete = events.find(e => e.event === 'complete');
      assert(complete, 'should have complete event');
    },
  },
};

async function runTest(name, config) {
  const fullPath = path.join(__dirname, '../examples', config.path);

  if (config.skip) {
    console.log(`⏭️  ${name}: SKIPPED`);
    console.log(`   Reason: ${config.skipReason}`);
    return { name, skipped: true, skipReason: config.skipReason };
  }

  console.log(`\n📜 ${name}:`);
  const startTime = Date.now();

  try {
    const modelPathToUse = config.modelPath || MODEL_PATH;
    const extraArgs = config.extraArgs || [];

    const events = await runExample(fullPath, config.timeout, extraArgs, modelPathToUse);
    config.validate(events);

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    console.log(`   ✅ PASSED (${elapsed}s)`);
    console.log(`   Events: ${events.length} total`);

    // Show key metrics from complete event if present
    const complete = events.find(e => e.event === 'complete');
    if (complete) {
      const metrics = [];
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
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`   ❌ FAILED (${elapsed}s)`);
    console.log(`   Error: ${err.message}`);
    return { name, passed: false, elapsed: parseFloat(elapsed), error: err.message };
  }
}

async function main() {
  const filterName = process.argv[2];

  console.log('=== Examples Integration Test ===');
  console.log(`Model: ${path.basename(MODEL_PATH)}`);

  const toRun = filterName
    ? { [filterName]: EXAMPLES[filterName] }
    : EXAMPLES;

  if (filterName && !EXAMPLES[filterName]) {
    console.error(`Unknown example: ${filterName}`);
    console.error(`Available: ${Object.keys(EXAMPLES).join(', ')}`);
    process.exit(1);
  }

  const results = [];

  for (const [name, config] of Object.entries(toRun)) {
    const result = await runTest(name, config);
    results.push(result);
  }

  // Summary
  console.log('\n' + '═'.repeat(60));
  console.log('EXAMPLES TEST SUMMARY');
  console.log('═'.repeat(60));
  console.log(`Model: ${path.basename(MODEL_PATH)}`);
  console.log();

  const passed = results.filter(r => r.passed).length;
  const failed = results.filter(r => !r.passed && !r.skipped).length;
  const skipped = results.filter(r => r.skipped).length;
  const totalTime = results.reduce((sum, r) => sum + (r.elapsed || 0), 0).toFixed(1);

  console.log('Results:');
  for (const r of results) {
    const status = r.skipped ? '⏭️ ' : (r.passed ? '✅' : '❌');
    const time = r.elapsed ? ` (${r.elapsed}s)` : '';
    const detail = r.skipped ? ` - ${r.skipReason}` : (r.error ? ` - ${r.error.slice(0, 50)}` : '');
    console.log(`  ${status} ${r.name}${time}${detail}`);
  }

  console.log();
  console.log(`Total: ${passed} passed, ${failed} failed, ${skipped} skipped in ${totalTime}s`);

  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
