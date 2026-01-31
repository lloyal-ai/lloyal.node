/**
 * Examples Integration Test
 *
 * Runs examples with --jsonl flag and validates structured output.
 * Each example emits JSONL events that we parse and assert on.
 *
 * Usage:
 *   node test/examples.js           # Run all examples
 *   node test/examples.js entropy   # Run specific example
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const MODEL_PATH = path.join(__dirname, '../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf');
const SUMMARY_MODEL_PATH = path.join(__dirname, '../models/slim-summarize.gguf');
if (!fs.existsSync(MODEL_PATH)) {
  console.error('❌ Test model not found!');
  console.error(`   Expected: ${MODEL_PATH}`);
  console.error('   Run: npm run download-models');
  process.exit(1);
}

/**
 * Run an example with --jsonl and collect events
 */
function runExample(scriptPath, timeout = 600000) {
  return new Promise((resolve, reject) => {
    const events = [];
    let stderr = '';

    const child = spawn('node', [scriptPath, MODEL_PATH, '--jsonl'], {
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
            console.log(line);
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
    timeout: 300000,
    validate(events) {
      const start = events.find(e => e.event === 'start');
      assert(start, 'should have start event');
      assert(start.targetTokens === 5000, 'should target 5000 tokens');

      const tokens = events.filter(e => e.event === 'token');
      assert(tokens.length > 100, 'should generate many tokens');

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
    timeout: 300000,
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
    timeout: 600000,
    skip: !fs.existsSync(SUMMARY_MODEL_PATH),
    skipReason: 'slim-summarize.gguf not found',
    validate(events) {
      const start = events.find(e => e.event === 'start');
      assert(start, 'should have start event');
      assert(start.targetTokens === 5000, 'should target 5000 tokens');

      const tokens = events.filter(e => e.event === 'token');
      assert(tokens.length > 100, 'should generate many tokens');

      // Check for token source field
      for (const t of tokens.slice(0, 10)) {
        assert(t.source === 'main', 'token should have source=main');
        assert(typeof t.surprisal === 'number', 'token should have surprisal');
      }

      // With 5000 target tokens and 2048 context, expect reseeds
      const reseeds = events.filter(e => e.event === 'reseed');
      assert(reseeds.length > 0, 'should have at least one reseed');

      // Check reseed events have sink info
      for (const r of reseeds) {
        assert(r.sinkTokens > 0, 'reseed should have sinkTokens > 0');
        assert(r.tailTokens > 0, 'reseed should have tailTokens > 0');
      }

      // If summary model was loaded, check summary events
      const summaryLoaded = events.find(e => e.event === 'summary_loaded');
      if (summaryLoaded) {
        const summaryCompletes = events.filter(e => e.event === 'summary_complete');
        assert(summaryCompletes.length > 0, 'should have summary_complete events');

        for (const sc of summaryCompletes) {
          assert(sc.summaryTokens > 0, 'summary should have tokens');
          assert(sc.compressionRatio > 0, 'summary should have compression ratio');
          assert(sc.durationMs > 0, 'summary should have duration');
        }

        const sinkUpdates = events.filter(e => e.event === 'sink_update');
        assert(sinkUpdates.length > 0, 'should have sink_update events');
      }

      const complete = events.find(e => e.event === 'complete');
      assert(complete, 'should have complete event');
      assert(complete.generatedTokens > 0, 'should generate tokens');
      assert(complete.reseeds > 0, 'should have reseeds > 0');
      assert(complete.finalPpl > 0, 'should have finalPpl');
    },
  },

};

async function runTest(name, config) {
  const fullPath = path.join(__dirname, '../examples', config.path);

  if (config.skip) {
    console.log(`⏭️  ${name}: SKIPPED (${config.skipReason})`);
    return { name, skipped: true };
  }

  console.log(`📜 ${name}: Running...`);

  try {
    const events = await runExample(fullPath, config.timeout);
    config.validate(events);
    console.log(`✅ ${name}: PASSED (${events.length} events)`);
    return { name, passed: true, eventCount: events.length };
  } catch (err) {
    console.log(`❌ ${name}: FAILED`);
    console.log(`   ${err.message}`);
    return { name, passed: false, error: err.message };
  }
}

async function main() {
  const filterName = process.argv[2];

  console.log('=== Examples Integration Test ===\n');
  console.log(`Model: ${path.basename(MODEL_PATH)}\n`);

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
  console.log('\n' + '═'.repeat(50));
  console.log('SUMMARY');
  console.log('═'.repeat(50));

  const passed = results.filter(r => r.passed).length;
  const failed = results.filter(r => !r.passed && !r.skipped).length;
  const skipped = results.filter(r => r.skipped).length;

  for (const r of results) {
    if (r.skipped) {
      console.log(`  ⏭️  ${r.name} (skipped)`);
    } else if (r.passed) {
      console.log(`  ✅ ${r.name}`);
    } else {
      console.log(`  ❌ ${r.name}`);
    }
  }

  console.log(`\nResult: ${passed} passed, ${failed} failed, ${skipped} skipped`);

  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
