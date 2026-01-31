#!/usr/bin/env node
/**
 * Best-of-N Sampling with Perplexity Selection (Parallel Streaming)
 *
 * Demonstrates why best-of-n beats single generation:
 * - Generate N candidates with high temperature (diverse)
 * - Select best by perplexity (model's confidence in its output)
 * - Lower perplexity = more coherent, higher quality
 *
 * Based on: "Best-of-N" / "Rejection Sampling" used in RLHF pipelines
 * See: Stiennon et al. 2020 "Learning to summarize from human feedback"
 *
 * KEY IMPLEMENTATION DETAIL:
 * Uses the Branch API for parallel generation. After prefilling the prompt,
 * we create a root branch and call captureLogits(). When forking to multiple
 * candidates, each fork inherits the root's logits snapshot, ensuring all
 * candidates start from the same probability distribution.
 *
 * Usage:
 *   node best-of-n.mjs [model-path]          # Human-readable output
 *   node best-of-n.mjs [model-path] --jsonl  # JSONL output for testing
 */

import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createContext, Branch } from '../../lib/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_MODEL = path.resolve(
  __dirname,
  '../../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf'
);

// Parse args
const args = process.argv.slice(2);
const jsonlMode = args.includes('--jsonl');
const modelPath = args.find(a => !a.startsWith('--')) || DEFAULT_MODEL;

/** Emit output - JSONL or human-readable */
function emit(event, data) {
  if (jsonlMode) {
    console.log(JSON.stringify({ event, ...data }));
  }
}

/**
 * Generate a single completion using Branch API
 *
 * @param {Branch} branch - Branch to generate with
 * @param {number} maxTokens - Maximum tokens to generate
 * @param {object} ctx - Context for detokenization
 * @returns {Promise<{ text: string, ppl: number, tokenCount: number }>}
 */
async function generateWithBranch(branch, maxTokens, ctx) {
  const tokens = [];

  for (let t = 0; t < maxTokens; t++) {
    const { token, isStop } = branch.produce();
    if (isStop) break;
    tokens.push(token);
    branch.commit(token);
  }

  const ppl = branch.perplexity;
  const text = await ctx.detokenize(tokens);

  return {
    text,
    ppl: Number.isFinite(ppl) ? ppl : 999,
    tokenCount: tokens.length,
  };
}

async function main() {
  const N = 5;           // Number of candidates
  const MAX_TOKENS = 60;
  const HIGH_TEMP = 0.9; // High temp for diversity
  const LOW_TEMP = 0.3;  // Low temp for single baseline

  if (!jsonlMode) {
    console.log('Best-of-N Sampling Demo (Parallel Streaming)');
    console.log('=============================================\n');
    console.log('Why best-of-n works:');
    console.log('  1. Generate N candidates with HIGH temperature (diverse)');
    console.log('  2. Score each by perplexity (model confidence)');
    console.log('  3. Select LOWEST perplexity (most coherent)\n');
    console.log(`Loading model: ${path.basename(modelPath)}`);
  }

  emit('start', { model: path.basename(modelPath), n: N, maxTokens: MAX_TOKENS, highTemp: HIGH_TEMP, lowTemp: LOW_TEMP });

  const ctx = await createContext({
    modelPath,
    contextSize: 2048,
    nSeqMax: N + 2, // Need slots for N candidates + baseline + trunk
  });

  // Use chat template for consistent behavior
  const userPrompt = 'Write a creative opening sentence for a fantasy novel.';
  const messages = [{ role: 'user', content: userPrompt }];
  const { prompt } = await ctx.formatChat(JSON.stringify(messages));

  if (!jsonlMode) {
    console.log(`\nPrompt: "${userPrompt}"`);
  }

  // Prefill prompt on seq 0
  const promptTokens = await ctx.tokenize(prompt);
  await ctx.decode(promptTokens, 0, 0);

  if (!jsonlMode) {
    console.log(`\nPrefill complete. Prompt length: ${promptTokens.length} tokens`);
  }

  // CRITICAL: Create root branch IMMEDIATELY after prefill to capture logits
  // The root branch stores a snapshot of the logits for fork operations
  const root = Branch.create(ctx, 0, promptTokens.length, {
    temperature: HIGH_TEMP,
    topP: 0.95,
  });
  root.captureLogits();

  // === Baseline: Single generation with low temperature ===
  if (!jsonlMode) {
    console.log('\n' + '='.repeat(70));
    console.log('BASELINE: Single generation (T=0.3)');
    console.log('='.repeat(70));
  }

  // Create baseline branch on seq 1 with LOW temperature
  // We capture logits now (they're still the prefill logits in context)
  const baselineBranch = Branch.create(ctx, 1, promptTokens.length, {
    temperature: LOW_TEMP,
    topP: 0.95,
  });
  baselineBranch.captureLogits();

  // Copy KV cache from seq 0 (prompt) to seq 1 (baseline)
  ctx.kvSeqCopy(0, 1);

  const baseline = await generateWithBranch(baselineBranch, MAX_TOKENS, ctx);

  emit('baseline', { ppl: baseline.ppl, text: baseline.text, tokenCount: baseline.tokenCount });

  if (!jsonlMode) {
    console.log(`  PPL: ${baseline.ppl.toFixed(2)} | "${baseline.text}"`);
  }

  baselineBranch.prune();

  // === Best-of-N: Parallel candidates with high temperature ===
  if (!jsonlMode) {
    console.log('\n' + '='.repeat(70));
    console.log(`BEST-OF-${N}: Generate ${N} candidates in parallel (T=${HIGH_TEMP})`);
    console.log('='.repeat(70));
  }

  // Fork N candidate branches from root
  // Each fork gets: copied logits snapshot + copied KV cache + copied sampler
  // CRITICAL: Reseed each branch's sampler for diversity (otherwise all produce identical output)
  const branches = [];
  for (let i = 0; i < N; i++) {
    const seqId = i + 2;  // seqIds: 2, 3, 4, 5, 6 (baseline used 1)
    const branch = root.fork(seqId);
    branch.reseedSampler(1000 + i);  // Unique seed per branch
    branches.push(branch);
  }

  // Parallel generation loop - interleaved round-robin
  const results = branches.map(() => ({ tokens: [], done: false }));

  for (let t = 0; t < MAX_TOKENS; t++) {
    let anyActive = false;

    for (let i = 0; i < N; i++) {
      if (results[i].done) continue;
      anyActive = true;

      const { token, text, isStop } = branches[i].produce();

      if (isStop) {
        results[i].done = true;
        continue;
      }

      results[i].tokens.push(token);
      branches[i].commit(token);

      emit('token', { candidateIndex: i, text, index: t });
    }

    if (!anyActive) break;
  }

  // Finalize results - get perplexity and detokenize
  const candidates = [];
  for (let i = 0; i < N; i++) {
    const ppl = branches[i].perplexity;
    const text = await ctx.detokenize(results[i].tokens);
    const tokenCount = results[i].tokens.length;

    candidates.push({
      text,
      ppl: Number.isFinite(ppl) ? ppl : 999,
      tokenCount,
    });

    emit('candidate', { index: i + 1, ppl: candidates[i].ppl, text, tokenCount });

    if (!jsonlMode) {
      const truncated = text.length > 55 ? text.slice(0, 55) + '...' : text;
      console.log(`  [${i + 1}] PPL: ${candidates[i].ppl.toFixed(2).padStart(6)} | "${truncated}"`);
    }

    branches[i].prune();
  }
  root.prune();

  // Select best
  const best = candidates.reduce((a, b) => (a.ppl < b.ppl ? a : b));
  const worst = candidates.reduce((a, b) => (a.ppl > b.ppl ? a : b));
  const bestIdx = candidates.indexOf(best) + 1;

  // Analysis
  const improvement = (baseline.ppl - best.ppl) / baseline.ppl;
  const pplRange = worst.ppl - best.ppl;

  emit('complete', {
    bestIndex: bestIdx,
    bestPpl: best.ppl,
    bestText: best.text,
    worstPpl: worst.ppl,
    baselinePpl: baseline.ppl,
    pplRange,
    improvement,
    bestBeatBaseline: best.ppl < baseline.ppl,
  });

  if (!jsonlMode) {
    // === Results ===
    console.log('\n' + '='.repeat(70));
    console.log('RESULTS');
    console.log('='.repeat(70));

    console.log(`\n  Best candidate [${bestIdx}] (PPL ${best.ppl.toFixed(2)}):`);
    console.log(`    "${best.text}"`);

    console.log(`\n  Baseline (PPL ${baseline.ppl.toFixed(2)}):`);
    console.log(`    "${baseline.text}"`);

    console.log('\n  Analysis:');
    console.log(`    - PPL range across candidates: ${best.ppl.toFixed(2)} - ${worst.ppl.toFixed(2)} (Δ${pplRange.toFixed(2)})`);
    if (best.ppl < baseline.ppl) {
      console.log(`    - Best-of-${N} beat baseline by ${(improvement * 100).toFixed(1)}% lower PPL`);
    } else {
      console.log(`    - Baseline was already good (low temp = focused)`);
    }

    console.log('\n' + '='.repeat(70));
    console.log('KEY INSIGHT');
    console.log('='.repeat(70));
    console.log(`
  Perplexity = exp(average surprisal) = "how surprised is the model?"

  Lower PPL = model is confident in what it wrote = usually more coherent
  Higher PPL = model was uncertain = may have inconsistencies

  Best-of-N trades compute for quality:
    - High temp generates diverse candidates (explore the space)
    - PPL filtering selects the coherent ones (exploit quality)

  Implementation note:
    Uses the Branch API for parallel generation. After prefilling the
    prompt, we create a root branch and capture its logits. When forking
    to N candidates, each fork inherits the root's logits snapshot,
    ensuring all candidates start from the same probability distribution.
    Generation happens in round-robin fashion, interleaving tokens across
    all candidates.
`);
  }

  ctx.dispose();
}

main().catch((err) => {
  console.error('Error:', err.message);
  console.error(err.stack);
  process.exit(1);
});
