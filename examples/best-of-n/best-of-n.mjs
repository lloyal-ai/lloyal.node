#!/usr/bin/env node
/**
 * Best-of-N Sampling with Perplexity Selection
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
 * After prefilling the prompt, we capture the logits. When forking to multiple
 * sequences, each candidate's FIRST token must be sampled from these captured
 * logits (not from whatever sequence was last decoded). This ensures all
 * candidates start from the same probability distribution.
 */

import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createContext } from '../../lib/index.js';

// Import tsampler for sampling (metrics handled by native API)
import {
  sampleWithStrategy,
  SamplerWorkspace,
  Xoroshiro128Plus,
} from '@lloyal-labs/tsampler';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_MODEL = path.resolve(
  __dirname,
  '../../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf'
);

/**
 * Generate a single completion and track perplexity
 *
 * @param ctx - The inference context
 * @param capturedLogits - Logits captured after prefill (for first token)
 * @param seqId - Sequence ID to use for this candidate
 * @param samplerParams - tsampler parameters (temperature, topP, etc.)
 * @param workspace - Preallocated tsampler workspace
 * @param prng - PRNG instance for this candidate
 * @param promptLen - Length of the prompt in tokens
 * @param maxTokens - Maximum tokens to generate
 */
async function generateOne(
  ctx,
  capturedLogits,
  seqId,
  samplerParams,
  workspace,
  prng,
  promptLen,
  maxTokens
) {
  // Clear any stale KV data for this sequence (defensive)
  await ctx.kvCacheRemove(seqId, 0, -1);

  // Fork KV state from seq 0 (prompt)
  ctx.kvSeqCopy(0, seqId);

  const tracker = ctx.createPerplexityTracker();
  const output = [];
  let pos = promptLen;

  // First token: sample from captured prefill logits
  // This ensures all candidates start from the same distribution
  let token = sampleWithStrategy(capturedLogits, {
    params: samplerParams,
    workspace,
    prng,
  });

  if (ctx.isStopToken(token)) {
    ctx.freePerplexityTracker(tracker);
    await ctx.kvCacheRemove(seqId, 0, -1);
    return { text: '', ppl: 999, tokenCount: 0 };
  }

  // Track surprisal for first token using captured logits (native API)
  const firstSurprisal = ctx.modelSurprisal(token, 'nats', capturedLogits);
  if (Number.isFinite(firstSurprisal)) {
    ctx.addSurprisal(tracker, firstSurprisal);
  }

  output.push(token);
  await ctx.decode([token], pos++, seqId);

  // Subsequent tokens: sample from fresh logits (now specific to this sequence)
  for (let t = 1; t < maxTokens; t++) {
    // Get fresh logits for this sequence
    const logits = new Float32Array(ctx.getLogits());

    // Sample with tsampler
    token = sampleWithStrategy(logits, {
      params: samplerParams,
      workspace,
      prng,
    });

    if (ctx.isStopToken(token)) break;

    // Track surprisal using context's method (reads current logits)
    const surprisal = ctx.modelSurprisal(token);
    if (Number.isFinite(surprisal)) {
      ctx.addSurprisal(tracker, surprisal);
    }

    output.push(token);
    await ctx.decode([token], pos++, seqId);
  }

  const ppl = ctx.getPerplexity(tracker);
  const text = await ctx.detokenize(output);
  ctx.freePerplexityTracker(tracker);

  // Clean up this sequence
  await ctx.kvCacheRemove(seqId, 0, -1);

  return {
    text,
    ppl: Number.isFinite(ppl) ? ppl : 999,
    tokenCount: output.length,
  };
}

async function main() {
  const modelPath = process.argv[2] || DEFAULT_MODEL;
  const N = 5;           // Number of candidates
  const MAX_TOKENS = 60;
  const HIGH_TEMP = 0.9; // High temp for diversity
  const LOW_TEMP = 0.3;  // Low temp for single baseline

  console.log('Best-of-N Sampling Demo');
  console.log('=======================\n');
  console.log('Why best-of-n works:');
  console.log('  1. Generate N candidates with HIGH temperature (diverse)');
  console.log('  2. Score each by perplexity (model confidence)');
  console.log('  3. Select LOWEST perplexity (most coherent)\n');
  console.log(`Loading model: ${path.basename(modelPath)}`);

  const ctx = await createContext({
    modelPath,
    contextSize: 2048,
    nSeqMax: N + 2, // Need slots for N candidates + baseline + trunk
  });

  // Use chat template for consistent behavior
  const userPrompt = 'Write a creative opening sentence for a fantasy novel.';
  const messages = [{ role: 'user', content: userPrompt }];
  const { prompt } = await ctx.formatChat(JSON.stringify(messages));

  console.log(`\nPrompt: "${userPrompt}"`);

  // Prefill prompt on seq 0
  const promptTokens = await ctx.tokenize(prompt);
  await ctx.decode(promptTokens, 0, 0);

  // Capture logits after prefill - these will be used for ALL candidates' first token
  // CRITICAL: Copy the logits buffer because it becomes invalid after next decode()
  const capturedLogits = new Float32Array(ctx.getLogits());

  console.log(`\nPrefill complete. Vocab size: ${capturedLogits.length}`);

  // tsampler setup - shared workspace, per-candidate PRNGs
  const workspace = new SamplerWorkspace(256);

  // Sampling params
  const highTempParams = { temperature: HIGH_TEMP, topP: 0.95 };
  const lowTempParams = { temperature: LOW_TEMP, topP: 0.95 };

  // === Baseline: Single generation with low temperature ===
  console.log('\n' + '='.repeat(70));
  console.log('BASELINE: Single generation (T=0.3)');
  console.log('='.repeat(70));

  const baselinePrng = new Xoroshiro128Plus(42);
  const baseline = await generateOne(
    ctx,
    capturedLogits,
    1,
    lowTempParams,
    workspace,
    baselinePrng,
    promptTokens.length,
    MAX_TOKENS
  );
  console.log(`  PPL: ${baseline.ppl.toFixed(2)} | "${baseline.text}"`);

  // === Best-of-N: Multiple candidates with high temperature ===
  console.log('\n' + '='.repeat(70));
  console.log(`BEST-OF-${N}: Generate ${N} candidates (T=${HIGH_TEMP}), select lowest PPL`);
  console.log('='.repeat(70));

  const candidates = [];
  for (let i = 0; i < N; i++) {
    // Each candidate gets its own PRNG for reproducibility
    const prng = new Xoroshiro128Plus(1000 + i);

    const result = await generateOne(
      ctx,
      capturedLogits,
      i + 2,  // seqId: 2, 3, 4, 5, 6 (baseline uses 1)
      highTempParams,
      workspace,
      prng,
      promptTokens.length,
      MAX_TOKENS
    );
    candidates.push(result);

    const truncated = result.text.length > 55
      ? result.text.slice(0, 55) + '...'
      : result.text;
    console.log(`  [${i + 1}] PPL: ${result.ppl.toFixed(2).padStart(6)} | "${truncated}"`);
  }

  // Select best
  const best = candidates.reduce((a, b) => (a.ppl < b.ppl ? a : b));
  const worst = candidates.reduce((a, b) => (a.ppl > b.ppl ? a : b));
  const bestIdx = candidates.indexOf(best) + 1;

  // === Results ===
  console.log('\n' + '='.repeat(70));
  console.log('RESULTS');
  console.log('='.repeat(70));

  console.log(`\n  Best candidate [${bestIdx}] (PPL ${best.ppl.toFixed(2)}):`);
  console.log(`    "${best.text}"`);

  console.log(`\n  Baseline (PPL ${baseline.ppl.toFixed(2)}):`);
  console.log(`    "${baseline.text}"`);

  // Analysis
  const improvement = ((baseline.ppl - best.ppl) / baseline.ppl * 100).toFixed(1);
  const pplRange = (worst.ppl - best.ppl).toFixed(2);

  console.log('\n  Analysis:');
  console.log(`    - PPL range across candidates: ${best.ppl.toFixed(2)} - ${worst.ppl.toFixed(2)} (Δ${pplRange})`);
  if (best.ppl < baseline.ppl) {
    console.log(`    - Best-of-${N} beat baseline by ${improvement}% lower PPL`);
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
    After prefilling the prompt, we capture the logits. All N candidates
    sample their FIRST token from these same captured logits, ensuring
    they all start from the same probability distribution. This is critical
    for fair comparison - otherwise later candidates would sample from
    earlier candidates' states!
`);

  ctx.dispose();
}

main().catch((err) => {
  console.error('Error:', err.message);
  console.error(err.stack);
  process.exit(1);
});
