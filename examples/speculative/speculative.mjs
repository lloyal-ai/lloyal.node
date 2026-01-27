#!/usr/bin/env node
/**
 * Speculative Decoding with Forkable KV State
 *
 * This example demonstrates the KV cache primitives needed for speculative decoding:
 * - Draft tokens speculatively on a sequence
 * - Fork KV state for verification
 * - Accept/reject tokens and continue from accepted prefix
 *
 * Real speculative decoding uses a small "draft" model and large "target" model.
 * This example uses the same model for both (demonstrating the mechanics, not speedup).
 *
 * References:
 * - Leviathan et al. 2023 "Fast Inference from Transformers via Speculative Decoding"
 * - Chen et al. 2023 "Accelerating Large Language Model Decoding with Speculative Sampling"
 */

import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createContext } from '../../lib/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_MODEL = path.resolve(
  __dirname,
  '../../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf'
);

/**
 * Simulate speculative decoding verification
 *
 * In real speculative decoding:
 * - Draft model generates N tokens quickly (small model or n-gram)
 * - Target model scores all N tokens in a single batch
 * - Compare: if target agrees with draft, accept; else reject and use target's token
 *
 * Here we simulate by accepting tokens with probability based on draft confidence.
 */
function simulateVerification(drafts, ctx) {
  // In production: compare draft probabilities to target probabilities
  // Here: accept high-confidence drafts (low entropy), reject uncertain ones
  let accepted = 0;

  for (const draft of drafts) {
    // Simulate: accept if draft was "confident" (entropy < threshold)
    // Real implementation would compare P_target(token) vs P_draft(token)
    if (draft.entropy < 2.0) {
      accepted++;
    } else {
      break; // First rejection stops the chain
    }
  }

  return accepted;
}

async function main() {
  const modelPath = process.argv[2] || DEFAULT_MODEL;
  const DRAFT_COUNT = 4;
  const GENERATION_LENGTH = 30;

  console.log('Speculative Decoding Demo');
  console.log('=========================\n');
  console.log(`Loading model: ${path.basename(modelPath)}`);

  const ctx = await createContext({
    modelPath,
    contextSize: 2048,
    nSeqMax: 4, // Enable multi-sequence for fork/verify pattern
  });

  const prompt = 'The quick brown fox';
  console.log(`\nPrompt: "${prompt}"`);

  // Prefill prompt on seq 0
  const promptTokens = await ctx.tokenize(prompt);
  await ctx.decode(promptTokens, 0, 0);

  const output = [];
  let pos = promptTokens.length;
  let totalDrafted = 0;
  let totalAccepted = 0;
  let iterations = 0;

  console.log(`\nGenerating ${GENERATION_LENGTH} tokens with speculative decoding...\n`);
  process.stdout.write(prompt);

  while (output.length < GENERATION_LENGTH) {
    iterations++;

    // === DRAFT PHASE ===
    // Generate N speculative tokens greedily (fast, low quality is fine)
    const drafts = [];
    const draftStartPos = pos;

    for (let i = 0; i < DRAFT_COUNT && output.length + drafts.length < GENERATION_LENGTH; i++) {
      const entropy = ctx.modelEntropy('nats');
      const token = ctx.sample({ temperature: 0.0 }); // Greedy drafting

      if (ctx.isStopToken(token)) break;

      drafts.push({
        token,
        text: ctx.tokenToText(token),
        entropy,
      });

      await ctx.decode([token], pos++, 0);
    }

    if (drafts.length === 0) break;
    totalDrafted += drafts.length;

    // === VERIFY PHASE ===
    // Fork KV state for verification (in real impl: run target model on seq 1)
    ctx.kvSeqCopy(0, 1);

    // Simulate verification - in production this compares draft vs target distributions
    const acceptedCount = simulateVerification(drafts, ctx);
    totalAccepted += acceptedCount;

    // === ACCEPT/REJECT ===
    // Keep only the accepted tokens
    const accepted = drafts.slice(0, acceptedCount);
    const rejected = drafts.slice(acceptedCount);

    // Output accepted tokens
    for (const d of accepted) {
      process.stdout.write(d.text);
      output.push(d.token);
    }

    // If we rejected some drafts, we need to:
    // 1. Remove rejected tokens from KV cache
    // 2. Sample one "bonus" token from the target model at the rejection point
    if (rejected.length > 0) {
      // Calculate position where rejection occurred
      const rejectPos = draftStartPos + acceptedCount;

      // Remove rejected tokens from KV cache (positions rejectPos to end)
      await ctx.kvCacheRemove(0, rejectPos, -1);

      // In real speculative decoding: sample from target distribution at rejection point
      // Here we just sample with some temperature for diversity
      const bonusToken = ctx.sample({ temperature: 0.7 });

      if (!ctx.isStopToken(bonusToken)) {
        process.stdout.write(ctx.tokenToText(bonusToken));
        output.push(bonusToken);
        await ctx.decode([bonusToken], rejectPos, 0);
        pos = rejectPos + 1;
      } else {
        pos = rejectPos;
      }
    } else {
      // All drafts accepted - pos is already correct
    }

    // Clean up verification sequence
    await ctx.kvCacheRemove(1, 0, -1);

    // Check for natural stopping
    if (output.length > 0 && ctx.isStopToken(output[output.length - 1])) {
      break;
    }
  }

  console.log('\n');

  // Statistics
  const acceptRate = totalDrafted > 0 ? (totalAccepted / totalDrafted * 100).toFixed(1) : 0;
  console.log('='.repeat(50));
  console.log('Statistics');
  console.log('='.repeat(50));
  console.log(`  Iterations: ${iterations}`);
  console.log(`  Tokens drafted: ${totalDrafted}`);
  console.log(`  Tokens accepted: ${totalAccepted}`);
  console.log(`  Accept rate: ${acceptRate}%`);
  console.log(`  Output tokens: ${output.length}`);

  console.log('\n' + '='.repeat(50));
  console.log('How Speculative Decoding Works');
  console.log('='.repeat(50));
  console.log(`
  1. DRAFT: Generate N tokens quickly (greedy, small model)
  2. FORK:  Copy KV state for verification
  3. VERIFY: Run target model on all N tokens in one batch
  4. ACCEPT: Keep tokens where target agrees with draft
  5. BONUS:  Sample one token from target at first rejection
  6. REPEAT: Continue from accepted prefix + bonus token

  Speedup comes from:
  - Draft model is faster than target model
  - Target verifies N tokens in ONE forward pass (batched)
  - Accept rate determines actual speedup: higher = better
`);

  ctx.dispose();
}

main().catch((err) => {
  console.error('Error:', err.message);
  console.error(err.stack);
  process.exit(1);
});
