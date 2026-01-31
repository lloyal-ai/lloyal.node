#!/usr/bin/env node
/**
 * Speculative Decoding with Branch API
 *
 * This example demonstrates speculative decoding using the Branch primitive:
 * - Main branch tracks committed state
 * - Fork a draft branch for speculative generation
 * - Prune draft on rejection, commit accepted tokens to main
 * - Sample bonus token from main at rejection point
 *
 * Real speculative decoding uses a small "draft" model and large "target" model.
 * This example uses the same model for both (demonstrating the mechanics, not speedup).
 *
 * Branch API Benefits:
 * - Atomic fork: KV + logits + sampler + perplexity cloned together
 * - produce/commit separation: sample without KV write, then commit
 * - Shared prefix: forked branches share KV for common prefix
 * - Clean cleanup: prune() removes divergent KV entries
 *
 * References:
 * - Leviathan et al. 2023 "Fast Inference from Transformers via Speculative Decoding"
 * - Chen et al. 2023 "Accelerating Large Language Model Decoding with Speculative Sampling"
 *
 * Usage:
 *   node speculative.mjs [model-path]          # Human-readable output
 *   node speculative.mjs [model-path] --jsonl  # JSONL output for testing
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
const modelPath = args.find((a) => !a.startsWith('--')) || DEFAULT_MODEL;

/** Emit output - JSONL or human-readable */
function emit(event, data) {
  if (jsonlMode) {
    console.log(JSON.stringify({ event, ...data }));
  }
}

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
function simulateVerification(drafts) {
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
  const DRAFT_COUNT = 4;
  const GENERATION_LENGTH = 30;

  if (!jsonlMode) {
    console.log('Speculative Decoding Demo (Branch API)');
    console.log('======================================\n');
    console.log(`Loading model: ${path.basename(modelPath)}`);
  }

  emit('start', {
    model: path.basename(modelPath),
    draftCount: DRAFT_COUNT,
    generationLength: GENERATION_LENGTH,
  });

  const ctx = await createContext({
    modelPath,
    contextSize: 2048,
    nSeqMax: 4, // Enable multi-sequence for fork/verify pattern
  });

  const prompt = 'The quick brown fox';
  if (!jsonlMode) {
    console.log(`\nPrompt: "${prompt}"`);
  }

  // Prefill prompt on seq 0
  const promptTokens = await ctx.tokenize(prompt);
  await ctx.decode(promptTokens, 0, 0);

  // Create main branch - tracks committed state
  // Uses greedy sampling for the "target" model behavior
  const main = Branch.create(ctx, 0, promptTokens.length, {
    temperature: 0.7, // For bonus token sampling
  });
  main.captureLogits();

  const output = [];
  let totalDrafted = 0;
  let totalAccepted = 0;
  let iterations = 0;

  if (!jsonlMode) {
    console.log(
      `\nGenerating ${GENERATION_LENGTH} tokens with speculative decoding...\n`
    );
    process.stdout.write(prompt);
  }

  while (output.length < GENERATION_LENGTH) {
    iterations++;

    // === DRAFT PHASE ===
    // Fork main branch for speculative drafting
    // Draft branch shares KV prefix with main, diverges as it generates
    const draft = main.fork(1);
    draft.reseedSampler(iterations); // Different seed each iteration for diversity

    const drafts = [];

    for (let i = 0; i < DRAFT_COUNT && output.length + drafts.length < GENERATION_LENGTH; i++) {
      // Get entropy BEFORE sampling (from current logits)
      const entropy = ctx.modelEntropy('nats');

      // produce() samples from captured logits (no KV write yet)
      const { token, text, isStop } = draft.produce();

      if (isStop) break;

      drafts.push({ token, text, entropy });

      // commit() accepts token + decodes + captures new logits
      draft.commit(token);
    }

    if (drafts.length === 0) {
      draft.prune();
      break;
    }
    totalDrafted += drafts.length;

    // === VERIFY PHASE ===
    // Simulate verification - in production this compares draft vs target distributions
    const acceptedCount = simulateVerification(drafts);
    totalAccepted += acceptedCount;

    // === CLEANUP DRAFT ===
    // Prune draft branch - removes its divergent KV entries
    // Main branch is unchanged (still at pre-draft position)
    draft.prune();

    // === ACCEPT PHASE ===
    // Commit accepted tokens to main branch
    const accepted = drafts.slice(0, acceptedCount);
    for (const d of accepted) {
      main.commit(d.token);
      if (!jsonlMode) {
        process.stdout.write(d.text);
      }
      emit('token', {
        token: d.token,
        text: d.text,
        entropy: d.entropy,
        accepted: true,
      });
      output.push(d.token);
    }

    // === BONUS TOKEN ===
    // If we rejected some drafts, sample a bonus token from main
    // Main is now at the accepted position with fresh logits
    const rejected = drafts.slice(acceptedCount);
    if (rejected.length > 0) {
      // produce() samples from main's current logits (at rejection point)
      const { token: bonusToken, text: bonusText, isStop } = main.produce();

      if (!isStop) {
        main.commit(bonusToken);
        if (!jsonlMode) {
          process.stdout.write(bonusText);
        }
        emit('token', { token: bonusToken, text: bonusText, bonus: true });
        output.push(bonusToken);
      }
    }

    emit('iteration', {
      iteration: iterations,
      drafted: drafts.length,
      accepted: acceptedCount,
      rejected: rejected.length,
      hasBonus: rejected.length > 0,
    });

    // Check for natural stopping
    if (output.length > 0 && ctx.isStopToken(output[output.length - 1])) {
      break;
    }
  }

  // Cleanup main branch
  main.prune();

  // Statistics
  const acceptRate = totalDrafted > 0 ? totalAccepted / totalDrafted : 0;

  emit('complete', {
    iterations,
    totalDrafted,
    totalAccepted,
    acceptRate,
    outputTokens: output.length,
  });

  if (!jsonlMode) {
    console.log('\n');
    console.log('='.repeat(50));
    console.log('Statistics');
    console.log('='.repeat(50));
    console.log(`  Iterations: ${iterations}`);
    console.log(`  Tokens drafted: ${totalDrafted}`);
    console.log(`  Tokens accepted: ${totalAccepted}`);
    console.log(`  Accept rate: ${(acceptRate * 100).toFixed(1)}%`);
    console.log(`  Output tokens: ${output.length}`);

    console.log('\n' + '='.repeat(50));
    console.log('How Speculative Decoding Works (Branch API)');
    console.log('='.repeat(50));
    console.log(`
  1. MAIN:   Create main branch tracking committed state
  2. FORK:   Fork draft branch (shares KV prefix with main)
  3. DRAFT:  produce/commit N tokens on draft branch
  4. VERIFY: Check draft confidence (entropy threshold)
  5. PRUNE:  Remove draft branch (cleans up divergent KV)
  6. COMMIT: Apply accepted tokens to main branch
  7. BONUS:  Sample one token from main at rejection point
  8. REPEAT: Continue from main's new position

  Branch API Advantages:
  - Atomic fork: KV + logits + sampler copied together
  - Shared prefix: Only divergent KV uses extra memory
  - Clean separation: produce() samples, commit() writes
  - Easy cleanup: prune() handles KV removal
`);
  }

  ctx.dispose();
}

main().catch((err) => {
  console.error('Error:', err.message);
  console.error(err.stack);
  process.exit(1);
});
