#!/usr/bin/env node
/**
 * Infinite context generation with BlinkKV + tsampler N-gram deduplication
 *
 * This example demonstrates:
 * - TypeScript sampling via tsampler (TTA pattern)
 * - N-gram tracking to detect sequence repetition
 * - Logit steering to prevent repeated sequences
 * - Branch API for KV management (prefill/commit)
 * - KV cache clear + re-prefill for infinite context
 *
 * The key insight: llama.cpp's token-level penalties degrade prose quality.
 * Instead, we track N-grams at the app level and steer away from repeats.
 *
 * Usage:
 *   node streaming-tsampler.mjs [model-path]          # Human-readable output
 *   node streaming-tsampler.mjs [model-path] --jsonl  # JSONL output for testing
 */

import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createContext, Branch } from '../../lib/index.js';

// Import tsampler from npm package
import {
  sampleWithStrategy,
  // TokenHistoryTracker, // Disabled - matching baseline
  Xoroshiro128Plus,
  SamplerWorkspace,
} from '@lloyal-labs/tsampler';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_MODEL = path.resolve(
  __dirname,
  '../../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf'
);

// Parse args
const args = process.argv.slice(2);
const jsonlMode = args.includes('--jsonl');
const modelPath = args.find(a => !a.startsWith('--')) || DEFAULT_MODEL;

// Parse --max-tokens for CI (default 5000)
const maxTokensArg = args.find(a => a.startsWith('--max-tokens='));
const TARGET_TOKENS = maxTokensArg ? parseInt(maxTokensArg.split('=')[1], 10) : 5000;

/** Emit output - JSONL or human-readable */
function emit(event, data) {
  if (jsonlMode) {
    console.log(JSON.stringify({ event, ...data }));
  }
}

/**
 * N-gram tracker for sequence-level repetition detection (threshold-based)
 *
 * Tracks N-grams and their followers. Only blocks when the SAME N-gram → follower
 * pattern is seen K times (threshold), indicating true looping behavior rather
 * than coincidental reuse.
 */
class NgramTracker {
  constructor(n = 4, threshold = 2) {
    this.n = n;
    this.threshold = threshold; // Block after seeing same pattern K times
    this.ngrams = new Map(); // ngram key -> Map<follower, count>
    this.recentTokens = [];
  }

  /**
   * Record a token and update N-gram history
   */
  accept(token) {
    this.recentTokens.push(token);

    // Once we have enough tokens, record the N-gram and what followed
    if (this.recentTokens.length > this.n) {
      const ngramTokens = this.recentTokens.slice(-this.n - 1, -1);
      const ngramKey = ngramTokens.join(',');

      // Get or create follower counts for this N-gram
      if (!this.ngrams.has(ngramKey)) {
        this.ngrams.set(ngramKey, new Map());
      }
      const followers = this.ngrams.get(ngramKey);

      // Increment count for this follower
      const count = followers.get(token) || 0;
      followers.set(token, count + 1);
    }
  }

  /**
   * Check if current context would repeat an N-gram above threshold
   * @returns {number|null} Token to block, or null if below threshold
   */
  getBlockedToken() {
    if (this.recentTokens.length < this.n) {
      return null;
    }

    const currentNgram = this.recentTokens.slice(-this.n);
    const ngramKey = currentNgram.join(',');

    const followers = this.ngrams.get(ngramKey);
    if (!followers) {
      return null;
    }

    // Find follower that has hit threshold (true loop)
    for (const [follower, count] of followers) {
      if (count >= this.threshold) {
        return follower;
      }
    }

    return null;
  }

  /**
   * Get stats for logging
   */
  stats() {
    let totalPatterns = 0;
    for (const followers of this.ngrams.values()) {
      totalPatterns += followers.size;
    }
    return {
      uniqueNgrams: this.ngrams.size,
      totalPatterns,
      totalTokens: this.recentTokens.length,
    };
  }
}

async function main() {
  // BlinkKV parameters
  const nCtx = parseInt(process.env.LLAMA_CTX_SIZE || '2048', 10);
  const TAIL_SIZE = 256;
  const NGRAM_SIZE = 6; // Track 6-grams for sequence detection
  const BLOCK_THRESHOLD = 2; // Only block after seeing same pattern K times

  if (!jsonlMode) {
    console.log(`Loading model: ${modelPath}`);
  }

  emit('start', { model: path.basename(modelPath), nCtx, tailSize: TAIL_SIZE, targetTokens: TARGET_TOKENS, ngramSize: NGRAM_SIZE, blockThreshold: BLOCK_THRESHOLD });

  const ctx = await createContext({
    modelPath,
    nCtx,
  });

  const prompt = `Write a comprehensive guide to machine learning, covering the following topics in extreme detail with examples, code snippets, and mathematical formulas:

1. Linear Regression - derivation, implementation, regularization
2. Logistic Regression - binary and multiclass
3. Neural Networks - backpropagation, activation functions
4. Convolutional Neural Networks - architectures, pooling, stride
5. Recurrent Neural Networks - LSTM, GRU, attention
6. Transformers - self-attention, positional encoding
7. Optimization - SGD, Adam, learning rate schedules
8. Regularization - dropout, batch normalization, weight decay

Begin:

# Comprehensive Machine Learning Guide

## Chapter 1: Linear Regression

`;
  if (!jsonlMode) {
    console.log(`\nPrompt: "${prompt.slice(0, 100)}..."`);
  }

  const promptTokens = await ctx.tokenize(prompt);

  // Track all generated tokens
  const allTokens = [...promptTokens];
  const sinks = [...promptTokens]; // Sink the entire prompt

  // tsampler setup
  const prng = new Xoroshiro128Plus(42); // Fixed seed for reproducibility
  // const tokenHistory = new TokenHistoryTracker(32); // Disabled - matching baseline
  const workspace = new SamplerWorkspace(256);

  // N-gram tracker for sequence-level deduplication
  const ngramTracker = new NgramTracker(NGRAM_SIZE, BLOCK_THRESHOLD);

  // Seed N-gram tracker with prompt tokens
  for (const token of promptTokens) {
    ngramTracker.accept(token);
  }

  if (!jsonlMode) {
    console.log(`\nContext size: ${nCtx}`);
    console.log(`Target tokens: ${TARGET_TOKENS}`);
    console.log(`Sink tokens (prompt): ${sinks.length}`);
    console.log(`Tail size: ${TAIL_SIZE}`);
    console.log(`N-gram size: ${NGRAM_SIZE}, block threshold: ${BLOCK_THRESHOLD}`);
    console.log(`\nGenerating with tsampler + N-gram deduplication (threshold-based)...\n`);
    process.stdout.write(prompt);
  }

  // Branch used purely for KV management — sampling done externally via tsampler
  let branch = Branch.create(ctx, 0, { temperature: 0 });
  await branch.prefill(promptTokens);

  // Manual PPL tracking (persists across branch reseeds)
  let nllSum = 0, nllCount = 0;
  let reseedCount = 0;
  let blockedCount = 0;

  for (let t = 0; t < TARGET_TOKENS; t++) {
    // Get logits from branch snapshot
    const originalLogits = branch.getLogits();
    const logits = new Float32Array(originalLogits);

    // N-gram deduplication: Check if we're about to repeat a sequence
    const blockedToken = ngramTracker.getBlockedToken();
    const wasBlocked = blockedToken !== null && blockedToken < logits.length;
    if (wasBlocked) {
      // Steer away from the repeat by setting logit to -Infinity
      logits[blockedToken] = -Infinity;
      blockedCount++;
    }

    // Sample with tsampler (TTA pattern)
    // Match baseline params exactly: temp 0.8, topP 0.9, no topK, no penalties
    const token = sampleWithStrategy(logits, {
      params: {
        temperature: 0.8,
        topP: 0.9,
      },
      workspace,
      prng,
    });

    // Check for EOS
    if (ctx.isStopToken(token)) {
      if (!jsonlMode) {
        console.log('\n[EOS token reached]');
      }
      emit('eos', { tokenIndex: t });
      break;
    }

    // Accept token into trackers
    // tokenHistory.accept(token); // Disabled - matching baseline
    ngramTracker.accept(token);

    // Track surprisal from original (unmodified) logits
    const surprisal = ctx.modelSurprisal(token, 'nats', originalLogits);
    nllSum += Math.max(0, surprisal);
    nllCount++;

    // Output token
    const text = ctx.tokenToText(token);
    if (!jsonlMode) {
      process.stdout.write(text);
    }
    emit('token', { index: t, token, text, surprisal, blocked: wasBlocked });

    // Store and advance KV (no sampler accept — we're using tsampler externally)
    allTokens.push(token);
    await branch.commit(token);

    // Cache full? Reseed at boundary
    if (branch.position >= nCtx) {
      const tail = allTokens.slice(-TAIL_SIZE);

      // Destroy current branch, clear KV, create fresh branch with re-prefill
      await branch.prune();
      await ctx.kvCacheClear();
      branch = Branch.create(ctx, 0, { temperature: 0 });
      await branch.prefill([...sinks, ...tail]);

      reseedCount++;

      const ppl = nllCount > 0 ? Math.exp(nllSum / nllCount) : 1;
      const stats = ngramTracker.stats();

      emit('reseed', { count: reseedCount, tokenIndex: t + 1, ppl, blockedCount, uniqueNgrams: stats.uniqueNgrams });

      if (!jsonlMode) {
        console.log(`\n  [Reseed ${reseedCount} at token ${t + 1}/${TARGET_TOKENS} | PPL: ${ppl.toFixed(2)} | Blocked: ${blockedCount} | Unique ${NGRAM_SIZE}-grams: ${stats.uniqueNgrams}]`);
      }
    }

    // Progress every 1000 tokens
    if ((t + 1) % 1000 === 0 && branch.position < nCtx && !jsonlMode) {
      const stats = ngramTracker.stats();
      console.log(`\n  [${t + 1}/${TARGET_TOKENS} | Blocked repeats: ${blockedCount} | Unique ${NGRAM_SIZE}-grams: ${stats.uniqueNgrams}]`);
    }
  }

  const finalPpl = nllCount > 0 ? Math.exp(nllSum / nllCount) : 1;
  const finalStats = ngramTracker.stats();
  await branch.prune();

  const generatedTokens = allTokens.length - promptTokens.length;
  emit('complete', {
    generatedTokens,
    reseeds: reseedCount,
    finalPpl,
    blockedCount,
    uniqueNgrams: finalStats.uniqueNgrams,
  });

  if (!jsonlMode) {
    console.log('\n\n' + '='.repeat(60));
    console.log(`Generated: ${generatedTokens} tokens`);
    console.log(`Reseeds: ${reseedCount}`);
    console.log(`Final perplexity: ${finalPpl.toFixed(2)}`);
    console.log(`Sequence repeats blocked: ${blockedCount}`);
    console.log(`Unique ${NGRAM_SIZE}-grams tracked: ${finalStats.uniqueNgrams}`);
    console.log('='.repeat(60));
  }

  ctx.dispose();
}

main().catch((err) => {
  console.error('Error:', err.message);
  console.error(err.stack);
  process.exit(1);
});
