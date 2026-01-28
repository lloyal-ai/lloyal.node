#!/usr/bin/env node
/**
 * Infinite context generation with BlinkKV
 *
 * Usage:
 *   node streaming.mjs [model-path]          # Human-readable output
 *   node streaming.mjs [model-path] --jsonl  # JSONL output for testing
 *
 * This example demonstrates:
 * - Generating tokens beyond context window limit
 * - clearAndReseed() for cache-local position reindexing
 * - Per-token perplexity measurement across reseeds
 *
 * Parameters from BlinkKV paper: 2048 context, 4 sinks, 256 tail
 */

import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createContext } from '../../lib/index.js';

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

async function main() {
  // BlinkKV paper parameters: 2048 context, 4 sinks, 256 tail
  const nCtx = 2048;
  const SINK_COUNT = 4;
  const TAIL_SIZE = 256;
  const TARGET_TOKENS = 5000;

  if (!jsonlMode) {
    console.log(`Loading model: ${modelPath}`);
  }

  emit('start', { model: path.basename(modelPath), nCtx, sinkCount: SINK_COUNT, tailSize: TAIL_SIZE, targetTokens: TARGET_TOKENS });

  const ctx = await createContext({
    modelPath,
    contextSize: nCtx,
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
  await ctx.decode(promptTokens, 0, 0);

  // Track all generated tokens (needed for reseeding)
  const allTokens = [...promptTokens];
  // Sink the entire prompt - it's the structural anchor
  const sinks = [...promptTokens];

  if (!jsonlMode) {
    console.log(`\nContext size: ${nCtx}`);
    console.log(`Target tokens: ${TARGET_TOKENS}`);
    console.log(`Sink tokens (prompt): ${sinks.length}`);
    console.log(`Tail size: ${TAIL_SIZE}`);
    console.log(`Cache size after reseed: ${sinks.length + TAIL_SIZE}`);
    console.log(`\nGenerating...\n`);
    process.stdout.write(prompt);
  }

  const tracker = ctx.createPerplexityTracker();
  let cachePos = promptTokens.length;
  let reseedCount = 0;

  for (let t = 0; t < TARGET_TOKENS; t++) {
    // Sample next token
    // NOTE: Token-level repeat penalties are NOT used for long-form generation.
    // llama.cpp's penalty system penalizes individual tokens (not sequences),
    // which degrades prose quality over 100+ tokens as common words accumulate
    // in the penalty buffer. For sequence-level deduplication, use N-gram
    // tracking with logit steering (TTA pattern) instead.
    const token = ctx.sample({
      temperature: 0.8,
      topP: 0.9,
    });
    if (ctx.isStopToken(token)) {
      if (!jsonlMode) {
        console.log('\n[EOS token reached]');
      }
      emit('eos', { tokenIndex: t });
      break;
    }

    // Track surprisal
    const surprisal = ctx.modelSurprisal(token);
    ctx.addSurprisal(tracker, surprisal);

    // Output token
    const text = ctx.tokenToText(token);
    if (!jsonlMode) {
      process.stdout.write(text);
    }
    emit('token', { index: t, token, text, surprisal });

    // Store token and decode
    allTokens.push(token);
    await ctx.decode([token], cachePos++, 0);

    // Cache full? Reseed at boundary
    if (cachePos >= nCtx) {
      const tail = allTokens.slice(-TAIL_SIZE);
      await ctx.clearAndReseed(sinks, tail);
      cachePos = sinks.length + TAIL_SIZE;
      reseedCount++;

      const ppl = ctx.getPerplexity(tracker);
      emit('reseed', { count: reseedCount, tokenIndex: t + 1, ppl });

      if (!jsonlMode) {
        console.log(`\n  [Reseed ${reseedCount} at token ${t + 1}/${TARGET_TOKENS} | PPL: ${ppl.toFixed(2)}]`);
      }
    }

    // Progress indicator every 1000 tokens
    if ((t + 1) % 1000 === 0 && reseedCount === 0 && !jsonlMode) {
      console.log(`\n  [${t + 1}/${TARGET_TOKENS} tokens]`);
    }
  }

  const finalPpl = ctx.getPerplexity(tracker);
  ctx.freePerplexityTracker(tracker);

  const generatedTokens = allTokens.length - promptTokens.length;
  emit('complete', { generatedTokens, reseeds: reseedCount, finalPpl });

  if (!jsonlMode) {
    console.log('\n\n' + '='.repeat(50));
    console.log(`Generated: ${generatedTokens} tokens`);
    console.log(`Reseeds: ${reseedCount}`);
    console.log(`Final perplexity: ${finalPpl.toFixed(2)}`);
    console.log('='.repeat(50));
  }

  ctx.dispose();
}

main().catch((err) => {
  console.error('Error:', err.message);
  process.exit(1);
});
