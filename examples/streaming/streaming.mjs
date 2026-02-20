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
 * - KV cache clear + re-prefill for cache-local position reindexing
 * - Per-token perplexity measurement across reseeds
 * - Branch API for generation (produce/commit loop)
 *
 * Parameters from BlinkKV paper: 2048 context, 4 sinks, 256 tail
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

// Parse --max-tokens for CI (default 5000)
const maxTokensArg = args.find(a => a.startsWith('--max-tokens='));
const TARGET_TOKENS = maxTokensArg ? parseInt(maxTokensArg.split('=')[1], 10) : 5000;

/** Emit output - JSONL or human-readable */
function emit(event, data) {
  if (jsonlMode) {
    console.log(JSON.stringify({ event, ...data }));
  }
}

async function main() {
  // BlinkKV paper parameters: 2048 context, 4 sinks, 256 tail
  const nCtx = parseInt(process.env.LLAMA_CTX_SIZE || '2048', 10);
  const SINK_COUNT = 4;
  const TAIL_SIZE = 256;

  if (!jsonlMode) {
    console.log(`Loading model: ${modelPath}`);
  }

  emit('start', { model: path.basename(modelPath), nCtx, sinkCount: SINK_COUNT, tailSize: TAIL_SIZE, targetTokens: TARGET_TOKENS });

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

  const samplingParams = { temperature: 0.8, topP: 0.9 };
  let branch = Branch.create(ctx, 0, samplingParams);
  await branch.prefill(promptTokens);

  // Manual PPL tracking (persists across branch reseeds)
  let nllSum = 0, nllCount = 0;
  let reseedCount = 0;

  for (let t = 0; t < TARGET_TOKENS; t++) {
    // NOTE: Token-level repeat penalties are NOT used for long-form generation.
    // llama.cpp's penalty system penalizes individual tokens (not sequences),
    // which degrades prose quality over 100+ tokens as common words accumulate
    // in the penalty buffer. For sequence-level deduplication, use N-gram
    // tracking with logit steering (TTA pattern) instead.
    const { token, isStop } = await branch.produce();
    if (isStop) {
      if (!jsonlMode) {
        console.log('\n[EOS token reached]');
      }
      emit('eos', { tokenIndex: t });
      break;
    }

    // Track surprisal from the logits used by produce()
    const branchLogits = branch.getLogits();
    const surprisal = ctx.modelSurprisal(token, 'nats', branchLogits);
    nllSum += Math.max(0, surprisal);
    nllCount++;

    // Output token
    const text = ctx.tokenToText(token);
    if (!jsonlMode) {
      process.stdout.write(text);
    }
    emit('token', { index: t, token, text, surprisal });

    // Store token and commit (decode + capture new logits)
    allTokens.push(token);
    await branch.commit(token);

    // Cache full? Reseed at boundary
    if (branch.position >= nCtx) {
      const tail = allTokens.slice(-TAIL_SIZE);

      // Destroy current branch, clear KV, create fresh branch with re-prefill
      await branch.prune();
      await ctx.kvCacheClear();
      branch = Branch.create(ctx, 0, samplingParams);
      await branch.prefill([...sinks, ...tail]);

      reseedCount++;

      const ppl = nllCount > 0 ? Math.exp(nllSum / nllCount) : 1;
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

  const finalPpl = nllCount > 0 ? Math.exp(nllSum / nllCount) : 1;
  await branch.prune();

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
