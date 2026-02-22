#!/usr/bin/env node
/**
 * EDT vs Fixed Temperature Comparison
 *
 * Based on: Zhang et al. 2024 "EDT: Improving Large Language Models'
 * Generation by Entropy-based Dynamic Temperature Sampling"
 * https://arxiv.org/abs/2403.14541
 *
 * This example demonstrates:
 * - EDT formula: T = T₀ · N^(θ/Entropy)
 * - Side-by-side comparison with fixed temperature
 * - Different prompt types: factual, creative, mixed
 * - Branch API for token generation (produce/commit loop)
 *
 *
 * Usage:
 *   npx tsx entropy.ts [model-path]          # Human-readable output
 *   npx tsx entropy.ts [model-path] --jsonl  # JSONL output for testing
 */

import * as path from 'node:path';
import { createContext, Branch } from '../../dist/index.js';
import type { SessionContext } from '../../dist/index.js';

const DEFAULT_MODEL = path.resolve(
  __dirname,
  '../../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf'
);

// Parse args
const args = process.argv.slice(2);
const jsonlMode = args.includes('--jsonl');
const modelPath = args.find(a => !a.startsWith('--')) || DEFAULT_MODEL;

/** Emit output - JSONL or human-readable */
function emit(event: string, data: Record<string, unknown>): void {
  if (jsonlMode) {
    console.log(JSON.stringify({ event, ...data }));
  }
}

// EDT parameters (Zhang et al. 2024)
const T0 = 1.0;    // Max temperature bound
const N = 0.8;     // Base (paper uses 0.8)
const THETA = 1.5; // Scale factor

/**
 * Calculate EDT temperature from entropy
 */
function edtTemperature(entropy: number): number {
  const safeEntropy = Math.max(entropy, 0.1);
  return T0 * Math.pow(N, THETA / safeEntropy);
}

/**
 * Generate with a specific sampling strategy
 *
 * Uses Branch API with per-token setSamplerParams() for EDT adaptation.
 * Each token gets a temperature computed from the current logit entropy.
 */
async function generate(ctx: SessionContext, prompt: string, strategy: number | 'edt', strategyName: string, maxTokens: number = 50): Promise<{text: string; avgEntropy: number; avgTemp: number; tokenCount: number; temps: number[]; entropies: number[]}> {
  const messages = [{ role: 'user', content: prompt }];
  const { prompt: formatted } = await ctx.formatChat(JSON.stringify(messages));
  const tokens = await ctx.tokenize(formatted);

  const baseTemp = strategy === 'edt' ? 0.8 : strategy;
  const branch = Branch.create(ctx, 0, { temperature: baseTemp, topP: 0.9 });
  await branch.prefill(tokens);

  const output: number[] = [];
  const temps: number[] = [];
  const entropies: number[] = [];

  for (let i = 0; i < maxTokens; i++) {
    const entropy = branch.modelEntropy('nats');
    entropies.push(entropy);

    const temp = strategy === 'edt' ? edtTemperature(entropy) : strategy;
    temps.push(temp);

    if (strategy === 'edt') branch.setSamplerParams({ temperature: temp, topP: 0.9 });

    const { token, isStop } = await branch.produce();
    if (isStop) break;

    const text = ctx.tokenToText(token);
    emit('token', { strategy: strategyName, token, text, entropy, temp });

    output.push(token);
    await branch.commit(token);
  }

  await branch.prune();

  const text = await ctx.detokenize(output);
  const avgEntropy = entropies.length > 0 ? entropies.reduce((a, b) => a + b, 0) / entropies.length : 0;
  const avgTemp = temps.length > 0 ? temps.reduce((a, b) => a + b, 0) / temps.length : 0;

  return { text, avgEntropy, avgTemp, tokenCount: output.length, temps, entropies };
}

type GenerateResult = Awaited<ReturnType<typeof generate>>;

/**
 * Run comparison for a single prompt
 */
async function compareStrategies(ctx: SessionContext, prompt: string, label: string): Promise<{fixed: GenerateResult; edt: GenerateResult}> {
  if (!jsonlMode) {
    console.log(`\n${'='.repeat(70)}`);
    console.log(`${label}: "${prompt}"`);
    console.log('='.repeat(70));
  }

  // Run with fixed temperature
  const fixed = await generate(ctx, prompt, 0.7, 'fixed');

  // Run with EDT
  const edt = await generate(ctx, prompt, 'edt', 'edt');

  // Show temperature adaptation
  const lowEntropyTokens = edt.entropies.filter(e => e < 1.0).length;
  const highEntropyTokens = edt.entropies.filter(e => e > 3.5).length;

  if (jsonlMode) {
    emit('comparison', {
      label,
      prompt,
      fixed: {
        text: fixed.text,
        tokenCount: fixed.tokenCount,
        avgEntropy: fixed.avgEntropy,
      },
      edt: {
        text: edt.text,
        tokenCount: edt.tokenCount,
        avgEntropy: edt.avgEntropy,
        avgTemp: edt.avgTemp,
        lowEntropyTokens,
        highEntropyTokens,
      },
    });
  } else {
    console.log(`\n  Fixed (T=0.7):`);
    console.log(`    Output: ${fixed.text.slice(0, 100)}${fixed.text.length > 100 ? '...' : ''}`);
    console.log(`    Tokens: ${fixed.tokenCount} | Avg entropy: ${fixed.avgEntropy.toFixed(2)} nats`);

    console.log(`\n  EDT (adaptive):`);
    console.log(`    Output: ${edt.text.slice(0, 100)}${edt.text.length > 100 ? '...' : ''}`);
    console.log(`    Tokens: ${edt.tokenCount} | Avg entropy: ${edt.avgEntropy.toFixed(2)} nats | Avg T: ${edt.avgTemp.toFixed(2)}`);
    console.log(`    Adaptation: ${lowEntropyTokens} confident tokens (T↓), ${highEntropyTokens} uncertain tokens (T↑)`);
  }

  return { fixed, edt };
}

async function main(): Promise<void> {
  if (!jsonlMode) {
    console.log('EDT vs Fixed Temperature Comparison');
    console.log('Based on Zhang et al. 2024: https://arxiv.org/abs/2403.14541\n');
    console.log(`Formula: T = T₀ · N^(θ/Entropy)  where T₀=${T0}, N=${N}, θ=${THETA}`);
    console.log(`Loading model: ${path.basename(modelPath)}`);
  }

  emit('start', { model: path.basename(modelPath), T0, N, THETA });

  const nCtx = parseInt(process.env.LLAMA_CTX_SIZE || '2048', 10);
  const ctx = await createContext({
    modelPath,
    nCtx,
  });

  // Test 1: Factual question (expect low entropy, EDT should use low temp)
  await compareStrategies(
    ctx,
    'What is 2 + 2? Answer with just the number.',
    'FACTUAL'
  );

  // Test 2: Creative prompt (expect higher entropy, EDT should adapt)
  await compareStrategies(
    ctx,
    'Write one sentence starting a mystery story.',
    'CREATIVE'
  );

  // Test 3: Technical explanation (mixed - some certain, some uncertain)
  await compareStrategies(
    ctx,
    'Explain in one sentence why the sky is blue.',
    'TECHNICAL'
  );

  emit('complete', { comparisons: 3 });

  if (!jsonlMode) {
    console.log(`\n${'='.repeat(70)}`);
    console.log('KEY INSIGHT');
    console.log('='.repeat(70));
    console.log(`
EDT adapts temperature based on model confidence:
  • Low entropy (confident)  → Low temperature  → Trust the model
  • High entropy (uncertain) → High temperature → Explore options

This is the OPPOSITE of naive intuition. When the model knows the answer,
don't add randomness - let it output what it knows.
`);
  }

  ctx.dispose();
}

main().catch((err) => {
  console.error('Error:', (err as Error).message);
  console.error((err as Error).stack);
  process.exit(1);
});
