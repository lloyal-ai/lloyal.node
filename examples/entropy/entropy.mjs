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
 */

import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createContext } from '../../lib/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_MODEL = path.resolve(
  __dirname,
  '../../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf'
);

// EDT parameters (Zhang et al. 2024)
const T0 = 1.0;    // Max temperature bound
const N = 0.8;     // Base (paper uses 0.8)
const THETA = 1.5; // Scale factor

/**
 * Calculate EDT temperature from entropy
 */
function edtTemperature(entropy) {
  const safeEntropy = Math.max(entropy, 0.1);
  return T0 * Math.pow(N, THETA / safeEntropy);
}

/**
 * Generate with a specific sampling strategy
 */
async function generate(ctx, prompt, strategy, maxTokens = 50) {
  const messages = [{ role: 'user', content: prompt }];
  const { prompt: formatted } = await ctx.formatChat(JSON.stringify(messages));

  const tokens = await ctx.tokenize(formatted);
  await ctx.decode(tokens, 0, 0);

  const output = [];
  const temps = [];
  const entropies = [];
  let pos = tokens.length;

  for (let i = 0; i < maxTokens; i++) {
    const entropy = ctx.modelEntropy('nats');
    entropies.push(entropy);

    let temp;
    if (strategy === 'edt') {
      temp = edtTemperature(entropy);
    } else {
      temp = strategy; // Fixed temperature
    }
    temps.push(temp);

    const token = ctx.sample({ temperature: temp });
    if (ctx.isStopToken(token)) break;

    output.push(token);
    await ctx.decode([token], pos++, 0);
  }

  // Clear KV cache for next run
  await ctx.kvCacheClear();

  const text = await ctx.detokenize(output);
  const avgEntropy = entropies.reduce((a, b) => a + b, 0) / entropies.length;
  const avgTemp = temps.reduce((a, b) => a + b, 0) / temps.length;

  return { text, avgEntropy, avgTemp, tokenCount: output.length, temps, entropies };
}

/**
 * Run comparison for a single prompt
 */
async function compareStrategies(ctx, prompt, label) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`${label}: "${prompt}"`);
  console.log('='.repeat(70));

  // Run with fixed temperature
  const fixed = await generate(ctx, prompt, 0.7);

  // Run with EDT
  const edt = await generate(ctx, prompt, 'edt');

  // Display results
  console.log(`\n  Fixed (T=0.7):`);
  console.log(`    Output: ${fixed.text.slice(0, 100)}${fixed.text.length > 100 ? '...' : ''}`);
  console.log(`    Tokens: ${fixed.tokenCount} | Avg entropy: ${fixed.avgEntropy.toFixed(2)} nats`);

  console.log(`\n  EDT (adaptive):`);
  console.log(`    Output: ${edt.text.slice(0, 100)}${edt.text.length > 100 ? '...' : ''}`);
  console.log(`    Tokens: ${edt.tokenCount} | Avg entropy: ${edt.avgEntropy.toFixed(2)} nats | Avg T: ${edt.avgTemp.toFixed(2)}`);

  // Show temperature adaptation
  const lowEntropyTokens = edt.entropies.filter(e => e < 1.0).length;
  const highEntropyTokens = edt.entropies.filter(e => e > 3.5).length;
  console.log(`    Adaptation: ${lowEntropyTokens} confident tokens (T↓), ${highEntropyTokens} uncertain tokens (T↑)`);

  return { fixed, edt };
}

async function main() {
  const modelPath = process.argv[2] || DEFAULT_MODEL;

  console.log('EDT vs Fixed Temperature Comparison');
  console.log('Based on Zhang et al. 2024: https://arxiv.org/abs/2403.14541\n');
  console.log(`Formula: T = T₀ · N^(θ/Entropy)  where T₀=${T0}, N=${N}, θ=${THETA}`);
  console.log(`Loading model: ${path.basename(modelPath)}`);

  const ctx = await createContext({
    modelPath,
    contextSize: 2048,
  });

  // Test 1: Factual question (expect low entropy, EDT should use low temp)
  await compareStrategies(
    ctx,
    'What is 2 + 2? Answer with just the number.',
    'FACTUAL (expect low entropy → low temp)'
  );

  // Test 2: Creative prompt (expect higher entropy, EDT should adapt)
  await compareStrategies(
    ctx,
    'Write one sentence starting a mystery story.',
    'CREATIVE (expect variable entropy → adaptive temp)'
  );

  // Test 3: Technical explanation (mixed - some certain, some uncertain)
  await compareStrategies(
    ctx,
    'Explain in one sentence why the sky is blue.',
    'TECHNICAL (expect mixed entropy)'
  );

  // Summary
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

  ctx.dispose();
}

main().catch((err) => {
  console.error('Error:', err.message);
  console.error(err.stack);
  process.exit(1);
});
