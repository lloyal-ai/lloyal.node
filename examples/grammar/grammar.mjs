#!/usr/bin/env node
/**
 * Grammar-constrained generation with forkable state
 *
 * Uses JS generators for backpressure - generation pauses at each yield,
 * allowing precise control over when to branch.
 *
 * Usage:
 *   node grammar.mjs /path/to/model.gguf
 *   node grammar.mjs  # uses default model path
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
 * Generator that yields tokens one at a time
 * Caller controls pace via next() - natural backpressure
 */
function* tokenGenerator(ctx, grammarHandle, maxTokens = 100) {
  for (let i = 0; i < maxTokens; i++) {
    // Apply grammar constraints to context logits
    const logits = ctx.getLogits();
    ctx.applySampler(grammarHandle, logits);

    const token = ctx.sample({ temperature: 0.7 });
    if (ctx.isStopToken(token)) return;

    // Advance grammar state
    ctx.acceptSamplerToken(grammarHandle, token);

    // Yield token and text - caller decides when to continue
    yield { token, text: ctx.tokenToText(token) };
  }
}

async function main() {
  const modelPath = process.argv[2] || DEFAULT_MODEL;

  console.log(`Loading model: ${path.basename(modelPath)}`);
  const ctx = await createContext({
    modelPath,
    contextSize: 2048,
    nSeqMax: 4,
  });

  // JSON schema with enum for branching demo
  const schema = {
    type: 'object',
    properties: {
      name: { type: 'string' },
      age: { type: 'number' },
      city: { enum: ['NYC', 'LA', 'Chicago', 'Seattle'] },
    },
    required: ['name', 'age', 'city'],
  };

  console.log('\nJSON Schema:');
  console.log(JSON.stringify(schema, null, 2));

  const grammar = ctx.jsonSchemaToGrammar(JSON.stringify(schema));
  console.log('\nGBNF Grammar (first 200 chars):');
  console.log(grammar.slice(0, 200) + '...\n');

  const grammarHandle = ctx.createSampler(grammar);

  const prompt = 'Generate a person as JSON:\n';
  console.log(`Prompt: "${prompt}"`);

  const tokens = await ctx.tokenize(prompt);
  await ctx.decode(tokens, 0, 0);
  let pos = tokens.length;

  // ===== PHASE 1: Generate until we see "city" key =====
  console.log('\nGenerating until "city" field...');
  process.stdout.write('  ');

  const gen = tokenGenerator(ctx, grammarHandle);
  const collectedTokens = [];
  let accumulated = '';

  for (const { token, text } of gen) {
    collectedTokens.push(token);
    accumulated += text;
    process.stdout.write(text);

    await ctx.decode([token], pos++, 0);

    // Stop when we see "city": - we want to branch here
    if (accumulated.includes('"city"')) {
      break;
    }
  }
  console.log('\n');

  // ===== PHASE 2: Save state for branching =====
  console.log('Saving KV cache and grammar state at branch point...');
  const kvSnapshot = await ctx.kvCacheSave(0);
  const grammarSnapshot = ctx.cloneSampler(grammarHandle);
  const branchPos = pos;

  // ===== PHASE 3: Complete with different cities =====
  const cities = ['NYC', 'LA', 'Chicago'];
  console.log(`\nExploring ${cities.length} city branches:\n`);

  for (const city of cities) {
    // Restore KV cache
    await ctx.kvCacheLoad(0, kvSnapshot);
    pos = branchPos;

    // Fresh grammar clone for this branch
    const branchGrammar = ctx.cloneSampler(grammarSnapshot);

    // Generate completion for this branch
    const branchGen = tokenGenerator(ctx, branchGrammar, 30);
    let branchText = '';

    for (const { token, text } of branchGen) {
      branchText += text;
      await ctx.decode([token], pos++, 0);
    }

    console.log(`  [${city} branch]: ${accumulated}${branchText}`);
    ctx.freeSamplerHandle(branchGrammar);
  }

  // Cleanup
  ctx.freeSamplerHandle(grammarHandle);
  ctx.freeSamplerHandle(grammarSnapshot);
  ctx.dispose();

  console.log('\nDone.');
}

main().catch((err) => {
  console.error('Error:', err.message);
  console.error(err.stack);
  process.exit(1);
});
