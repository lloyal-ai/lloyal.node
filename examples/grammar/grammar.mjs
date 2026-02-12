#!/usr/bin/env node
/**
 * Grammar-constrained generation with forkable state
 *
 * Uses JS generators for backpressure - generation pauses at each yield,
 * allowing precise control over when to branch.
 *
 * Usage:
 *   node grammar.mjs [model-path]          # Human-readable output
 *   node grammar.mjs [model-path] --jsonl  # JSONL output for testing
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
  if (!jsonlMode) {
    console.log(`Loading model: ${path.basename(modelPath)}`);
  }

  emit('start', { model: path.basename(modelPath) });

  const nCtx = parseInt(process.env.LLAMA_CTX_SIZE || '2048', 10);
  const ctx = await createContext({
    modelPath,
    nCtx,
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

  if (!jsonlMode) {
    console.log('\nJSON Schema:');
    console.log(JSON.stringify(schema, null, 2));
  }

  const grammar = ctx.jsonSchemaToGrammar(JSON.stringify(schema));
  if (!jsonlMode) {
    console.log('\nGBNF Grammar (first 200 chars):');
    console.log(grammar.slice(0, 200) + '...\n');
  }

  const grammarHandle = ctx.createSampler(grammar);

  const prompt = 'Generate a person as JSON:\n';
  if (!jsonlMode) {
    console.log(`Prompt: "${prompt}"`);
  }

  const tokens = await ctx.tokenize(prompt);
  await ctx.decode(tokens, 0, 0);
  let pos = tokens.length;

  // ===== PHASE 1: Generate until we see "city" key =====
  if (!jsonlMode) {
    console.log('\nGenerating until "city" field...');
    process.stdout.write('  ');
  }

  const gen = tokenGenerator(ctx, grammarHandle);
  const collectedTokens = [];
  let accumulated = '';

  for (const { token, text } of gen) {
    collectedTokens.push(token);
    accumulated += text;
    if (!jsonlMode) {
      process.stdout.write(text);
    }
    emit('token', { phase: 'prefix', token, text });

    await ctx.decode([token], pos++, 0);

    // Stop when we see "city": - we want to branch here
    if (accumulated.includes('"city"')) {
      break;
    }
  }
  if (!jsonlMode) {
    console.log('\n');
  }

  // ===== PHASE 2: Save state for branching =====
  if (!jsonlMode) {
    console.log('Saving KV cache and grammar state at branch point...');
  }
  const kvSnapshot = await ctx.kvCacheSave(0);
  const grammarSnapshot = ctx.cloneSampler(grammarHandle);
  const branchPos = pos;

  emit('branch_point', { prefix: accumulated, position: branchPos });

  // ===== PHASE 3: Complete with different cities =====
  const cities = ['NYC', 'LA', 'Chicago'];
  if (!jsonlMode) {
    console.log(`\nExploring ${cities.length} city branches:\n`);
  }

  const branches = [];
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
      emit('token', { phase: 'branch', city, token, text });
      await ctx.decode([token], pos++, 0);
    }

    const fullOutput = accumulated + branchText;
    branches.push({ city, output: fullOutput });

    if (!jsonlMode) {
      console.log(`  [${city} branch]: ${fullOutput}`);
    }
    emit('branch_complete', { city, output: fullOutput });

    ctx.freeSamplerHandle(branchGrammar);
  }

  // Validate JSON outputs
  let validJsonCount = 0;
  for (const b of branches) {
    try {
      JSON.parse(b.output);
      validJsonCount++;
    } catch {
      // Invalid JSON
    }
  }

  emit('complete', {
    branchCount: branches.length,
    validJsonCount,
    branches: branches.map(b => ({ city: b.city, output: b.output })),
  });

  // Cleanup
  ctx.freeSamplerHandle(grammarHandle);
  ctx.freeSamplerHandle(grammarSnapshot);
  ctx.dispose();

  if (!jsonlMode) {
    console.log('\nDone.');
  }
}

main().catch((err) => {
  console.error('Error:', err.message);
  console.error(err.stack);
  process.exit(1);
});
