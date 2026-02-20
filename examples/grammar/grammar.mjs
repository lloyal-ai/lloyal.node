#!/usr/bin/env node
/**
 * Grammar-constrained generation with forkable state
 *
 * Uses Branch API for grammar-constrained generation with tree branching.
 * Grammar state is automatically cloned on fork(), so each branch can
 * diverge independently while maintaining valid JSON output.
 *
 * Usage:
 *   node grammar.mjs [model-path]          # Human-readable output
 *   node grammar.mjs [model-path] --jsonl  # JSONL output for testing
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

/** Emit output - JSONL or human-readable */
function emit(event, data) {
  if (jsonlMode) {
    console.log(JSON.stringify({ event, ...data }));
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

  const grammar = await ctx.jsonSchemaToGrammar(JSON.stringify(schema));
  if (!jsonlMode) {
    console.log('\nGBNF Grammar (first 200 chars):');
    console.log(grammar.slice(0, 200) + '...\n');
  }

  const prompt = 'Generate a person as JSON:\n';
  if (!jsonlMode) {
    console.log(`Prompt: "${prompt}"`);
  }

  const tokens = await ctx.tokenize(prompt);

  // Root branch with grammar constraint — grammar state cloned automatically on fork()
  const root = Branch.create(ctx, 0, { temperature: 0.7, topP: 0.9 }, undefined, grammar);
  await root.prefill(tokens);

  // ===== PHASE 1: Generate until we see "city" key =====
  if (!jsonlMode) {
    console.log('\nGenerating until "city" field...');
    process.stdout.write('  ');
  }

  let accumulated = '';

  for (let i = 0; i < 100; i++) {
    const { token, text, isStop } = await root.produce();
    if (isStop) break;

    accumulated += text;
    if (!jsonlMode) {
      process.stdout.write(text);
    }
    emit('token', { phase: 'prefix', token, text });

    await root.commit(token);

    // Stop when we see "city": - we want to branch here
    if (accumulated.includes('"city"')) {
      break;
    }
  }
  if (!jsonlMode) {
    console.log('\n');
  }

  // ===== PHASE 2: Fork and complete with different branches =====
  const cities = ['NYC', 'LA', 'Chicago'];
  if (!jsonlMode) {
    console.log(`Forking into ${cities.length} branches at branch point...\n`);
  }

  emit('branch_point', { prefix: accumulated, position: root.position });

  const results = [];
  for (const city of cities) {
    const child = await root.fork();
    child.reseedSampler(results.length + 42);

    let branchText = '';
    for (let i = 0; i < 30; i++) {
      const { token, text, isStop } = await child.produce();
      if (isStop) break;

      branchText += text;
      emit('token', { phase: 'branch', city, token, text });

      await child.commit(token);
    }

    const fullOutput = accumulated + branchText;
    results.push({ city, output: fullOutput });

    if (!jsonlMode) {
      console.log(`  [${city} branch]: ${fullOutput}`);
    }
    emit('branch_complete', { city, output: fullOutput });

    await child.prune();
  }

  await root.prune();

  // Validate JSON outputs
  let validJsonCount = 0;
  for (const b of results) {
    try {
      JSON.parse(b.output);
      validJsonCount++;
    } catch {
      // Invalid JSON
    }
  }

  emit('complete', {
    branchCount: results.length,
    validJsonCount,
    branches: results.map(b => ({ city: b.city, output: b.output })),
  });

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
