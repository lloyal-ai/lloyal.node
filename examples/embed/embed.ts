#!/usr/bin/env node
/**
 * Embedding extraction example using lloyal.node
 *
 * Usage:
 *   npx tsx embed.ts /path/to/embedding-model.gguf          # Human-readable output
 *   npx tsx embed.ts /path/to/embedding-model.gguf --jsonl  # JSONL output for testing
 *   npx tsx embed.ts  # uses default nomic-embed model path
 *
 * This example demonstrates:
 * - Creating an embedding context with pooling enabled
 * - Encoding text and extracting embeddings
 * - Computing cosine similarity between embeddings
 */

import * as path from 'node:path';
import { createContext, PoolingType } from '../../dist/index.js';
import type { SessionContext } from '../../dist/index.js';

// Default to nomic-embed-text model in fixtures
const DEFAULT_MODEL = path.resolve(
  __dirname,
  '../../liblloyal/tests/fixtures/nomic-embed-text-v1.5.Q4_K_M.gguf'
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

/**
 * Compute cosine similarity between two vectors
 */
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same dimension');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (normA * normB);
}

/**
 * Get embedding for a text
 */
async function getEmbedding(ctx: SessionContext, text: string): Promise<Float32Array> {
  // Tokenize the text
  const tokens = await ctx.tokenize(text);

  // Clear KV cache (important: each text needs fresh context)
  await ctx.kvCacheClear();

  // Encode tokens for embedding extraction
  await ctx.encode(tokens);

  // Get L2-normalized embedding
  const embedding = ctx.getEmbeddings(true);

  return embedding;
}

async function main(): Promise<void> {
  if (!jsonlMode) {
    console.log('='.repeat(60));
    console.log('lloyal.node Embedding Example');
    console.log('='.repeat(60));
    console.log(`\nLoading embedding model: ${modelPath}`);
    console.log('This may take a moment...\n');
  }

  // Create context with embedding mode enabled
  const ctx = await createContext({
    modelPath,
    nCtx: 512,
    nThreads: 4,
    embeddings: true,
    poolingType: PoolingType.MEAN,
  });

  emit('start', {
    model: path.basename(modelPath),
    embeddingDim: ctx.getEmbeddingDimension(),
    hasPooling: ctx.hasPooling()
  });

  if (!jsonlMode) {
    console.log('Model loaded!');
    console.log(`  Embedding dimension: ${ctx.getEmbeddingDimension()}`);
    console.log(`  Vocabulary size: ${ctx.vocabSize}`);
    console.log(`  Has pooling: ${ctx.hasPooling()}`);
    console.log();
  }

  // Example texts to embed
  const texts = [
    'The cat sat on the mat.',
    'A cat rested on the rug.',
    'Stock prices rose sharply today.',
    'The feline lounged on the carpet.',
  ];

  if (!jsonlMode) {
    console.log('Generating embeddings for sample texts...\n');
  }

  // Get embeddings for all texts
  const embeddings: { text: string; embedding: Float32Array }[] = [];
  for (const text of texts) {
    const start = performance.now();
    const embedding = await getEmbedding(ctx, text);
    const elapsed = (performance.now() - start).toFixed(1);

    embeddings.push({ text, embedding });

    emit('embedding', { text, dimension: embedding.length, elapsed: parseFloat(elapsed) });

    if (!jsonlMode) {
      console.log(`  "${text}" (${elapsed}ms)`);
    }
  }

  if (!jsonlMode) {
    console.log('\n' + '='.repeat(60));
    console.log('Similarity Matrix');
    console.log('='.repeat(60) + '\n');
    console.log('Comparing all pairs:\n');
  }

  // Compute and emit similarity matrix
  for (let i = 0; i < embeddings.length; i++) {
    for (let j = i + 1; j < embeddings.length; j++) {
      const sim = cosineSimilarity(
        embeddings[i].embedding,
        embeddings[j].embedding
      );

      emit('similarity', { i, j, similarity: sim });

      if (!jsonlMode) {
        const bar = '\u2588'.repeat(Math.round(sim * 20));
        console.log(`  [${i}] vs [${j}]: ${sim.toFixed(4)} ${bar}`);
        console.log(`      "${texts[i].substring(0, 30)}..."`);
        console.log(`      "${texts[j].substring(0, 30)}..."`);
        console.log();
      }
    }
  }

  // Semantic search demo
  if (!jsonlMode) {
    console.log('='.repeat(60));
    console.log('Semantic Search Demo');
    console.log('='.repeat(60) + '\n');
  }

  const query = 'Where did the kitty rest?';

  if (!jsonlMode) {
    console.log(`Query: "${query}"\n`);
  }

  const queryEmbedding = await getEmbedding(ctx, query);

  // Rank texts by similarity to query
  const ranked = texts
    .map((text, i) => ({
      text,
      similarity: cosineSimilarity(queryEmbedding, embeddings[i].embedding),
    }))
    .sort((a, b) => b.similarity - a.similarity);

  emit('search', { query, results: ranked.map(r => ({ text: r.text, similarity: r.similarity })) });

  if (!jsonlMode) {
    console.log('Results (ranked by similarity):\n');
    ranked.forEach((result, i) => {
      const bar = '\u2588'.repeat(Math.round(result.similarity * 20));
      console.log(`  ${i + 1}. ${result.similarity.toFixed(4)} ${bar}`);
      console.log(`     "${result.text}"`);
      console.log();
    });
  }

  emit('complete', { embeddings: texts.length, queriesRun: 1 });

  // Cleanup
  ctx.dispose();

  if (!jsonlMode) {
    console.log('Done!');
  }
}

main().catch((err) => {
  console.error('Error:', (err as Error).message);
  console.error((err as Error).stack);
  process.exit(1);
});
