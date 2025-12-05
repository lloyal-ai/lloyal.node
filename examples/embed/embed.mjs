#!/usr/bin/env node
/**
 * Embedding extraction example using lloyal.node
 *
 * Usage:
 *   node embed.mjs /path/to/embedding-model.gguf
 *   node embed.mjs  # uses default nomic-embed model path
 *
 * This example demonstrates:
 * - Creating an embedding context with pooling enabled
 * - Encoding text and extracting embeddings
 * - Computing cosine similarity between embeddings
 */

import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createContext } from '../../lib/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Default to nomic-embed-text model in fixtures
const DEFAULT_MODEL = path.resolve(
  __dirname,
  '../../liblloyal/tests/fixtures/nomic-embed-text-v1.5.Q4_K_M.gguf'
);

// Pooling types (matches llama.cpp LLAMA_POOLING_TYPE_*)
const PoolingType = {
  NONE: 0,
  MEAN: 1,
  CLS: 2,
  LAST: 3,
};

/**
 * Compute cosine similarity between two vectors
 */
function cosineSimilarity(a, b) {
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
async function getEmbedding(ctx, text) {
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

async function main() {
  const modelPath = process.argv[2] || DEFAULT_MODEL;

  console.log('='.repeat(60));
  console.log('lloyal.node Embedding Example');
  console.log('='.repeat(60));
  console.log(`\nLoading embedding model: ${modelPath}`);
  console.log('This may take a moment...\n');

  // Create context with embedding mode enabled
  const ctx = await createContext({
    modelPath,
    nCtx: 512,
    nThreads: 4,
    embeddings: true,
    poolingType: PoolingType.MEAN,
  });

  console.log('Model loaded!');
  console.log(`  Embedding dimension: ${ctx.getEmbeddingDimension()}`);
  console.log(`  Vocabulary size: ${ctx.vocabSize}`);
  console.log(`  Has pooling: ${ctx.hasPooling()}`);
  console.log();

  // Example texts to embed
  const texts = [
    'The cat sat on the mat.',
    'A cat rested on the rug.',
    'Stock prices rose sharply today.',
    'The feline lounged on the carpet.',
  ];

  console.log('Generating embeddings for sample texts...\n');

  // Get embeddings for all texts
  const embeddings = [];
  for (const text of texts) {
    const start = performance.now();
    const embedding = await getEmbedding(ctx, text);
    const elapsed = (performance.now() - start).toFixed(1);

    embeddings.push({ text, embedding });
    console.log(`  "${text}" (${elapsed}ms)`);
  }

  console.log('\n' + '='.repeat(60));
  console.log('Similarity Matrix');
  console.log('='.repeat(60) + '\n');

  // Print similarity matrix
  console.log('Comparing all pairs:\n');

  for (let i = 0; i < embeddings.length; i++) {
    for (let j = i + 1; j < embeddings.length; j++) {
      const sim = cosineSimilarity(
        embeddings[i].embedding,
        embeddings[j].embedding
      );

      const bar = '█'.repeat(Math.round(sim * 20));
      console.log(`  [${i}] vs [${j}]: ${sim.toFixed(4)} ${bar}`);
      console.log(`      "${texts[i].substring(0, 30)}..."`);
      console.log(`      "${texts[j].substring(0, 30)}..."`);
      console.log();
    }
  }

  // Semantic search demo
  console.log('='.repeat(60));
  console.log('Semantic Search Demo');
  console.log('='.repeat(60) + '\n');

  const query = 'Where did the kitty rest?';
  console.log(`Query: "${query}"\n`);

  const queryEmbedding = await getEmbedding(ctx, query);

  // Rank texts by similarity to query
  const ranked = texts
    .map((text, i) => ({
      text,
      similarity: cosineSimilarity(queryEmbedding, embeddings[i].embedding),
    }))
    .sort((a, b) => b.similarity - a.similarity);

  console.log('Results (ranked by similarity):\n');
  ranked.forEach((result, i) => {
    const bar = '█'.repeat(Math.round(result.similarity * 20));
    console.log(`  ${i + 1}. ${result.similarity.toFixed(4)} ${bar}`);
    console.log(`     "${result.text}"`);
    console.log();
  });

  // Cleanup
  ctx.dispose();
  console.log('Done!');
}

main().catch((err) => {
  console.error('Error:', err.message);
  console.error(err.stack);
  process.exit(1);
});
