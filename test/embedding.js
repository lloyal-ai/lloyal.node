/**
 * Embedding test - verify embedding extraction works correctly
 *
 * Tests embedding functionality:
 * 1. Context creation with embedding mode
 * 2. Embedding extraction and dimension validation
 * 3. L2 normalization verification
 * 4. Cosine similarity between similar/different texts
 *
 * Requires LLAMA_EMBED_MODEL environment variable pointing to an embedding model
 * (e.g., nomic-embed-text-v1.5.Q4_K_M.gguf)
 *
 * Skips gracefully if no embedding model is available (for CI compatibility)
 */

const path = require('path');
const fs = require('fs');

// Embedding model path - check multiple locations
const POSSIBLE_PATHS = [
  process.env.LLAMA_EMBED_MODEL,
  path.join(__dirname, '../models/nomic-embed-text-v1.5.Q4_K_M.gguf'),
  path.join(__dirname, '../liblloyal/tests/fixtures/nomic-embed-text-v1.5.Q4_K_M.gguf'),
].filter(Boolean);

const EMBED_MODEL_PATH = POSSIBLE_PATHS.find(p => fs.existsSync(p));

// Skip gracefully if no embedding model found (CI may not have one)
if (!EMBED_MODEL_PATH) {
  console.log('=== Embedding Test ===\n');
  console.log('⏭️  SKIPPED: No embedding model found');
  console.log('   Checked paths:');
  POSSIBLE_PATHS.forEach(p => console.log(`     - ${p}`));
  console.log('\n   To run embedding tests:');
  console.log('   1. Set LLAMA_EMBED_MODEL=/path/to/embedding-model.gguf');
  console.log('   2. Or place nomic-embed-text-v1.5.Q4_K_M.gguf in models/');
  console.log('\n✅ Test skipped (embedding model not available)');
  process.exit(0);  // Exit successfully - this is expected in CI without embedding model
}

console.log('=== Embedding Test ===\n');
console.log(`Model: ${path.basename(EMBED_MODEL_PATH)}`);
console.log(`Size: ${(fs.statSync(EMBED_MODEL_PATH).size / 1024 / 1024).toFixed(2)} MB\n`);

// Load addon
const addon = require('../build/Release/lloyal.node');

/**
 * Compute cosine similarity between two vectors
 */
function cosineSimilarity(a, b) {
  if (a.length !== b.length) {
    throw new Error(`Vector length mismatch: ${a.length} vs ${b.length}`);
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
 * Compute L2 norm of a vector
 */
function l2Norm(vec) {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) {
    sum += vec[i] * vec[i];
  }
  return Math.sqrt(sum);
}

/**
 * Extract embedding for a text
 */
async function getEmbedding(ctx, text) {
  // Tokenize
  const tokens = await ctx.tokenize(text);

  // Clear KV cache between texts
  await ctx.kvCacheClear();

  // Encode tokens for embedding extraction
  await ctx.encode(tokens);

  // Get L2-normalized embeddings
  const embeddings = ctx.getEmbeddings(true);

  return embeddings;
}

async function runTests() {
  let ctx = null;
  let testsPassed = 0;
  let testsFailed = 0;

  try {
    // ===== TEST 1: Create embedding context =====
    console.log('📦 Test 1: createContext() with embedding mode');
    ctx = await addon.createContext({
      modelPath: EMBED_MODEL_PATH,
      nCtx: 512,
      nBatch: 512,
      nThreads: 4,
      embeddings: true,
      poolingType: 1  // LLAMA_POOLING_TYPE_MEAN
    });
    console.log('✓ Embedding context created successfully');
    testsPassed++;

    // ===== TEST 2: Verify pooling is enabled =====
    console.log('\n🔍 Test 2: hasPooling()');
    const hasPool = ctx.hasPooling();
    console.log(`✓ hasPooling(): ${hasPool}`);
    if (!hasPool) {
      console.error('❌ FAIL: Pooling should be enabled');
      testsFailed++;
    } else {
      testsPassed++;
    }

    // ===== TEST 3: Get embedding dimension =====
    console.log('\n📐 Test 3: getEmbeddingDimension()');
    const dim = ctx.getEmbeddingDimension();
    console.log(`✓ Embedding dimension: ${dim}`);
    if (dim <= 0) {
      console.error('❌ FAIL: Dimension should be positive');
      testsFailed++;
    } else {
      testsPassed++;
    }

    // ===== TEST 4: Extract embedding =====
    console.log('\n🧮 Test 4: Extract embedding');
    const testText = "Hello world";
    const emb1 = await getEmbedding(ctx, testText);
    console.log(`✓ Embedded "${testText}" → Float32Array(${emb1.length})`);

    if (emb1.length !== dim) {
      console.error(`❌ FAIL: Embedding length ${emb1.length} doesn't match dimension ${dim}`);
      testsFailed++;
    } else {
      console.log(`✓ Embedding length matches dimension`);
      testsPassed++;
    }

    // ===== TEST 5: Verify L2 normalization =====
    console.log('\n📏 Test 5: L2 normalization verification');
    const norm = l2Norm(emb1);
    console.log(`✓ L2 norm: ${norm.toFixed(6)}`);
    if (Math.abs(norm - 1.0) > 0.01) {
      console.error(`❌ FAIL: L2 norm should be ~1.0, got ${norm}`);
      testsFailed++;
    } else {
      console.log('✓ Embedding is unit-normalized');
      testsPassed++;
    }

    // ===== TEST 6: Cosine similarity - identical texts =====
    console.log('\n🔄 Test 6: Cosine similarity - identical texts');
    const emb1_copy = await getEmbedding(ctx, testText);
    const simIdentical = cosineSimilarity(emb1, emb1_copy);
    console.log(`✓ Similarity("${testText}", "${testText}"): ${simIdentical.toFixed(4)}`);
    if (simIdentical < 0.99) {
      console.error(`❌ FAIL: Identical texts should have similarity ~1.0, got ${simIdentical}`);
      testsFailed++;
    } else {
      console.log('✓ Identical texts have high similarity');
      testsPassed++;
    }

    // ===== TEST 7: Cosine similarity - similar texts =====
    console.log('\n📊 Test 7: Cosine similarity - similar vs different texts');

    const textA = "The cat sat on the mat";
    const textB = "A cat rested on the rug";
    const textC = "Stock prices rose sharply";

    const embA = await getEmbedding(ctx, textA);
    const embB = await getEmbedding(ctx, textB);
    const embC = await getEmbedding(ctx, textC);

    const simSimilar = cosineSimilarity(embA, embB);
    const simDifferent = cosineSimilarity(embA, embC);

    console.log(`✓ Similar sentences: "${textA}" vs "${textB}"`);
    console.log(`  Similarity: ${simSimilar.toFixed(4)}`);
    console.log(`✓ Different sentences: "${textA}" vs "${textC}"`);
    console.log(`  Similarity: ${simDifferent.toFixed(4)}`);

    if (simSimilar <= simDifferent) {
      console.error('❌ FAIL: Similar sentences should have higher similarity than different ones');
      testsFailed++;
    } else {
      console.log('✓ Similar sentences have higher similarity than different ones');
      testsPassed++;
    }

    if (simSimilar < 0.5) {
      console.error(`❌ FAIL: Similar sentences should have similarity > 0.5, got ${simSimilar}`);
      testsFailed++;
    } else {
      console.log('✓ Similar sentences have reasonable similarity (> 0.5)');
      testsPassed++;
    }

    // ===== SUMMARY =====
    console.log('\n═══════════════════════════════════════');
    console.log(`Results: ${testsPassed}/${testsPassed + testsFailed} tests passed`);

    if (testsFailed === 0) {
      console.log('✅ All embedding tests passed!');
      process.exit(0);
    } else {
      console.log(`❌ ${testsFailed} tests failed`);
      process.exit(1);
    }

  } catch (err) {
    console.error('\n❌ Test failed:', err.message);
    console.error(err.stack);
    process.exit(1);
  } finally {
    // ===== CLEANUP =====
    if (ctx) {
      console.log('\n🧹 Cleanup: dispose()');
      ctx.dispose();
      console.log('✓ Context disposed');
    }
  }
}

// Run tests
runTests().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
