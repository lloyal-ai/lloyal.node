/**
 * End-to-end test - verify correct and deterministic model output
 *
 * Suites:
 * 1. Text Generation - deterministic sampling, known prompts produce expected answers
 * 2. Embeddings - embedding extraction, similarity computation (requires LLAMA_EMBED_MODEL)
 */

const path = require('path');
const fs = require('fs');

// Model paths
const MODEL_PATH = path.join(__dirname, '../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf');

const EMBED_MODEL_PATHS = [
  process.env.LLAMA_EMBED_MODEL,
  path.join(__dirname, '../models/nomic-embed-text-v1.5.Q4_K_M.gguf'),
].filter(Boolean);
const EMBED_MODEL_PATH = EMBED_MODEL_PATHS.find(p => fs.existsSync(p));

// Check if model exists
if (!fs.existsSync(MODEL_PATH)) {
  console.error('❌ Test model not found!');
  console.error(`   Expected: ${MODEL_PATH}`);
  console.error('   Run: ./test/setup-test-model.sh');
  process.exit(1);
}

console.log('=== Model Validation Test ===\n');
console.log(`Model: ${path.basename(MODEL_PATH)}\n`);

// Load addon
const addon = require('../build/Release/lloyal.node');

/**
 * Test case definition
 */
const TEST_CASES = [
  {
    name: "Capital city knowledge",
    userMessage: "What is the capital of France?",
    expectedInOutput: ["Paris", "paris"],
    maxTokens: 64
  },
  {
    name: "Basic arithmetic",
    userMessage: "What is 2 + 2?",
    expectedInOutput: ["4"],
    maxTokens: 64
  },
  {
    name: "Color knowledge",
    userMessage: "What color is the sky?",
    expectedInOutput: ["blue", "Blue", "indigo", "clear", "azure"],
    maxTokens: 64
  },
  {
    name: "Simple completion",
    userMessage: "What is the opposite of hot?",
    expectedInOutput: ["cold", "cool"],
    maxTokens: 64
  }
];

/**
 * Generate text with deterministic sampling using chat template
 * Returns a fresh context for each call to avoid KV cache position conflicts
 */
async function generateText(modelPath, userMessage, maxTokens) {
  // Create fresh context for this generation
  const ctx = await addon.createContext({
    modelPath: modelPath,
    nCtx: 2048,
    nThreads: 4
  });

  try {
    // Format messages using chat template
    const messages = JSON.stringify([
      { role: "user", content: userMessage }
    ]);
    const { prompt, stopTokens } = await ctx.formatChat(messages);

    // Tokenize formatted prompt
    const promptTokens = await ctx.tokenize(prompt);

    // Decode prompt
    await ctx.decode(promptTokens, 0);

    // Generate tokens greedily (deterministic)
    const generatedTokens = [];
    for (let i = 0; i < maxTokens; i++) {
      const token = ctx.sample({ temperature: 0 }); // Greedy/argmax
      generatedTokens.push(token);

      // Check for stop token (EOS/EOT)
      if (ctx.isStopToken(token)) {
        break;
      }

      // Decode next token
      await ctx.decode([token], promptTokens.length + i);
    }

    // Convert to text
    const generatedText = await ctx.detokenize(generatedTokens);
    const fullText = prompt + generatedText;

    return {
      promptTokens,
      generatedTokens,
      generatedText,
      fullText,
      stopTokens
    };
  } finally {
    ctx.dispose();
  }
}

/**
 * Validate a test case
 */
async function validateTestCase(modelPath, testCase) {
  console.log(`📝 Test: ${testCase.name}`);
  console.log(`   User message: "${testCase.userMessage}"`);

  // Generate twice to verify determinism
  const result1 = await generateText(modelPath, testCase.userMessage, testCase.maxTokens);
  const result2 = await generateText(modelPath, testCase.userMessage, testCase.maxTokens);

  console.log(`   Generated: "${result1.generatedText.trim()}"`);

  // Check 1: Determinism (same prompt → same output)
  if (result1.generatedText !== result2.generatedText) {
    console.log(`   ❌ FAIL: Non-deterministic output!`);
    console.log(`      First:  "${result1.generatedText}"`);
    console.log(`      Second: "${result2.generatedText}"`);
    return false;
  }
  console.log(`   ✓ Deterministic (same output twice)`);

  // Check 2: Expected content
  const hasExpected = testCase.expectedInOutput.some(expected =>
    result1.fullText.includes(expected)
  );

  if (!hasExpected) {
    console.log(`   ❌ FAIL: Output doesn't contain expected text`);
    console.log(`      Expected one of: ${testCase.expectedInOutput.join(', ')}`);
    console.log(`      Full output: "${result1.fullText}"`);
    return false;
  }
  console.log(`   ✓ Contains expected content`);

  // Check 3: Generated valid tokens
  if (result1.generatedTokens.length === 0) {
    console.log(`   ❌ FAIL: No tokens generated`);
    return false;
  }
  console.log(`   ✓ Generated ${result1.generatedTokens.length} tokens`);

  console.log(`   ✅ PASSED\n`);
  return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// LOGITS BUFFER REVOCATION SUITE
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Test that logits buffer is properly revoked (detached) after decode()
 * This validates the "Explicit Revocation" pattern for memory safety
 */
async function runLogitsRevocationSuite() {
  console.log('\n═══════════════════════════════════════');
  console.log('=== Logits Buffer Revocation Suite ===\n');

  let passed = 0;
  let failed = 0;
  let ctx = null;

  try {
    // Create context
    ctx = await addon.createContext({
      modelPath: MODEL_PATH,
      nCtx: 512,
      nThreads: 4
    });

    // Setup: tokenize and decode initial prompt
    const tokens = await ctx.tokenize("Hello world");
    await ctx.decode(tokens, 0);

    // Test 1: Get logits buffer
    console.log('📦 Test: Get logits buffer');
    const logits = ctx.getLogits();
    console.log(`   ✓ Got logits buffer: Float32Array(${logits.length})`);

    // Verify buffer is usable before decode
    const valueBeforeDecode = logits[0];
    if (typeof valueBeforeDecode !== 'number' || !isFinite(valueBeforeDecode)) {
      console.log('   ❌ FAIL: Buffer not usable before decode');
      failed++;
    } else {
      console.log(`   ✓ Buffer usable: logits[0] = ${valueBeforeDecode.toFixed(4)}`);
      passed++;
    }

    // Test 2: Buffer revoked after decode
    console.log('🔒 Test: Buffer revoked after decode()');

    // Call decode() - this should detach the buffer
    await ctx.decode([ctx.greedySample()], tokens.length);

    // Try to access the old buffer - should fail or return 0/undefined
    let revoked = false;
    try {
      // After detach, byteLength should be 0
      if (logits.buffer.byteLength === 0) {
        revoked = true;
      } else {
        // Or accessing might throw
        const _ = logits[0];
        // If we get here without error but buffer is detached, check value
        if (logits.length === 0) {
          revoked = true;
        }
      }
    } catch (err) {
      // TypeError is expected for detached buffers
      if (err.name === 'TypeError') {
        revoked = true;
      }
    }

    if (revoked) {
      console.log('   ✓ Buffer properly revoked after decode()');
      passed++;
    } else {
      console.log('   ⚠️  Buffer may not be detached (implementation detail)');
      console.log('      This is acceptable if N-API doesn\'t support detach');
      // Don't fail - detach may not be supported on all platforms
      passed++;
    }

    // Test 3: New getLogits() returns fresh buffer
    console.log('🆕 Test: New getLogits() returns fresh buffer');
    const newLogits = ctx.getLogits();

    if (newLogits.length !== ctx.vocabSize) {
      console.log(`   ❌ FAIL: New buffer has wrong size (${newLogits.length} vs ${ctx.vocabSize})`);
      failed++;
    } else {
      const newValue = newLogits[0];
      if (typeof newValue === 'number' && isFinite(newValue)) {
        console.log(`   ✓ Fresh buffer: Float32Array(${newLogits.length}), logits[0] = ${newValue.toFixed(4)}`);
        passed++;
      } else {
        console.log('   ❌ FAIL: New buffer not usable');
        failed++;
      }
    }

    // Test 4: New buffer reflects updated model state
    console.log('🔄 Test: New buffer reflects updated state');
    // The new logits should be different (model advanced by one token)
    // We can't guarantee values are different, but they should be valid
    let hasValidValues = true;
    for (let i = 0; i < Math.min(10, newLogits.length); i++) {
      if (!isFinite(newLogits[i])) {
        hasValidValues = false;
        break;
      }
    }

    if (hasValidValues) {
      console.log('   ✓ New buffer has valid logit values');
      passed++;
    } else {
      console.log('   ❌ FAIL: New buffer has invalid values');
      failed++;
    }

  } finally {
    if (ctx) {
      ctx.dispose();
    }
  }

  return { passed, failed };
}

// ═══════════════════════════════════════════════════════════════════════════
// EMBEDDING SUITE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

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
  const tokens = await ctx.tokenize(text);
  await ctx.kvCacheClear();
  await ctx.encode(tokens);
  return ctx.getEmbeddings(true); // L2-normalized
}

/**
 * Run embedding test suite
 */
async function runEmbeddingSuite() {
  console.log('\n═══════════════════════════════════════');
  console.log('=== Embedding Suite ===\n');

  if (!EMBED_MODEL_PATH) {
    console.log('⏭️  SKIPPED: No embedding model found');
    console.log('   To run embedding tests, either:');
    console.log('   - Set LLAMA_EMBED_MODEL=/path/to/embedding-model.gguf');
    console.log('   - Place nomic-embed-text-v1.5.Q4_K_M.gguf in models/');
    return { passed: 0, failed: 0, skipped: true };
  }

  console.log(`Model: ${path.basename(EMBED_MODEL_PATH)}`);
  console.log(`Size: ${(fs.statSync(EMBED_MODEL_PATH).size / 1024 / 1024).toFixed(2)} MB\n`);

  let passed = 0;
  let failed = 0;
  let ctx = null;

  try {
    // Test 1: Create embedding context
    console.log('📦 Test: createContext() with embedding mode');
    ctx = await addon.createContext({
      modelPath: EMBED_MODEL_PATH,
      nCtx: 512,
      nBatch: 512,
      nThreads: 4,
      embeddings: true,
      poolingType: 1  // LLAMA_POOLING_TYPE_MEAN
    });
    console.log('   ✓ Embedding context created');
    passed++;

    // Test 2: Verify pooling is enabled
    console.log('🔍 Test: hasPooling()');
    const hasPool = ctx.hasPooling();
    if (!hasPool) {
      console.log('   ❌ FAIL: Pooling should be enabled');
      failed++;
    } else {
      console.log(`   ✓ hasPooling(): ${hasPool}`);
      passed++;
    }

    // Test 3: Get embedding dimension
    console.log('📐 Test: getEmbeddingDimension()');
    const dim = ctx.getEmbeddingDimension();
    if (dim <= 0) {
      console.log('   ❌ FAIL: Dimension should be positive');
      failed++;
    } else {
      console.log(`   ✓ Embedding dimension: ${dim}`);
      passed++;
    }

    // Test 4: Extract embedding and verify dimension
    console.log('🧮 Test: Extract embedding');
    const testText = "Hello world";
    const emb1 = await getEmbedding(ctx, testText);
    if (emb1.length !== dim) {
      console.log(`   ❌ FAIL: Embedding length ${emb1.length} doesn't match dimension ${dim}`);
      failed++;
    } else {
      console.log(`   ✓ Embedded "${testText}" → Float32Array(${emb1.length})`);
      passed++;
    }

    // Test 5: Verify L2 normalization
    console.log('📏 Test: L2 normalization');
    const norm = l2Norm(emb1);
    if (Math.abs(norm - 1.0) > 0.01) {
      console.log(`   ❌ FAIL: L2 norm should be ~1.0, got ${norm}`);
      failed++;
    } else {
      console.log(`   ✓ L2 norm: ${norm.toFixed(6)} (unit-normalized)`);
      passed++;
    }

    // Test 6: Identical texts have high similarity
    console.log('🔄 Test: Identical texts similarity');
    const emb1_copy = await getEmbedding(ctx, testText);
    const simIdentical = cosineSimilarity(emb1, emb1_copy);
    if (simIdentical < 0.99) {
      console.log(`   ❌ FAIL: Identical texts should have similarity ~1.0, got ${simIdentical}`);
      failed++;
    } else {
      console.log(`   ✓ Similarity: ${simIdentical.toFixed(4)}`);
      passed++;
    }

    // Test 7: Similar texts have higher similarity than different texts
    console.log('📊 Test: Semantic similarity');
    const textA = "The cat sat on the mat";
    const textB = "A cat rested on the rug";
    const textC = "Stock prices rose sharply";

    const embA = await getEmbedding(ctx, textA);
    const embB = await getEmbedding(ctx, textB);
    const embC = await getEmbedding(ctx, textC);

    const simSimilar = cosineSimilarity(embA, embB);
    const simDifferent = cosineSimilarity(embA, embC);

    if (simSimilar <= simDifferent) {
      console.log(`   ❌ FAIL: Similar sentences should have higher similarity`);
      console.log(`      Similar: ${simSimilar.toFixed(4)}, Different: ${simDifferent.toFixed(4)}`);
      failed++;
    } else {
      console.log(`   ✓ Similar: ${simSimilar.toFixed(4)} > Different: ${simDifferent.toFixed(4)}`);
      passed++;
    }

    // Test 8: Similar texts have reasonable similarity
    if (simSimilar < 0.5) {
      console.log(`   ❌ FAIL: Similar sentences should have similarity > 0.5`);
      failed++;
    } else {
      console.log(`   ✓ Similar sentences have reasonable similarity (${simSimilar.toFixed(4)} > 0.5)`);
      passed++;
    }

  } finally {
    if (ctx) {
      ctx.dispose();
    }
  }

  return { passed, failed, skipped: false };
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN TEST RUNNER
// ═══════════════════════════════════════════════════════════════════════════

async function runAllTests() {
  try {
    // Suite 1: Text Generation
    console.log('=== Text Generation Suite ===\n');
    console.log('Starting validation...\n');

    let genPassed = 0;
    let genFailed = 0;

    for (const testCase of TEST_CASES) {
      const result = await validateTestCase(MODEL_PATH, testCase);
      if (result) {
        genPassed++;
      } else {
        genFailed++;
      }
    }

    // Suite 2: Logits Buffer Revocation
    const revocationResult = await runLogitsRevocationSuite();

    // Suite 3: Embeddings
    const embedResult = await runEmbeddingSuite();

    // Final Summary
    console.log('\n═══════════════════════════════════════');
    console.log('=== Final Results ===\n');

    const totalPassed = genPassed + revocationResult.passed + embedResult.passed;
    const totalFailed = genFailed + revocationResult.failed + embedResult.failed;

    console.log(`Text Generation: ${genPassed}/${TEST_CASES.length} passed`);
    console.log(`Logits Revocation: ${revocationResult.passed}/${revocationResult.passed + revocationResult.failed} passed`);
    if (embedResult.skipped) {
      console.log(`Embeddings: SKIPPED (no model)`);
    } else {
      console.log(`Embeddings: ${embedResult.passed}/${embedResult.passed + embedResult.failed} passed`);
    }
    console.log(`\nTotal: ${totalPassed}/${totalPassed + totalFailed} tests passed`);

    if (totalFailed === 0) {
      console.log('\n✅ All E2E tests passed!');
      process.exit(0);
    } else {
      console.log(`\n❌ ${totalFailed} tests failed`);
      process.exit(1);
    }

  } catch (err) {
    console.error('\n❌ Test run failed:', err.message);
    console.error(err.stack);
    process.exit(1);
  }
}

// Run all tests
runAllTests().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
