/**
 * End-to-end test - verify correct and deterministic model output
 *
 * Uses deterministic sampling (temp=0) to validate:
 * 1. Known prompts produce expected answers
 * 2. Same prompt always produces same output (determinism)
 * 3. Complete inference pipeline works correctly
 */

const path = require('path');
const fs = require('fs');

const MODEL_PATH = path.join(__dirname, '../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf');

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

async function runValidation() {
  try {
    console.log('Starting validation...\n');

    // Run all test cases
    let passed = 0;
    let failed = 0;

    for (const testCase of TEST_CASES) {
      const result = await validateTestCase(MODEL_PATH, testCase);
      if (result) {
        passed++;
      } else {
        failed++;
      }
    }

    // Summary
    console.log('═══════════════════════════════════════');
    console.log(`Results: ${passed}/${TEST_CASES.length} tests passed`);

    if (failed === 0) {
      console.log('✅ All validation tests passed!');
      console.log('   Model is generating correct, deterministic output.');
      process.exit(0);
    } else {
      console.log(`❌ ${failed} tests failed`);
      console.log('   Model may not be working correctly.');
      process.exit(1);
    }

  } catch (err) {
    console.error('\n❌ Validation failed:', err.message);
    console.error(err.stack);
    process.exit(1);
  }
}

// Run validation
runValidation().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
