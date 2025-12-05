/**
 * API test - verify all SessionContext methods work correctly
 *
 * Tests API functionality and performance benchmarks
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

console.log('=== liblloyal-node Integration Test ===\n');
console.log(`Model: ${MODEL_PATH}`);
console.log(`Size: ${(fs.statSync(MODEL_PATH).size / 1024 / 1024).toFixed(2)} MB\n`);

// Load addon
const addon = require('../build/Release/lloyal.node');

async function runTests() {
  let ctx = null;

  try {
    // ===== TEST 1: Create context =====
    console.log('📦 Test 1: createContext()');
    ctx = await addon.createContext({
      modelPath: MODEL_PATH,
      nCtx: 512,
      nThreads: 4
    });
    console.log('✓ Context created successfully');
    console.log(`✓ Vocab size: ${ctx.vocabSize}\n`);

    // ===== TEST 2: Tokenization =====
    console.log('🔤 Test 2: tokenize() and detokenize()');
    const testText = "Hello world";
    const tokens = await ctx.tokenize(testText);
    console.log(`✓ Tokenized "${testText}" → ${tokens.length} tokens: [${tokens.join(', ')}]`);

    const reconstructed = await ctx.detokenize(tokens);
    console.log(`✓ Detokenized → "${reconstructed}"\n`);

    // ===== TEST 3: Single token conversion =====
    console.log('🔡 Test 3: tokenToText()');
    if (tokens.length > 0) {
      const firstToken = tokens[0];
      const tokenText = ctx.tokenToText(firstToken);
      console.log(`✓ Token ${firstToken} → "${tokenText}"\n`);
    }

    // ===== TEST 4: Decode and get logits =====
    console.log('🧮 Test 4: decode() and getLogits()');
    await ctx.decode(tokens, 0);
    console.log('✓ Decoded tokens successfully');

    const logits = ctx.getLogits();
    console.log(`✓ Got logits: Float32Array(${logits.length})`);

    // Verify logits are valid (not all zeros, not all NaN)
    let hasNonZero = false;
    let hasNaN = false;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] !== 0.0) hasNonZero = true;
      if (isNaN(logits[i])) hasNaN = true;
    }
    console.log(`✓ Logits valid: has variation=${hasNonZero}, has NaN=${hasNaN}\n`);

    if (!hasNonZero) {
      throw new Error('Logits are all zeros!');
    }
    if (hasNaN) {
      throw new Error('Logits contain NaN values!');
    }

    // ===== TEST 5: Native entropy computation =====
    console.log('📊 Test 5: computeEntropy()');
    const entropy = ctx.computeEntropy();
    console.log(`✓ Entropy: ${entropy.toFixed(4)} nats\n`);

    if (!isFinite(entropy) || entropy < 0) {
      throw new Error(`Invalid entropy: ${entropy}`);
    }

    // ===== TEST 6: Greedy sampling =====
    console.log('🎲 Test 6: greedySample()');
    const greedyToken = ctx.greedySample();
    console.log(`✓ Greedy sampled token: ${greedyToken}`);

    if (greedyToken < 0 || greedyToken >= ctx.vocabSize) {
      throw new Error(`Invalid token ID: ${greedyToken}`);
    }

    const greedyText = ctx.tokenToText(greedyToken);
    console.log(`✓ Token text: "${greedyText}"\n`);

    // ===== TEST 7: Parameterized sampling =====
    console.log('🎯 Test 7: sample() with parameters');

    // Test 7a: Greedy (temperature = 0)
    const sampledGreedy = ctx.sample({ temperature: 0 });
    console.log(`✓ Greedy sample (temp=0): token ${sampledGreedy}`);

    // Should match greedySample() result
    if (sampledGreedy !== greedyToken) {
      console.warn(`⚠️  Warning: greedy sample mismatch (${sampledGreedy} vs ${greedyToken})`);
    }

    // Test 7b: Creative sampling
    const sampledCreative = ctx.sample({
      temperature: 0.8,
      topK: 40,
      topP: 0.95,
      minP: 0.05,
      seed: 42,
      penalties: {
        repeat: 1.1,
        frequency: 0.0,
        presence: 0.0,
        lastN: 64
      }
    });
    console.log(`✓ Creative sample: token ${sampledCreative}`);

    if (sampledCreative < 0 || sampledCreative >= ctx.vocabSize) {
      throw new Error(`Invalid sampled token: ${sampledCreative}`);
    }

    console.log(`✓ Token text: "${ctx.tokenToText(sampledCreative)}"\n`);

    // ===== TEST 8: Stop token detection =====
    console.log('🛑 Test 8: isStopToken()');
    const isGreedyStop = ctx.isStopToken(greedyToken);
    console.log(`✓ Token ${greedyToken} is stop token: ${isGreedyStop}\n`);

    // ===== TEST 9: KV cache operations =====
    console.log('💾 Test 9: KV cache operations');

    // Test kvCacheSize (after decode in TEST 4)
    const initialSize = ctx.kvCacheSize();
    console.log(`✓ kvCacheSize(): ${initialSize} (after initial decode)`);

    // Test kvCacheClear
    await ctx.kvCacheClear();
    const afterClear = ctx.kvCacheSize();
    console.log(`✓ kvCacheClear(): size now ${afterClear} (should be -1 for empty)`);

    if (afterClear !== -1) {
      console.warn(`⚠️  Warning: Expected -1 for empty cache, got ${afterClear}`);
    }

    // Restore state by re-decoding
    await ctx.decode(tokens, 0);
    const afterRestore = ctx.kvCacheSize();
    console.log(`✓ Re-decoded tokens, size now: ${afterRestore}\n`);

    // ===== TEST 10: Multi-turn generation with position tracking =====
    console.log('🔄 Test 10: Multi-turn generation (position tracking)');

    // Clear cache first
    await ctx.kvCacheClear();

    // Track position manually (Nitro pattern - don't use kvCacheSize during generation!)
    await ctx.decode(tokens, 0);
    let position = tokens.length;

    let generatedTokens = [...tokens];
    const maxTokens = 10;

    for (let i = 0; i < maxTokens; i++) {
      const nextToken = ctx.sample({ temperature: 0.8, seed: 42 + i });
      generatedTokens.push(nextToken);

      if (ctx.isStopToken(nextToken)) {
        console.log(`✓ Generated ${i + 1} tokens (stopped at EOS)`);
        break;
      }

      // Decode and manually increment position (Nitro pattern)
      await ctx.decode([nextToken], position);
      position += 1;
    }

    const generatedText = await ctx.detokenize(generatedTokens);
    console.log(`✓ Full text: "${generatedText}"`);
    console.log(`✓ Final position: ${position}\n`);

    // ===== TEST 11: Logits buffer memoization =====
    console.log('🧠 Test 11: Logits buffer memoization');

    // Call getLogits() twice in same step - should return same underlying buffer
    const logits1 = ctx.getLogits();
    const logits2 = ctx.getLogits();

    // Check they have same length and same values (memoized)
    if (logits1.length !== logits2.length) {
      throw new Error(`Memoization failed: different lengths (${logits1.length} vs ${logits2.length})`);
    }

    // Check first few values are identical (same buffer)
    let memoMatch = true;
    for (let i = 0; i < Math.min(100, logits1.length); i++) {
      if (logits1[i] !== logits2[i]) {
        memoMatch = false;
        break;
      }
    }

    if (!memoMatch) {
      throw new Error('Memoization failed: buffers have different values');
    }
    console.log('✓ getLogits() returns memoized buffer (same step = same buffer)');

    // Modify one, check other is modified (same underlying memory)
    const originalValue = logits1[0];
    logits1[0] = -999.0;
    if (logits2[0] !== -999.0) {
      console.log('⚠️  Warning: Buffers may not share memory (could be copy)');
    } else {
      console.log('✓ Buffers share same underlying memory (zero-copy confirmed)');
    }
    logits1[0] = originalValue; // Restore
    console.log();

    // ===== TEST 12: withLogits() helper =====
    console.log('🔒 Test 12: withLogits() helper');

    const { withLogits } = require('..');

    // Test synchronous usage (should work)
    const maxLogit = withLogits(ctx, (logits) => {
      let max = logits[0];
      for (let i = 1; i < logits.length; i++) {
        if (logits[i] > max) max = logits[i];
      }
      return max;
    });
    console.log(`✓ withLogits() sync: max logit = ${maxLogit.toFixed(4)}`);

    // Test async rejection (should throw)
    let asyncThrew = false;
    try {
      withLogits(ctx, async (logits) => {
        return logits[0]; // Returning Promise is not allowed
      });
    } catch (err) {
      if (err.message.includes('synchronous')) {
        asyncThrew = true;
      }
    }

    if (!asyncThrew) {
      throw new Error('withLogits() should throw when callback returns Promise');
    }
    console.log('✓ withLogits() rejects async callbacks (safety enforced)\n');

    // ===== TEST 13: Benchmarking native vs TS preparation =====
    console.log('⏱️  Test 13: Performance check');
    const iterations = 100;

    // Warm up
    for (let i = 0; i < 10; i++) {
      ctx.greedySample();
      ctx.computeEntropy();
      ctx.sample({ temperature: 0.8 });
    }

    // Benchmark native greedy
    const startGreedy = Date.now();
    for (let i = 0; i < iterations; i++) {
      ctx.greedySample();
    }
    const greedyTime = Date.now() - startGreedy;
    console.log(`✓ greedySample() x${iterations}: ${greedyTime}ms (${(greedyTime/iterations).toFixed(3)}ms avg)`);

    // Benchmark native entropy
    const startEntropy = Date.now();
    for (let i = 0; i < iterations; i++) {
      ctx.computeEntropy();
    }
    const entropyTime = Date.now() - startEntropy;
    console.log(`✓ computeEntropy() x${iterations}: ${entropyTime}ms (${(entropyTime/iterations).toFixed(3)}ms avg)`);

    // Benchmark native sample
    const startSample = Date.now();
    for (let i = 0; i < iterations; i++) {
      ctx.sample({ temperature: 0.8, topK: 40, topP: 0.95 });
    }
    const sampleTime = Date.now() - startSample;
    console.log(`✓ sample() x${iterations}: ${sampleTime}ms (${(sampleTime/iterations).toFixed(3)}ms avg)`);

    // Benchmark getLogits (for TS sampling baseline)
    const startLogits = Date.now();
    for (let i = 0; i < iterations; i++) {
      const l = ctx.getLogits();
    }
    const logitsTime = Date.now() - startLogits;
    console.log(`✓ getLogits() x${iterations}: ${logitsTime}ms (${(logitsTime/iterations).toFixed(3)}ms avg)`);
    console.log(`  (This is the baseline overhead for TS sampling)\n`);

    // ===== SUCCESS =====
    console.log('✅ All integration tests passed!\n');

  } catch (err) {
    console.error('\n❌ Test failed:', err.message);
    console.error(err.stack);
    process.exit(1);
  } finally {
    // ===== CLEANUP =====
    if (ctx) {
      console.log('🧹 Cleanup: dispose()');
      ctx.dispose();
      console.log('✓ Context disposed\n');
    }
  }
}

// Run tests
runTests().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
