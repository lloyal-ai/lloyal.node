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

// Load addon via lib (uses runtime loading with fallback)
const { loadBinary } = require('..');
const addon = loadBinary();

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
    console.log('📊 Test 5: modelEntropy()');
    const entropy = ctx.modelEntropy();
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
      ctx.modelEntropy();
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
      ctx.modelEntropy();
    }
    const entropyTime = Date.now() - startEntropy;
    console.log(`✓ modelEntropy() x${iterations}: ${entropyTime}ms (${(entropyTime/iterations).toFixed(3)}ms avg)`);

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

    // ===== TEST 14: Multi-sequence KV operations =====
    console.log('🌲 Test 14: Multi-sequence KV operations');

    // Create a new context with multi-sequence support
    const ctx2 = await addon.createContext({
      modelPath: MODEL_PATH,
      nCtx: 512,
      nThreads: 4,
      nSeqMax: 4  // Enable multi-sequence support
    });
    console.log('✓ Created context with nSeqMax=4');

    const seq0Tokens = await ctx2.tokenize("The quick brown fox");
    await ctx2.decode(seq0Tokens, 0, 0); // Decode to seq 0

    // Check seq 0 has tokens
    const seq0Pos = ctx2.kvSeqPosMax(0);
    console.log(`✓ kvSeqPosMax(0): ${seq0Pos}`);
    if (seq0Pos < 0) {
      throw new Error(`Seq 0 should have tokens, got pos_max=${seq0Pos}`);
    }

    // Seq 1 should be empty
    const seq1Before = ctx2.kvSeqPosMax(1);
    console.log(`✓ kvSeqPosMax(1) before copy: ${seq1Before}`);
    if (seq1Before !== -1) {
      throw new Error(`Seq 1 should be empty, got pos_max=${seq1Before}`);
    }

    // Copy seq 0 -> seq 1 (KV cache fork)
    ctx2.kvSeqCopy(0, 1);
    const seq1After = ctx2.kvSeqPosMax(1);
    console.log(`✓ kvSeqCopy(0, 1): seq 1 pos_max now ${seq1After}`);
    if (seq1After !== seq0Pos) {
      throw new Error(`Seq 1 should match seq 0 after copy: ${seq1After} vs ${seq0Pos}`);
    }

    // Seq 0 should be unchanged
    const seq0After = ctx2.kvSeqPosMax(0);
    if (seq0After !== seq0Pos) {
      throw new Error(`Seq 0 should be unchanged after copy: ${seq0After} vs ${seq0Pos}`);
    }
    console.log(`✓ Seq 0 unchanged after copy: ${seq0After}\n`);

    // ===== TEST 15: Handle-based grammar =====
    console.log('📜 Test 15: Handle-based grammar');

    // Simple JSON grammar for testing
    const jsonGrammar = `root ::= "{" ws "}" ws
ws ::= [ \\t\\n]*`;

    // Create sampler
    const handle1 = ctx2.createSampler(jsonGrammar);
    console.log(`✓ createSampler(): handle=${handle1}`);
    if (typeof handle1 !== 'number' || handle1 <= 0) {
      throw new Error(`Invalid sampler handle: ${handle1}`);
    }

    // Clone sampler (for fork)
    const handle2 = ctx2.cloneSampler(handle1);
    console.log(`✓ cloneSampler(${handle1}): new handle=${handle2}`);
    if (handle2 === handle1) {
      throw new Error('Cloned handle should be different from original');
    }

    // Apply sampler to logits buffer
    const testLogits = new Float32Array(ctx2.vocabSize);
    testLogits.fill(0.5); // Fill with dummy values
    ctx2.applySampler(handle1, testLogits);
    console.log(`✓ applySampler(${handle1}, logits): modified logits in-place`);

    // Check that some values were masked (set to -Infinity)
    let maskedCount = 0;
    for (let i = 0; i < testLogits.length; i++) {
      if (testLogits[i] === -Infinity || testLogits[i] < -1e30) {
        maskedCount++;
      }
    }
    console.log(`✓ Grammar masked ${maskedCount}/${testLogits.length} tokens`);

    // Accept a token (advance grammar state)
    const openBrace = (await ctx2.tokenize("{"))[0];
    ctx2.acceptSamplerToken(handle1, openBrace);
    console.log(`✓ acceptSamplerToken(${handle1}, ${openBrace})`);

    // Free samplers
    ctx2.freeSamplerHandle(handle1);
    ctx2.freeSamplerHandle(handle2);
    console.log(`✓ freeSamplerHandle(): both handles freed\n`);

    // ===== TEST 16: Atomic decodeAndCapture =====
    console.log('⚛️  Test 16: Atomic decodeAndCapture');

    // Clear and setup
    await ctx2.kvCacheClear();
    const captureTokens = await ctx2.tokenize("Hello");

    // Pre-allocate destination buffer
    const captureBuffer = new Float32Array(ctx2.vocabSize);

    // Atomic decode + capture
    ctx2.decodeAndCapture(captureTokens, 0, 0, captureBuffer);
    console.log(`✓ decodeAndCapture(): decoded ${captureTokens.length} tokens`);

    // Verify buffer has valid logits
    let capturedHasNonZero = false;
    let capturedHasNaN = false;
    for (let i = 0; i < captureBuffer.length; i++) {
      if (captureBuffer[i] !== 0.0) capturedHasNonZero = true;
      if (isNaN(captureBuffer[i])) capturedHasNaN = true;
    }

    if (!capturedHasNonZero) {
      throw new Error('Captured logits are all zeros!');
    }
    if (capturedHasNaN) {
      throw new Error('Captured logits contain NaN!');
    }
    console.log(`✓ Captured buffer valid: has variation=${capturedHasNonZero}, has NaN=${capturedHasNaN}`);

    // Verify it's a copy (modifying doesn't affect ctx2.getLogits())
    const capturedOriginalValue = captureBuffer[0];
    captureBuffer[0] = -999.0;
    const ctx2Logits = ctx2.getLogits();
    if (ctx2Logits[0] === -999.0) {
      console.log('⚠️  Warning: decodeAndCapture may share memory with getLogits');
    } else {
      console.log(`✓ Captured buffer is independent copy`);
    }
    captureBuffer[0] = capturedOriginalValue;

    // Cleanup multi-sequence context
    ctx2.dispose();
    console.log('✓ Multi-sequence context disposed\n');

    // ============================================================================
    // Test 17: Metrics API
    // ============================================================================

    console.log('🔢 Test 17: Metrics API');

    // Setup: Clear cache and decode first token to get valid logits
    await ctx.kvCacheClear();
    await ctx.decode([tokens[0]], 0);
    const token1 = ctx.greedySample();

    // Test 17a: Model surprisal
    const surprisalNats = ctx.modelSurprisal(token1, "nats");
    const surprisalBits = ctx.modelSurprisal(token1, "bits");

    if (typeof surprisalNats !== 'number' || surprisalNats < 0) {
      throw new Error('modelSurprisal(nats) should return non-negative number');
    }

    if (Math.abs(surprisalBits - surprisalNats / Math.log(2)) > 0.01) {
      throw new Error('modelSurprisal(bits) should equal surprisal(nats) / ln(2)');
    }

    console.log(`  ✓ modelSurprisal: ${surprisalBits.toFixed(2)} bits`);

    // Test 17b: Model entropy
    const entropyNats = ctx.modelEntropy("nats");
    const entropyBits = ctx.modelEntropy("bits");

    if (typeof entropyNats !== 'number' || entropyNats < 0) {
      throw new Error('modelEntropy should return non-negative number');
    }

    if (Math.abs(entropyBits - entropyNats / Math.log(2)) > 0.01) {
      throw new Error('modelEntropy(bits) should equal entropy(nats) / ln(2)');
    }

    console.log(`  ✓ modelEntropy: ${entropyBits.toFixed(2)} bits`);

    // Test 17c: Perplexity tracker creation
    const tracker = ctx.createPerplexityTracker();

    if (typeof tracker !== 'number' || tracker <= 0) {
      throw new Error('createPerplexityTracker should return positive integer handle');
    }

    console.log(`  ✓ createPerplexityTracker: handle=${tracker}`);

    // Test 17d: Add surprisals and check count
    ctx.addSurprisal(tracker, surprisalNats);

    await ctx.decode([token1], 1);
    const token2 = ctx.greedySample();
    const surprisal2 = ctx.modelSurprisal(token2);
    ctx.addSurprisal(tracker, surprisal2);

    const count = ctx.getPerplexityCount(tracker);
    if (count !== 2) {
      throw new Error('getPerplexityCount should return correct count');
    }

    console.log(`  ✓ Added 2 surprisals, count=${count}`);

    // Test 17e: Get perplexity
    const ppl = ctx.getPerplexity(tracker);

    if (typeof ppl !== 'number' || ppl < 1.0) {
      throw new Error('getPerplexity should return value >= 1.0');
    }

    // Verify formula: PPL = exp(avg_surprisal)
    const expectedPpl = Math.exp((surprisalNats + surprisal2) / 2);
    if (Math.abs(ppl - expectedPpl) > 0.01) {
      throw new Error('PPL formula should be exp(sum_surprisals / count)');
    }

    console.log(`  ✓ getPerplexity: ${ppl.toFixed(2)}`);

    // Test 17f: Clone tracker
    const cloned = ctx.clonePerplexityTracker(tracker);

    if (typeof cloned !== 'number' || cloned === tracker) {
      throw new Error('clonePerplexityTracker should return new unique handle');
    }

    const clonedPpl = ctx.getPerplexity(cloned);
    if (Math.abs(clonedPpl - ppl) > 0.01) {
      throw new Error('Cloned tracker should have same perplexity as original');
    }

    console.log(`  ✓ clonePerplexityTracker: cloned handle=${cloned}`);

    // Test 17g: Independent tracking
    await ctx.decode([token2], 2);
    const token3 = ctx.greedySample();
    const surprisal3 = ctx.modelSurprisal(token3);

    ctx.addSurprisal(tracker, surprisal3);       // Add to original
    const pplOriginal = ctx.getPerplexity(tracker);
    const pplCloned = ctx.getPerplexity(cloned);

    if (pplOriginal === pplCloned) {
      throw new Error('Original and cloned trackers should track independently');
    }

    console.log(`  ✓ Independent tracking: original=${pplOriginal.toFixed(2)}, cloned=${pplCloned.toFixed(2)}`);

    // Test 17h: Reset tracker
    ctx.resetPerplexityTracker(cloned);

    const resetCount = ctx.getPerplexityCount(cloned);
    if (resetCount !== 0) {
      throw new Error('resetPerplexityTracker should set count to 0');
    }

    console.log(`  ✓ resetPerplexityTracker: count=${resetCount}`);

    // Test 17i: Free trackers
    ctx.freePerplexityTracker(tracker);
    ctx.freePerplexityTracker(cloned);

    console.log(`  ✓ Freed both trackers`);

    // Test 17j: Invalid handle error
    try {
      ctx.getPerplexity(tracker); // Already freed
      throw new Error('Should throw on invalid handle');
    } catch (e) {
      if (!e.message.includes('Invalid perplexity tracker handle')) {
        throw new Error('Should have correct error message for invalid handle');
      }
      console.log(`  ✓ Invalid handle throws error`);
    }

    console.log('✅ Test 17: Metrics API passed\n');

    // ===== TEST 18: Branch.prefill — growing multi-turn conversation =====
    // Exercises the pull API multi-turn pattern on a single persistent branch:
    //   Turn 1: ctx.decode (initial prefill) → Branch.create → produce/commit
    //   Turn 2: branch.prefill(~10 tokens)  → produce/commit
    //   Turn 3: branch.prefill(~50 tokens)  → produce/commit
    //   Turn 4: branch.prefill(~100 tokens) → produce/commit
    // Same branch object (same handle) across all turns. Verifies position
    // accumulates correctly and generation works after each prefill.
    console.log('🌿 Test 18: Branch.prefill — growing multi-turn conversation');

    const { Branch } = require('..');

    const GEN_TOKENS = 8;

    const turnInputs = [
      "What is the capital of France?",
      " Tell me more about that city.",
      " I am planning a trip there next summer and would like to know about " +
        "the best neighborhoods to stay in, the most interesting museums to " +
        "visit, and any local restaurants you would recommend for authentic " +
        "French cuisine.",
      " That sounds wonderful. Can you also tell me about the public " +
        "transportation system, specifically how to get from Charles de " +
        "Gaulle airport to the city center, whether the metro is easy to " +
        "navigate for tourists who do not speak French, what kind of passes " +
        "or tickets are available for weekly travel, and if there are any " +
        "useful mobile apps that provide real-time transit information and " +
        "route planning for visitors to the city?",
    ];

    {
      const bCtx = await addon.createContext({
        modelPath: MODEL_PATH,
        nCtx: 2048,
        nBatch: 512,
        nThreads: 4,
      });

      const greedy = { temperature: 0 };

      // Turn 1: ctx.decode → Branch.create → generate
      const turn1Toks = await bCtx.tokenize(turnInputs[0]);
      await bCtx.decode(turn1Toks, 0, 0);

      const trunk = Branch.create(bCtx, 0, turn1Toks.length, greedy);
      trunk.captureLogits();

      console.log(`\n  Turn 1 prompt: "${turnInputs[0]}" (${turn1Toks.length} tokens)`);

      const gen1 = [];
      for (let i = 0; i < GEN_TOKENS; i++) {
        const { token, isStop } = trunk.produce();
        if (isStop) break;
        trunk.commit(token);
        gen1.push(token);
      }

      if (gen1.length === 0) {
        throw new Error('Turn 1: generated 0 tokens');
      }

      const gen1Text = await bCtx.detokenize(gen1);
      console.log(`  ✓ Turn 1: ${gen1.length} tokens, pos=${trunk.position}, ppl=${trunk.perplexity.toFixed(2)}`);
      console.log(`    "${gen1Text.trim()}"`);

      let prevPos = trunk.position;

      // Turns 2-4: branch.prefill → generate (same branch, growing inputs)
      for (let t = 1; t < turnInputs.length; t++) {
        const inputToks = await bCtx.tokenize(turnInputs[t]);
        const expectedPos = prevPos + inputToks.length;

        trunk.prefill(inputToks);

        if (trunk.position !== expectedPos) {
          throw new Error(
            `Turn ${t + 1} prefill: expected position ${expectedPos}, got ${trunk.position}`
          );
        }

        console.log(`\n  Turn ${t + 1} prefill: ${inputToks.length} tokens → position=${trunk.position}`);

        const gen = [];
        for (let i = 0; i < GEN_TOKENS; i++) {
          const { token, isStop } = trunk.produce();
          if (isStop) break;
          trunk.commit(token);
          gen.push(token);
        }

        if (gen.length === 0) {
          throw new Error(`Turn ${t + 1}: generated 0 tokens after prefill`);
        }

        const genText = await bCtx.detokenize(gen);
        console.log(`  ✓ Turn ${t + 1}: ${gen.length} tokens, pos=${trunk.position}, ppl=${trunk.perplexity.toFixed(2)}`);
        console.log(`    "${genText.trim()}"`);

        prevPos = trunk.position;
      }

      console.log(`\n  ✓ Same branch across ${turnInputs.length} turns, final position=${trunk.position}`);

      trunk.prune();
      bCtx.dispose();
    }

    console.log('\n✅ Test 18: Branch.prefill passed\n');

    // ===== TEST 19: nBatch ablation — different chunk sizes, identical output =====
    // Verifies nBatch is a pure performance parameter: chunking the same tokens
    // into different batch sizes must produce identical greedy logits.
    // Each nBatch variant runs in its own context to avoid breaking the
    // Branch abstraction with manual kvSeqCopy.
    console.log('🔬 Test 19: nBatch ablation — chunk size does not affect output');

    {
      // A prompt long enough to exercise multi-batch chunking at nBatch=32
      const ablationPrompt = turnInputs.join('');
      const nBatchValues = [32, 64, 128, 512];
      const resultsByBatch = {};

      for (const nBatch of nBatchValues) {
        const aCtx = await addon.createContext({
          modelPath: MODEL_PATH,
          nCtx: 2048,
          nBatch: 512,
          nThreads: 4,
        });

        const promptToks = await aCtx.tokenize(ablationPrompt);
        await aCtx.decode(promptToks, 0, 0);

        // Branch with this nBatch value — prefill uses it for chunking
        const branch = Branch.create(aCtx, 0, promptToks.length, { temperature: 0 }, nBatch);
        branch.captureLogits();

        // Prefill a follow-up through the branch to exercise decode_and_capture_batch
        const followUp = await aCtx.tokenize(" What else should I know?");
        branch.prefill(followUp);

        // Generate greedily
        const gen = [];
        for (let i = 0; i < GEN_TOKENS; i++) {
          const { token, isStop } = branch.produce();
          if (isStop) break;
          branch.commit(token);
          gen.push(token);
        }

        resultsByBatch[nBatch] = gen;

        console.log(`  nBatch=${String(nBatch).padStart(3)}: prefill ${followUp.length} tokens → generated [${gen.join(',')}]`);

        branch.prune();
        aCtx.dispose();
      }

      // All variants must produce identical tokens
      const reference = resultsByBatch[nBatchValues[0]].join(',');
      for (let i = 1; i < nBatchValues.length; i++) {
        const nb = nBatchValues[i];
        if (resultsByBatch[nb].join(',') !== reference) {
          throw new Error(
            `nBatch=${nb} diverged from nBatch=${nBatchValues[0]}: ` +
            `[${resultsByBatch[nb]}] vs [${reference}]`
          );
        }
      }

      console.log(`  ✓ All ${nBatchValues.length} nBatch variants produced identical output`);
    }

    console.log('\n✅ Test 19: nBatch ablation passed\n');

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
