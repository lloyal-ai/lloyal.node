/**
 * Integration tests - verify all SessionContext methods with real models
 *
 * Tests API functionality across different model architectures.
 * All tests use real models (no stubs/mocks).
 *
 * Usage:
 *   npm run test:integration
 *   MODEL_PATH=models/Llama-3.2-1B-Instruct-Q4_K_M.gguf npm run test:integration
 *
 * Optional embedding tests:
 *   LLAMA_EMBED_MODEL=models/nomic-embed-text-v1.5.Q4_K_M.gguf npm run test:integration
 */

const path = require('path');
const fs = require('fs');

const MODEL_PATH = process.env.MODEL_PATH
  ? path.resolve(process.env.MODEL_PATH)
  : path.join(__dirname, '../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf');
const EMBED_MODEL_PATH = process.env.LLAMA_EMBED_MODEL ||
  (fs.existsSync(path.join(__dirname, '../models/nomic-embed-text-v1.5.Q4_K_M.gguf'))
    ? path.join(__dirname, '../models/nomic-embed-text-v1.5.Q4_K_M.gguf')
    : null);

if (!fs.existsSync(MODEL_PATH)) {
  console.error('Test model not found:', MODEL_PATH);
  process.exit(1);
}

console.log('=== lloyal.node Integration Tests ===\n');
console.log(`Model: ${path.basename(MODEL_PATH)}`);
console.log(`Size: ${(fs.statSync(MODEL_PATH).size / 1024 / 1024).toFixed(1)} MB\n`);

const { loadBinary, Branch, withLogits } = require('..');
let addon;
try {
  addon = require('../build/Release/lloyal.node');
} catch {
  addon = loadBinary();
}

// Test tracking
let passed = 0;
let failed = 0;

function ok(msg) {
  passed++;
  console.log(`  [PASS] ${msg}`);
}

function fail(msg) {
  failed++;
  console.log(`  [FAIL] ${msg}`);
}

function assert(condition, msg) {
  if (condition) {
    ok(msg);
  } else {
    fail(msg);
    throw new Error(msg);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// CORE API TESTS
// ═══════════════════════════════════════════════════════════════════════════

async function testCoreAPI(ctx) {
  console.log('\n--- Core API ---');

  // createContext validated by caller

  // tokenize / detokenize
  const text = "Hello world";
  const tokens = await ctx.tokenize(text);
  assert(tokens.length > 0, `tokenize("${text}") → ${tokens.length} tokens`);

  const reconstructed = await ctx.detokenize(tokens);
  assert(typeof reconstructed === 'string', `detokenize() → "${reconstructed}"`);

  // tokenToText
  const tokenText = ctx.tokenToText(tokens[0]);
  assert(typeof tokenText === 'string', `tokenToText(${tokens[0]}) → "${tokenText}"`);

  // decode + getLogits
  await ctx.decode(tokens, 0);
  const logits = ctx.getLogits();
  assert(logits instanceof Float32Array, `getLogits() → Float32Array(${logits.length})`);
  assert(logits.length === ctx.vocabSize, `logits.length === vocabSize (${ctx.vocabSize})`);

  // Validate logits are not garbage
  let hasNonZero = false, hasNaN = false;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] !== 0.0) hasNonZero = true;
    if (isNaN(logits[i])) hasNaN = true;
  }
  assert(hasNonZero && !hasNaN, 'logits valid (non-zero, no NaN)');

  // modelEntropy
  const entropy = ctx.modelEntropy();
  assert(isFinite(entropy) && entropy >= 0, `modelEntropy() → ${entropy.toFixed(4)} nats`);

  // greedySample
  const greedy = ctx.greedySample();
  assert(greedy >= 0 && greedy < ctx.vocabSize, `greedySample() → ${greedy}`);

  // sample with params
  const sampled = ctx.sample({ temperature: 0 });
  assert(sampled === greedy, `sample({temp:0}) === greedySample() (${sampled})`);

  // isStopToken - EOS should be a stop token
  const eos = ctx.getEogToken();
  assert(ctx.isStopToken(eos), `isStopToken(EOS=${eos}) → true`);

  // Logits memoization
  const logits1 = ctx.getLogits();
  const logits2 = ctx.getLogits();
  assert(logits1[0] === logits2[0], 'getLogits() memoized (same step = same buffer)');

  // withLogits helper
  const maxLogit = withLogits(ctx, (l) => {
    let max = l[0];
    for (let i = 1; i < l.length; i++) if (l[i] > max) max = l[i];
    return max;
  });
  assert(isFinite(maxLogit), `withLogits() sync → max=${maxLogit.toFixed(2)}`);

  let asyncRejected = false;
  try {
    withLogits(ctx, async () => 1);
  } catch {
    asyncRejected = true;
  }
  assert(asyncRejected, 'withLogits() rejects async callbacks');
}

// ═══════════════════════════════════════════════════════════════════════════
// KV CACHE TESTS
// ═══════════════════════════════════════════════════════════════════════════

async function testKVCache(ctx) {
  console.log('\n--- KV Cache ---');

  await ctx.kvCacheClear();
  const tokens = await ctx.tokenize("Test prompt");
  await ctx.decode(tokens, 0);

  const sizeBefore = ctx.kvCacheSize();
  assert(sizeBefore >= 0, `kvCacheSize() after decode → ${sizeBefore}`);

  await ctx.kvCacheClear();
  const sizeAfter = ctx.kvCacheSize();
  assert(sizeAfter === -1, `kvCacheClear() → size=${sizeAfter} (empty)`);
}

// ═══════════════════════════════════════════════════════════════════════════
// MULTI-SEQUENCE TESTS
// ═══════════════════════════════════════════════════════════════════════════

async function testMultiSequence() {
  console.log('\n--- Multi-Sequence KV ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: 512,
    nThreads: 4,
    nSeqMax: 4
  });

  try {
    const tokens = await ctx.tokenize("The quick brown fox");
    await ctx.decode(tokens, 0, 0);

    const seq0Pos = ctx.kvSeqPosMax(0);
    assert(seq0Pos >= 0, `kvSeqPosMax(0) → ${seq0Pos}`);

    const seq1Before = ctx.kvSeqPosMax(1);
    assert(seq1Before === -1, `kvSeqPosMax(1) before copy → ${seq1Before} (empty)`);

    ctx.kvSeqCopy(0, 1);
    const seq1After = ctx.kvSeqPosMax(1);
    assert(seq1After === seq0Pos, `kvSeqCopy(0,1) → seq1 pos=${seq1After}`);

    const seq0After = ctx.kvSeqPosMax(0);
    assert(seq0After === seq0Pos, `seq0 unchanged after copy → ${seq0After}`);
  } finally {
    ctx.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// GRAMMAR TESTS
// ═══════════════════════════════════════════════════════════════════════════

async function testGrammar() {
  console.log('\n--- Grammar Sampling ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: 512,
    nThreads: 4
  });

  try {
    const grammar = `root ::= "{" ws "}" ws
ws ::= [ \\t\\n]*`;

    // Handle-based API
    const handle = ctx.createSampler(grammar);
    assert(typeof handle === 'number' && handle > 0, `createSampler() → handle=${handle}`);

    const cloned = ctx.cloneSampler(handle);
    assert(cloned !== handle, `cloneSampler() → new handle=${cloned}`);

    const testLogits = new Float32Array(ctx.vocabSize).fill(0.5);
    ctx.applySampler(handle, testLogits);

    let masked = 0, validToken = -1;
    for (let i = 0; i < testLogits.length; i++) {
      if (testLogits[i] < -1e30) masked++;
      else if (validToken === -1) validToken = i;
    }
    assert(masked > 0 && validToken >= 0, `applySampler() masked ${masked} tokens`);

    ctx.acceptSamplerToken(handle, validToken);
    ok(`acceptSamplerToken(${validToken})`);

    ctx.freeSamplerHandle(handle);
    ctx.freeSamplerHandle(cloned);
    ok('freeSamplerHandle() both handles');

    // Branch API with grammar
    await ctx.kvCacheClear();
    const prompt = await ctx.tokenize("Output: ");
    await ctx.decode(prompt, 0, 0);

    const branch = Branch.create(ctx, 0, prompt.length, { temperature: 0 }, undefined, grammar);
    branch.captureLogits();

    const output = [];
    for (let i = 0; i < 10; i++) {
      const { token, text, isStop } = branch.produce();
      if (isStop) break;
      branch.commit(token);
      output.push(text);
    }

    const result = output.join('');
    assert(/^\{\s*\}\s*$/.test(result), `Branch+grammar → "${result}"`);
    branch.prune();
  } finally {
    ctx.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// METRICS API TESTS
// ═══════════════════════════════════════════════════════════════════════════

async function testMetrics(ctx) {
  console.log('\n--- Metrics API ---');

  await ctx.kvCacheClear();
  const tokens = await ctx.tokenize("Hello");
  await ctx.decode(tokens, 0);

  const token1 = ctx.greedySample();
  const surprisal = ctx.modelSurprisal(token1, "nats");
  assert(surprisal >= 0, `modelSurprisal() → ${surprisal.toFixed(2)} nats`);

  const surprisalBits = ctx.modelSurprisal(token1, "bits");
  assert(Math.abs(surprisalBits - surprisal / Math.log(2)) < 0.01, 'bits = nats / ln(2)');

  const tracker = ctx.createPerplexityTracker();
  assert(tracker > 0, `createPerplexityTracker() → ${tracker}`);

  ctx.addSurprisal(tracker, surprisal);
  await ctx.decode([token1], tokens.length);
  ctx.addSurprisal(tracker, ctx.modelSurprisal(ctx.greedySample()));

  const count = ctx.getPerplexityCount(tracker);
  assert(count === 2, `getPerplexityCount() → ${count}`);

  const ppl = ctx.getPerplexity(tracker);
  assert(ppl >= 1.0, `getPerplexity() → ${ppl.toFixed(2)}`);

  const clonedTracker = ctx.clonePerplexityTracker(tracker);
  assert(clonedTracker !== tracker, `clonePerplexityTracker() → ${clonedTracker}`);

  ctx.resetPerplexityTracker(clonedTracker);
  assert(ctx.getPerplexityCount(clonedTracker) === 0, 'resetPerplexityTracker() → count=0');

  ctx.freePerplexityTracker(tracker);
  ctx.freePerplexityTracker(clonedTracker);
  ok('freePerplexityTracker() both');

  let threwOnInvalid = false;
  try {
    ctx.getPerplexity(tracker);
  } catch {
    threwOnInvalid = true;
  }
  assert(threwOnInvalid, 'Invalid handle throws');
}

// ═══════════════════════════════════════════════════════════════════════════
// BRANCH PREFILL TESTS
// ═══════════════════════════════════════════════════════════════════════════

async function testBranchPrefill() {
  console.log('\n--- Branch.prefill Multi-Turn ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: 2048,
    nBatch: 512,
    nThreads: 4
  });

  try {
    const GEN_TOKENS = 5;
    const turns = [
      "What is the capital of France?",
      " Tell me more.",
      " What about transportation?"
    ];

    const messages = [{ role: 'user', content: turns[0] }];
    const { prompt } = await ctx.formatChat(JSON.stringify(messages));
    const promptToks = await ctx.tokenize(prompt);
    await ctx.decode(promptToks, 0, 0);

    const branch = Branch.create(ctx, 0, promptToks.length, { temperature: 0 });
    branch.captureLogits();

    // Turn 1
    const gen1 = [];
    for (let i = 0; i < GEN_TOKENS; i++) {
      const { token, isStop } = branch.produce();
      if (isStop) break;
      branch.commit(token);
      gen1.push(token);
    }
    assert(gen1.length > 0, `Turn 1: generated ${gen1.length} tokens`);

    // Track assistant response for string-diff warm continuation
    const assistantText1 = await ctx.detokenize(gen1);
    messages.push({ role: 'assistant', content: assistantText1 });

    // Warm continuation: string-diff formatChat() + turn separator
    const sep = ctx.getTurnSeparator();

    // Turn 2-3: prefill using string-diff warm pattern + generate
    for (let t = 1; t < turns.length; t++) {
      messages.push({ role: 'user', content: turns[t] });
      const { prompt: full } = await ctx.formatChat(JSON.stringify(messages));
      const { prompt: prefix } = await ctx.formatChat(
        JSON.stringify(messages.slice(0, -1)),
        { addGenerationPrompt: false }
      );
      const delta = await ctx.tokenize(full.substring(prefix.length), false);
      const prefillToks = [...sep, ...delta];

      const posBefore = branch.position;
      branch.prefill(prefillToks);
      assert(branch.position === posBefore + prefillToks.length,
        `Turn ${t + 1}: prefill ${prefillToks.length} tokens → pos=${branch.position}`);

      const gen = [];
      for (let i = 0; i < GEN_TOKENS; i++) {
        const { token, isStop } = branch.produce();
        if (isStop) break;
        branch.commit(token);
        gen.push(token);
      }
      assert(gen.length > 0, `Turn ${t + 1}: generated ${gen.length} tokens`);

      // Track assistant response
      const assistantText = await ctx.detokenize(gen);
      messages.push({ role: 'assistant', content: assistantText });
    }

    branch.prune();
  } finally {
    ctx.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// WARM vs COLD PARITY - Semantic proof that warm continuation == cold start
// ═══════════════════════════════════════════════════════════════════════════

async function testWarmColdParity() {
  console.log('\n--- Warm vs Cold Parity ---');

  const GEN_TOKENS = 10;
  const userMessages = [
    "What is the capital of France?",
    " Tell me more about it."
  ];

  // === WARM PATH: decode turn 1, prefill turn 2 delta, generate ===
  const warmCtx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: 2048,
    nBatch: 512,
    nThreads: 4
  });

  let assistantContent;
  let warmGen2;

  try {
    // Turn 1: format, decode, generate
    const msgs1 = [{ role: 'user', content: userMessages[0] }];
    const { prompt: prompt1 } = await warmCtx.formatChat(JSON.stringify(msgs1));
    const toks1 = await warmCtx.tokenize(prompt1);
    await warmCtx.decode(toks1, 0, 0);

    const branch = Branch.create(warmCtx, 0, toks1.length, { temperature: 0 });
    branch.captureLogits();

    const gen1 = [];
    for (let i = 0; i < GEN_TOKENS; i++) {
      const { token, isStop } = branch.produce();
      if (isStop) break;
      branch.commit(token);
      gen1.push(token);
    }

    assistantContent = await warmCtx.detokenize(gen1);

    // Turn 2: string-diff warm continuation
    const sep = warmCtx.getTurnSeparator();
    const allMessages = [
      { role: 'user', content: userMessages[0] },
      { role: 'assistant', content: assistantContent },
      { role: 'user', content: userMessages[1] }
    ];
    const { prompt: full } = await warmCtx.formatChat(JSON.stringify(allMessages));
    const { prompt: prefix } = await warmCtx.formatChat(
      JSON.stringify(allMessages.slice(0, -1)),
      { addGenerationPrompt: false }
    );
    const deltaToks = await warmCtx.tokenize(full.substring(prefix.length), false);
    branch.prefill([...sep, ...deltaToks]);

    warmGen2 = [];
    for (let i = 0; i < GEN_TOKENS; i++) {
      const { token, isStop } = branch.produce();
      if (isStop) break;
      branch.commit(token);
      warmGen2.push(token);
    }

    branch.prune();
  } finally {
    warmCtx.dispose();
  }

  // === COLD PATH: decode full 2-turn conversation from scratch, generate ===
  const coldCtx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: 2048,
    nBatch: 512,
    nThreads: 4
  });

  let coldGen2;

  try {
    const msgs = [
      { role: 'user', content: userMessages[0] },
      { role: 'assistant', content: assistantContent },
      { role: 'user', content: userMessages[1] }
    ];
    const { prompt: coldPrompt } = await coldCtx.formatChat(JSON.stringify(msgs));
    const coldToks = await coldCtx.tokenize(coldPrompt);
    await coldCtx.decode(coldToks, 0, 0);

    const branch = Branch.create(coldCtx, 0, coldToks.length, { temperature: 0 });
    branch.captureLogits();

    coldGen2 = [];
    for (let i = 0; i < GEN_TOKENS; i++) {
      const { token, isStop } = branch.produce();
      if (isStop) break;
      branch.commit(token);
      coldGen2.push(token);
    }

    branch.prune();
  } finally {
    coldCtx.dispose();
  }

  // === COMPARE ===
  const warmStr = warmGen2.join(',');
  const coldStr = coldGen2.join(',');

  // Log divergence diagnostics BEFORE assert (assert throws on failure)
  if (warmStr !== coldStr) {
    for (let i = 0; i < Math.max(warmGen2.length, coldGen2.length); i++) {
      if (warmGen2[i] !== coldGen2[i]) {
        console.log(`  First divergence at position ${i}: warm=${warmGen2[i]} cold=${coldGen2[i]}`);
        break;
      }
    }
  }

  assert(warmStr === coldStr,
    warmStr === coldStr
      ? `Warm==Cold parity: ${warmGen2.length} tokens match`
      : `Warm==Cold parity FAILED: warm=[${warmStr}] vs cold=[${coldStr}]`);
}

// ═══════════════════════════════════════════════════════════════════════════
// WARM CONTINUATION SEMANTIC RECALL - Proves context survives delta-only prefill
// ═══════════════════════════════════════════════════════════════════════════

async function testWarmSemanticRecall() {
  if (!EMBED_MODEL_PATH) {
    console.log('\n--- Warm Semantic Recall (SKIPPED - no LLAMA_EMBED_MODEL) ---');
    return;
  }

  console.log('\n--- Warm Semantic Recall ---');

  const GEN_TOKENS = 40;

  // Helper: cosine similarity
  function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  }

  // Phase 1: Generate multi-turn conversation via warm continuation
  let recallText;
  {
    const ctx = await addon.createContext({
      modelPath: MODEL_PATH,
      nCtx: 2048,
      nBatch: 512,
      nThreads: 4
    });

    try {
      const sep = ctx.getTurnSeparator();
      let branch;
      const messages = [];

      // Helper: string-diff warm continuation
      async function warmTurn(userContent) {
        messages.push({ role: 'user', content: userContent });
        const { prompt: full } = await ctx.formatChat(JSON.stringify(messages));
        const { prompt: prefix } = await ctx.formatChat(
          JSON.stringify(messages.slice(0, -1)),
          { addGenerationPrompt: false }
        );
        const delta = await ctx.tokenize(full.substring(prefix.length), false);
        branch.prefill([...sep, ...delta]);

        const gen = [];
        for (let i = 0; i < GEN_TOKENS; i++) {
          const { token, isStop } = branch.produce();
          if (isStop) break;
          branch.commit(token);
          gen.push(token);
        }
        const text = await ctx.detokenize(gen);
        messages.push({ role: 'assistant', content: text });
        return text;
      }

      // Turn 1: Plant a specific, recallable fact
      messages.push({ role: 'user', content: 'Remember this: my dog is named Max.' });
      const { prompt } = await ctx.formatChat(JSON.stringify(messages));
      const promptToks = await ctx.tokenize(prompt);
      await ctx.decode(promptToks, 0, 0);

      branch = Branch.create(ctx, 0, promptToks.length, { temperature: 0 });
      branch.captureLogits();

      // Generate turn 1 response
      const gen = [];
      for (let i = 0; i < GEN_TOKENS; i++) {
        const { token, isStop } = branch.produce();
        if (isStop) break;
        branch.commit(token);
        gen.push(token);
      }
      const turn1Response = await ctx.detokenize(gen);
      messages.push({ role: 'assistant', content: turn1Response });

      // Turn 2: Distractor
      await warmTurn('What is 2 + 2?');

      // Turn 3: Another distractor
      await warmTurn('Name three colors.');

      // Turn 4: Recall — only answerable from turn 1 context
      recallText = await warmTurn('What is my dog\'s name?');

      branch.prune();
    } finally {
      ctx.dispose();
    }
  }

  // Phase 2: Score via embedding similarity (chat model fully released)
  {
    const embedCtx = await addon.createContext({
      modelPath: EMBED_MODEL_PATH,
      nCtx: 512,
      nBatch: 512,
      nThreads: 4,
      embeddings: true,
      poolingType: 1  // MEAN
    });

    try {
      async function embed(text) {
        const tokens = await embedCtx.tokenize(text);
        await embedCtx.kvCacheClear();
        await embedCtx.encode(tokens);
        return embedCtx.getEmbeddings(true);
      }

      console.log(`  Recall response: "${recallText.trim().slice(0, 120)}"`);

      const embResponse = await embed(recallText);
      const embCorrect = await embed('The dog is named Max.');
      const embWrong = await embed('Red, blue, and green are three colors.');

      const simCorrect = cosine(embResponse, embCorrect);
      const simWrong = cosine(embResponse, embWrong);

      assert(simCorrect > simWrong,
        `Semantic recall: correct=${simCorrect.toFixed(3)} > wrong=${simWrong.toFixed(3)}`);
    } finally {
      embedCtx.dispose();
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// BRANCH STEER TESTS - Dynamic per-sample logit manipulation
// ═══════════════════════════════════════════════════════════════════════════

async function testBranchSteer() {
  console.log('\n--- Branch.steer ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: 512,
    nThreads: 4
  });

  try {
    const tokens = await ctx.tokenize("The quick brown");
    await ctx.decode(tokens, 0, 0);

    // Use greedy sampling for deterministic tests
    const branch = Branch.create(ctx, 0, tokens.length, { temperature: 0 });
    branch.captureLogits();

    // Get the greedy token (what would be sampled without steer)
    const greedyToken = branch.sample();
    assert(greedyToken >= 0, `Greedy sample → ${greedyToken}`);

    // Block the greedy token with steer
    branch.steer([{ token: greedyToken, bias: -Infinity }]);

    // Sample again - should get a different token
    const steeredToken = branch.sample();
    assert(steeredToken !== greedyToken,
      `steer() blocks greedy: ${greedyToken} → ${steeredToken}`);

    // Clear steer - should get greedy token again
    branch.clearSteer();
    const afterClear = branch.sample();
    assert(afterClear === greedyToken,
      `clearSteer() restores greedy: ${afterClear} === ${greedyToken}`);

    // Test multiple blocks
    branch.steer([
      { token: greedyToken, bias: -Infinity },
      { token: steeredToken, bias: -Infinity },
    ]);
    const doubleBlocked = branch.sample();
    assert(doubleBlocked !== greedyToken && doubleBlocked !== steeredToken,
      `Multiple blocks: ${doubleBlocked} ≠ {${greedyToken}, ${steeredToken}}`);

    // Test boost (positive bias)
    branch.clearSteer();
    branch.steer([{ token: 42, bias: 100.0 }]);  // Massive boost to token 42
    const boosted = branch.sample();
    assert(boosted === 42, `Boost token 42 → ${boosted}`);

    branch.prune();
    ok('steer()/clearSteer() work correctly');

    // Test fork invariant: steer is NOT cloned on fork
    const tokens2 = await ctx.tokenize("Hello world");
    await ctx.decode(tokens2, 0, 0);

    const parent = Branch.create(ctx, 0, tokens2.length, { temperature: 0 });
    parent.captureLogits();

    const parentGreedy = parent.sample();

    // Apply steer to parent - block the greedy token
    parent.steer([{ token: parentGreedy, bias: -Infinity }]);
    const parentSteered = parent.sample();
    assert(parentSteered !== parentGreedy, `Parent steered: ${parentSteered} ≠ ${parentGreedy}`);

    // Fork from parent - child should NOT inherit steer
    const child = parent.fork(1);
    const childSample = child.sample();
    assert(childSample === parentGreedy,
      `Fork does NOT inherit steer: child=${childSample} === greedy=${parentGreedy}`);

    // Verify parent still has steer active
    const parentStillSteered = parent.sample();
    assert(parentStillSteered === parentSteered,
      `Parent retains steer after fork: ${parentStillSteered} === ${parentSteered}`);

    // Apply different steer to child - should not affect parent
    child.steer([{ token: 99, bias: 100.0 }]);
    const childBoosted = child.sample();
    assert(childBoosted === 99, `Child can set own steer: ${childBoosted} === 99`);

    // Parent should be unaffected by child's steer
    const parentUnaffected = parent.sample();
    assert(parentUnaffected === parentSteered,
      `Parent unaffected by child steer: ${parentUnaffected} === ${parentSteered}`);

    child.prune();
    parent.prune();
    ok('steer() NOT cloned on fork (fork invariant)');
  } finally {
    ctx.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// NBATCH ABLATION - Chunk size must not affect output
// ═══════════════════════════════════════════════════════════════════════════

async function testNBatchAblation() {
  console.log('\n--- nBatch Ablation ---');

  const nBatchValues = [32, 64, 128, 512];
  const results = {};

  for (const nBatch of nBatchValues) {
    const ctx = await addon.createContext({
      modelPath: MODEL_PATH,
      nCtx: 1024,
      nBatch,
      nThreads: 4
    });

    try {
      const messages = [{ role: 'user', content: "Hello, how are you today?" }];
      const { prompt } = await ctx.formatChat(JSON.stringify(messages));
      const promptToks = await ctx.tokenize(prompt);
      await ctx.decode(promptToks, 0, 0);

      const branch = Branch.create(ctx, 0, promptToks.length, { temperature: 0 }, nBatch);
      branch.captureLogits();

      const followUp = await ctx.tokenize(" What else?");
      branch.prefill(followUp);

      const gen = [];
      for (let i = 0; i < 5; i++) {
        const { token, isStop } = branch.produce();
        if (isStop) break;
        branch.commit(token);
        gen.push(token);
      }

      results[nBatch] = gen.join(',');
      branch.prune();
    } finally {
      ctx.dispose();
    }
  }

  const ref = results[nBatchValues[0]];
  let allMatch = true;
  for (const nb of nBatchValues) {
    if (results[nb] !== ref) allMatch = false;
  }

  assert(allMatch, `All nBatch values produce identical output`);
}

// ═══════════════════════════════════════════════════════════════════════════
// TOKENIZER BEHAVIOR TESTS
// ═══════════════════════════════════════════════════════════════════════════

async function testTokenizer(ctx) {
  console.log('\n--- Tokenizer ---');

  // getEogToken
  const eog = ctx.getEogToken();
  assert(Number.isInteger(eog), `getEogToken() → ${eog}`);
  assert(ctx.isStopToken(eog), `EOS ${eog} is stop token`);

  const eogText = ctx.tokenToText(eog);
  assert(eogText.length > 0, `EOS text: "${eogText}"`);

  // tokenize with addSpecial
  const withSpecial = await ctx.tokenize('Hello world', true);
  const noSpecial = await ctx.tokenize('Hello world', false);

  assert(noSpecial.length <= withSpecial.length,
    `addSpecial=false (${noSpecial.length}) <= addSpecial=true (${withSpecial.length})`);

  // getTurnSeparator
  const sep = ctx.getTurnSeparator();
  assert(Array.isArray(sep) && sep.length > 0, `getTurnSeparator() → [${sep.join(',')}]`);

  const hasStop = sep.some(t => ctx.isStopToken(t));
  assert(hasStop, 'Separator contains stop token');

  const sepText = sep.map(t => ctx.tokenToText(t)).join('');
  ok(`Separator text: ${JSON.stringify(sepText)}`);

  // Caching
  const sep2 = ctx.getTurnSeparator();
  assert(sep.length === sep2.length && sep.every((t, i) => t === sep2[i]),
    'getTurnSeparator() cached');
}

// ═══════════════════════════════════════════════════════════════════════════
// DETERMINISM TEST - Same prompt must produce identical output
// ═══════════════════════════════════════════════════════════════════════════

async function testDeterminism() {
  console.log('\n--- Determinism ---');

  async function generate(prompt) {
    const ctx = await addon.createContext({
      modelPath: MODEL_PATH,
      nCtx: 512,
      nThreads: 4
    });

    try {
      const messages = [{ role: 'user', content: prompt }];
      const { prompt: formatted } = await ctx.formatChat(JSON.stringify(messages));
      const tokens = await ctx.tokenize(formatted);
      await ctx.decode(tokens, 0);

      const gen = [];
      for (let i = 0; i < 20; i++) {
        const token = ctx.sample({ temperature: 0 });
        if (ctx.isStopToken(token)) break;
        gen.push(token);
        await ctx.decode([token], tokens.length + i);
      }
      return gen.join(',');
    } finally {
      ctx.dispose();
    }
  }

  const prompt = "Count from 1 to 5.";
  const run1 = await generate(prompt);
  const run2 = await generate(prompt);

  assert(run1 === run2, `Deterministic: run1 === run2 (${run1.split(',').length} tokens)`);
}

// ═══════════════════════════════════════════════════════════════════════════
// EMBEDDING TESTS (optional)
// ═══════════════════════════════════════════════════════════════════════════

async function testEmbeddings() {
  if (!EMBED_MODEL_PATH) {
    console.log('\n--- Embeddings (SKIPPED - no LLAMA_EMBED_MODEL) ---');
    return;
  }

  console.log('\n--- Embeddings ---');
  console.log(`  Model: ${path.basename(EMBED_MODEL_PATH)}`);

  const ctx = await addon.createContext({
    modelPath: EMBED_MODEL_PATH,
    nCtx: 512,
    nBatch: 512,
    nThreads: 4,
    embeddings: true,
    poolingType: 1
  });

  try {
    assert(ctx.hasPooling(), 'hasPooling() → true');

    const dim = ctx.getEmbeddingDimension();
    assert(dim > 0, `getEmbeddingDimension() → ${dim}`);

    async function embed(text) {
      const tokens = await ctx.tokenize(text);
      await ctx.kvCacheClear();
      await ctx.encode(tokens);
      return ctx.getEmbeddings(true);
    }

    const emb1 = await embed("Hello world");
    assert(emb1.length === dim, `embed("Hello world") → Float32Array(${emb1.length})`);

    // L2 norm should be ~1.0
    let norm = 0;
    for (let i = 0; i < emb1.length; i++) norm += emb1[i] * emb1[i];
    norm = Math.sqrt(norm);
    assert(Math.abs(norm - 1.0) < 0.01, `L2 normalized: norm=${norm.toFixed(4)}`);

    // Cosine similarity
    function cosine(a, b) {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
      }
      return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    const emb1Copy = await embed("Hello world");
    const simIdentical = cosine(emb1, emb1Copy);
    assert(simIdentical > 0.99, `Identical texts similarity: ${simIdentical.toFixed(4)}`);

    const embSimilar = await embed("The cat sat on the mat");
    const embDifferent = await embed("Stock prices rose sharply");
    const embCat = await embed("A cat rested on the rug");

    const simSimilar = cosine(embSimilar, embCat);
    const simDifferent = cosine(embSimilar, embDifferent);
    assert(simSimilar > simDifferent,
      `Semantic: similar=${simSimilar.toFixed(3)} > different=${simDifferent.toFixed(3)}`);
  } finally {
    ctx.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// ATOMIC DECODE AND CAPTURE
// ═══════════════════════════════════════════════════════════════════════════

async function testDecodeAndCapture() {
  console.log('\n--- decodeAndCapture ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: 512,
    nThreads: 4
  });

  try {
    const tokens = await ctx.tokenize("Hello");
    const buffer = new Float32Array(ctx.vocabSize);

    ctx.decodeAndCapture(tokens, 0, 0, buffer);

    let valid = false;
    for (let i = 0; i < buffer.length; i++) {
      if (buffer[i] !== 0 && !isNaN(buffer[i])) valid = true;
    }
    assert(valid, `decodeAndCapture() filled buffer with valid logits`);

    // Verify it's a copy
    const orig = buffer[0];
    buffer[0] = -999;
    const ctxLogits = ctx.getLogits();
    const isCopy = ctxLogits[0] !== -999;
    buffer[0] = orig;
    assert(isCopy, 'Captured buffer is independent copy');
  } finally {
    ctx.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

async function testChatInOut(ctx) {
  console.log('\n── chat_in / chat_out ──');

  // formatChat with empty options object (new signature)
  const messages = [{ role: 'user', content: 'Hello' }];
  const result = await ctx.formatChat(JSON.stringify(messages), {});
  assert(result.prompt.includes('Hello'), 'formatChat with options: prompt contains Hello');
  assert(typeof result.format === 'number', 'formatChat returns format as number');
  assert(typeof result.grammar === 'string', 'formatChat returns grammar as string');
  assert(typeof result.grammarLazy === 'boolean', 'formatChat returns grammarLazy');
  assert(typeof result.thinkingForcedOpen === 'boolean', 'formatChat returns thinkingForcedOpen');
  assert(typeof result.reasoningFormat === 'number', 'formatChat returns reasoningFormat');
  assert(Array.isArray(result.grammarTriggers), 'formatChat returns grammarTriggers array');
  assert(Array.isArray(result.preservedTokens), 'formatChat returns preservedTokens array');
  ok('formatChat with options returns extended result');

  // Backward compat: string second argument still works
  const backCompat = await ctx.formatChat(JSON.stringify(messages));
  assert(backCompat.prompt.includes('Hello'), 'formatChat backward compat works');
  ok('formatChat backward compat (no second arg)');

  // formatChat with tools
  const tools = [{
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get weather',
      parameters: { type: 'object', properties: { location: { type: 'string' } } }
    }
  }];
  const toolResult = await ctx.formatChat(JSON.stringify(messages), {
    tools: JSON.stringify(tools),
    toolChoice: 'auto'
  });
  assert(typeof toolResult.format === 'number', 'formatChat with tools returns format');
  assert(typeof toolResult.grammar === 'string', 'formatChat with tools returns grammar');
  ok('formatChat with tools');

  // parseChatOutput
  const parsed = ctx.parseChatOutput('Hello world', toolResult.format);
  assert(typeof parsed.content === 'string', 'parseChatOutput returns content');
  assert(parsed.content.includes('Hello'), 'parseChatOutput content contains Hello');
  assert(typeof parsed.reasoningContent === 'string', 'parseChatOutput returns reasoningContent');
  assert(Array.isArray(parsed.toolCalls), 'parseChatOutput returns toolCalls array');
  ok('parseChatOutput basic');

  // parseChatOutput with options
  const parsedWithOpts = ctx.parseChatOutput('Some output', toolResult.format, {
    reasoningFormat: toolResult.reasoningFormat,
    isPartial: false,
    thinkingForcedOpen: false
  });
  assert(typeof parsedWithOpts.content === 'string', 'parseChatOutput with options');
  ok('parseChatOutput with options');
}

async function main() {
  let mainCtx = null;

  try {
    // Create main context for reusable tests
    mainCtx = await addon.createContext({
      modelPath: MODEL_PATH,
      nCtx: 512,
      nThreads: 4
    });
    ok(`createContext() → vocabSize=${mainCtx.vocabSize}`);

    // Run test suites
    await testCoreAPI(mainCtx);
    await testKVCache(mainCtx);
    await testMetrics(mainCtx);
    await testTokenizer(mainCtx);
    await testChatInOut(mainCtx);

    // Tests that create their own contexts
    await testMultiSequence();
    await testGrammar();
    await testBranchPrefill();
    await testWarmColdParity();
    await testWarmSemanticRecall();
    await testBranchSteer();
    await testNBatchAblation();
    await testDeterminism();
    await testDecodeAndCapture();
    await testEmbeddings();

    // Summary
    console.log('\n═══════════════════════════════════════');
    console.log(`PASSED: ${passed}`);
    console.log(`FAILED: ${failed}`);

    if (failed === 0) {
      console.log('\nAll tests passed!');
      process.exit(0);
    } else {
      console.log(`\n${failed} test(s) failed`);
      process.exit(1);
    }
  } catch (err) {
    console.error('\nFatal error:', err.message);
    console.error(err.stack);
    process.exit(1);
  } finally {
    if (mainCtx) mainCtx.dispose();
  }
}

main();
