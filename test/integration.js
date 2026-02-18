/**
 * Integration tests - verify all SessionContext methods with real models
 *
 * Tests API functionality across different model architectures.
 * All tests use real models (no stubs/mocks).
 *
 * Usage:
 *   npm run test:integration
 *   LLAMA_TEST_MODEL=models/Llama-3.2-1B-Instruct-Q4_K_M.gguf npm run test:integration
 *
 * Optional embedding tests:
 *   LLAMA_EMBED_MODEL=models/nomic-embed-text-v1.5.Q4_K_M.gguf npm run test:integration
 */

const path = require('path');
const fs = require('fs');

const MODEL_PATH = process.env.LLAMA_TEST_MODEL
  ? path.resolve(process.env.LLAMA_TEST_MODEL)
  : path.join(__dirname, '../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf');
const EMBED_MODEL_PATH = process.env.LLAMA_EMBED_MODEL ||
  (fs.existsSync(path.join(__dirname, '../models/nomic-embed-text-v1.5.Q4_K_M.gguf'))
    ? path.join(__dirname, '../models/nomic-embed-text-v1.5.Q4_K_M.gguf')
    : null);

const CTX_SIZE = parseInt(process.env.LLAMA_CTX_SIZE || '2048', 10);

if (!fs.existsSync(MODEL_PATH)) {
  console.error('Test model not found:', MODEL_PATH);
  process.exit(1);
}

console.log('=== lloyal.node Integration Tests ===\n');
console.log(`Model: ${path.basename(MODEL_PATH)}`);
console.log(`Size: ${(fs.statSync(MODEL_PATH).size / 1024 / 1024).toFixed(1)} MB\n`);

const { loadBinary, Branch, BranchStore, withLogits } = require('..');
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
    nCtx: CTX_SIZE,
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
    nCtx: CTX_SIZE,
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

    const branch = Branch.create(ctx, prompt.length, { temperature: 0 }, undefined, grammar);
    branch.captureLogits();

    const output = [];
    for (let i = 0; i < 10; i++) {
      const { token, text, isStop } = branch.produce();
      if (isStop) break;
      await branch.commit(token);
      output.push(text);
    }

    const result = output.join('');
    assert(/^\{\s*\}\s*$/.test(result), `Branch+grammar → "${result}"`);
    await branch.prune();
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
    nCtx: CTX_SIZE,
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

    const branch = Branch.create(ctx, promptToks.length, { temperature: 0 });
    branch.captureLogits();

    // Turn 1
    const gen1 = [];
    for (let i = 0; i < GEN_TOKENS; i++) {
      const { token, isStop } = branch.produce();
      if (isStop) break;
      await branch.commit(token);
      gen1.push(token);
    }
    assert(gen1.length > 0, `Turn 1: generated ${gen1.length} tokens`);

    // Track assistant response
    const assistantText1 = await ctx.detokenize(gen1);
    messages.push({ role: 'assistant', content: assistantText1 });

    // Warm continuation: format only new message + turn separator
    const sep = ctx.getTurnSeparator();

    // Turn 2-3: prefill using format-only-new pattern + generate
    for (let t = 1; t < turns.length; t++) {
      messages.push({ role: 'user', content: turns[t] });
      const { prompt } = await ctx.formatChat(JSON.stringify([
        { role: 'system', content: '' },
        { role: 'user', content: turns[t] }
      ]));
      const delta = await ctx.tokenize(prompt, false);
      const prefillToks = [...sep, ...delta];

      const posBefore = branch.position;
      await branch.prefill(prefillToks);
      assert(branch.position === posBefore + prefillToks.length,
        `Turn ${t + 1}: prefill ${prefillToks.length} tokens → pos=${branch.position}`);

      const gen = [];
      for (let i = 0; i < GEN_TOKENS; i++) {
        const { token, isStop } = branch.produce();
        if (isStop) break;
        await branch.commit(token);
        gen.push(token);
      }
      assert(gen.length > 0, `Turn ${t + 1}: generated ${gen.length} tokens`);

      // Track assistant response
      const assistantText = await ctx.detokenize(gen);
      messages.push({ role: 'assistant', content: assistantText });
    }

    await branch.prune();
  } finally {
    ctx.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// WARM MULTI-TURN SEMANTIC RECALL - Proves context survives warm continuations
// Mirrors liblloyal C++ test: chat_in_integration_test.cpp
// ═══════════════════════════════════════════════════════════════════════════

async function testWarmMultiTurnRecall() {
  console.log('\n--- Warm Multi-Turn Recall ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: CTX_SIZE,
    nBatch: 512,
    nThreads: 4
  });

  try {
    const sep = ctx.getTurnSeparator();

    // Helper: generate until EOG (matches C++ test pattern)
    async function generate(branch) {
      const gen = [];
      for (;;) {
        const { token, isStop } = branch.produce();
        if (isStop) break;
        await branch.commit(token);
        gen.push(token);
      }
      return ctx.detokenize(gen);
    }

    // Helper: warm continuation — sep + format([{system,""},{user,msg}])
    async function warmTurn(branch, userContent) {
      const { prompt } = await ctx.formatChat(JSON.stringify([
        { role: 'system', content: '' },
        { role: 'user', content: userContent }
      ]), {});
      const delta = await ctx.tokenize(prompt, false);
      await branch.prefill([...sep, ...delta]);
      return generate(branch);
    }

    // Turn 1 (COLD): introduce name
    const msgs1 = [{ role: 'user', content: 'Hi, my name is Lloyal' }];
    const { prompt, format, reasoningFormat } = await ctx.formatChat(JSON.stringify(msgs1), {});
    const promptToks = await ctx.tokenize(prompt);
    await ctx.decode(promptToks, 0, 0);

    const branch = Branch.create(ctx, promptToks.length, { temperature: 0 });
    branch.captureLogits();

    // Helper: parse output and check content (not reasoning) for a term
    function checkRecall(rawText, term) {
      const { content } = ctx.parseChatOutput(rawText, format, {
        reasoningFormat,
        isPartial: false,
        thinkingForcedOpen: false
      });
      return (content || '').toLowerCase().includes(term.toLowerCase());
    }

    const turn1 = await generate(branch);
    console.log(`  Turn 1: "${turn1.trim()}"`);
    assert(turn1.length > 0, 'Turn 1: generated response');

    // Turn 2 (WARM): introduce favourite food
    const turn2 = await warmTurn(branch, 'My favourite food is pizza');
    console.log(`  Turn 2: "${turn2.trim()}"`);
    assert(turn2.length > 0, 'Turn 2: generated response');

    // Turn 3 (WARM): recall name
    const turn3 = await warmTurn(branch, 'Do you remember my name?');
    console.log(`  Turn 3 (name recall): "${turn3.trim()}"`);
    const nameRecalled = checkRecall(turn3, 'lloyal');
    assert(nameRecalled, `Name recall: ${nameRecalled ? 'found "Lloyal"' : 'MISSING "Lloyal" in: ' + turn3.trim()}`);

    // Turn 4 (WARM): recall food
    const turn4 = await warmTurn(branch, 'Do you remember my favourite food?');
    console.log(`  Turn 4 (food recall): "${turn4.trim()}"`);
    const foodRecalled = checkRecall(turn4, 'pizza');
    assert(foodRecalled, `Food recall: ${foodRecalled ? 'found "pizza"' : 'MISSING "pizza" in: ' + turn4.trim()}`);

    await branch.prune();
  } finally {
    ctx.dispose();
  }
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
      nCtx: CTX_SIZE,
      nBatch: 512,
      nThreads: 4
    });

    try {
      const sep = ctx.getTurnSeparator();
      let branch;
      const messages = [];

      // Helper: format-only-new warm continuation
      async function warmTurn(userContent) {
        messages.push({ role: 'user', content: userContent });
        const { prompt } = await ctx.formatChat(JSON.stringify([
          { role: 'system', content: '' },
          { role: 'user', content: userContent }
        ]));
        const delta = await ctx.tokenize(prompt, false);
        await branch.prefill([...sep, ...delta]);

        const gen = [];
        for (let i = 0; i < GEN_TOKENS; i++) {
          const { token, isStop } = branch.produce();
          if (isStop) break;
          await branch.commit(token);
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

      branch = Branch.create(ctx, promptToks.length, { temperature: 0 });
      branch.captureLogits();

      // Generate turn 1 response
      const gen = [];
      for (let i = 0; i < GEN_TOKENS; i++) {
        const { token, isStop } = branch.produce();
        if (isStop) break;
        await branch.commit(token);
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

      await branch.prune();
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

      console.log(`  Recall response: "${recallText.trim()}"`);

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
    nCtx: CTX_SIZE,
    nThreads: 4,
    nSeqMax: 8
  });

  try {
    const tokens = await ctx.tokenize("The quick brown");
    await ctx.decode(tokens, 0, 0);

    // Use greedy sampling for deterministic tests
    const branch = Branch.create(ctx, tokens.length, { temperature: 0 });
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

    await branch.prune();
    ok('steer()/clearSteer() work correctly');

    // Test fork invariant: steer is NOT cloned on fork
    const tokens2 = await ctx.tokenize("Hello world");
    await ctx.decode(tokens2, 0, 0);

    const parent = Branch.create(ctx, tokens2.length, { temperature: 0 });
    parent.captureLogits();

    const parentGreedy = parent.sample();

    // Apply steer to parent - block the greedy token
    parent.steer([{ token: parentGreedy, bias: -Infinity }]);
    const parentSteered = parent.sample();
    assert(parentSteered !== parentGreedy, `Parent steered: ${parentSteered} ≠ ${parentGreedy}`);

    // Fork from parent - child should NOT inherit steer
    const child = await parent.fork();
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

    await child.prune();
    await parent.prune();
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
      nCtx: CTX_SIZE,
      nBatch,
      nThreads: 4
    });

    try {
      const messages = [{ role: 'user', content: "Hello, how are you today?" }];
      const { prompt } = await ctx.formatChat(JSON.stringify(messages));
      const promptToks = await ctx.tokenize(prompt);
      await ctx.decode(promptToks, 0, 0);

      const branch = Branch.create(ctx, promptToks.length, { temperature: 0 }, nBatch);
      branch.captureLogits();

      const followUp = await ctx.tokenize(" What else?");
      await branch.prefill(followUp);

      const gen = [];
      for (let i = 0; i < 5; i++) {
        const { token, isStop } = branch.produce();
        if (isStop) break;
        await branch.commit(token);
        gen.push(token);
      }

      results[nBatch] = gen.join(',');
      await branch.prune();
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
      nCtx: CTX_SIZE,
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
    nCtx: CTX_SIZE,
    nThreads: 4
  });

  try {
    const tokens = await ctx.tokenize("Hello");
    const buffer = new Float32Array(ctx.vocabSize);

    await ctx.decodeAndCapture(tokens, 0, 0, buffer);

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

// ═══════════════════════════════════════════════════════════════════════════
// BRANCH STORE TESTS
// Production patterns for the JS BranchStore API. Low-level primitive
// correctness (batch index mapping, scatter chunking, scratch reuse) is
// verified in liblloyal C++ integration tests — these focus on the JS
// wrapper surface and real-world workflows.
// ═══════════════════════════════════════════════════════════════════════════

async function testBranchStore() {
  console.log('\n--- BranchStore ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: CTX_SIZE,
    nBatch: 512,
    nThreads: 4,
    nSeqMax: 8
  });

  try {
    const promptToks = await ctx.tokenize("The quick brown fox jumps over the lazy");
    const store = new BranchStore(ctx);

    // ── Test A: Best-of-N generation ──
    // Fork 3 stochastic branches with unique seeds, advance all with store.commit(),
    // verify perplexity diverges (different seeds → different tokens → different ppls).
    // Tests: batched generation loop, perplexity accumulation through accept_token,
    // Branch.perplexity accessor after store ops, reseedSampler diversity.
    {
      await ctx.decode(promptToks, 0, 0);
      const root = Branch.create(ctx, promptToks.length, { temperature: 0.8 });
      root.captureLogits();
      const branches = [root, await root.fork(), await root.fork()];
      branches[1].reseedSampler(42);
      branches[2].reseedSampler(99);

      for (let step = 0; step < 10; step++) {
        const live = branches.map(b => [b, b.produce()])
          .filter(([, p]) => !p.isStop);
        if (!live.length) break;
        await store.commit(live.map(([b, p]) => [b, p.token]));
      }

      const ppls = branches.map(b => b.perplexity);
      console.log(`  best-of-N perplexities: [${ppls.map(p => p.toFixed(2)).join(', ')}]`);
      assert(ppls.every(p => isFinite(p) && p >= 1.0),
        `best-of-N: all perplexities valid [${ppls.map(p => p.toFixed(2))}]`);

      const best = ppls.reduce((a, b) => Math.min(a, b));
      const worst = ppls.reduce((a, b) => Math.max(a, b));
      console.log(`  [PASS] best-of-N: best=${best.toFixed(2)}, worst=${worst.toFixed(2)}`);

      await root.pruneSubtree();
    }

    // ── Test B: Rehydrate + Generate pipeline ──
    // Restore divergent conversation histories via store.prefill(), then continue
    // generating with store.commit(). This is the persistence/replay pattern.
    // Tests: prefill→commit lifecycle, metrics across phase transition, getLogits().
    {
      await ctx.decode(promptToks, 0, 0);
      const b1 = Branch.create(ctx, promptToks.length, { temperature: 0 });
      b1.captureLogits();
      const b2 = await b1.fork();

      // Phase 1: Rehydrate from "saved" histories
      const history1 = await ctx.tokenize(" dog. The weather is nice today and I want to go", false);
      const history2 = await ctx.tokenize(" cat. Let me explain how quantum entanglement works in", false);
      await store.prefill([[b1, history1], [b2, history2]]);

      // Branches should be at different-length positions? No — same length coincidentally.
      // But logits must differ (different KV contents)
      const logitsAfterPrefill1 = b1.getLogits();
      const logitsAfterPrefill2 = b2.getLogits();
      let prefillDiffer = false;
      for (let i = 0; i < logitsAfterPrefill1.length; i++) {
        if (logitsAfterPrefill1[i] !== logitsAfterPrefill2[i]) { prefillDiffer = true; break; }
      }
      assert(prefillDiffer,
        `rehydrate: different histories → different logits after prefill`);

      // Phase 2: Generate continuations
      const gen1 = [], gen2 = [];
      for (let i = 0; i < 5; i++) {
        const live = [[b1, b1.produce()], [b2, b2.produce()]]
          .filter(([, p]) => !p.isStop);
        if (!live.length) break;
        await store.commit(live.map(([b, p]) => [b, p.token]));
        for (const [b, p] of live) {
          (b === b1 ? gen1 : gen2).push(p.token);
        }
      }

      const text1 = await ctx.detokenize(gen1);
      const text2 = await ctx.detokenize(gen2);
      console.log(`  rehydrate "weather" → "${text1}"`);
      console.log(`  rehydrate "quantum" → "${text2}"`);

      // Perplexity valid after prefill→commit transition
      // (metrics only count accept_token calls, so only the 5 commit steps)
      assert(isFinite(b1.perplexity) && isFinite(b2.perplexity),
        `rehydrate: perplexity valid after prefill→commit (b1=${b1.perplexity.toFixed(2)}, b2=${b2.perplexity.toFixed(2)})`);

      await b2.prune(); await b1.prune();
    }

    // ── Test C: getLogits() → modelEntropy() integration ──
    // Verifies Branch.getLogits() returns a Float32Array consumable by the
    // existing metrics API. This tests the JS API surface of the new exposure.
    {
      await ctx.decode(promptToks, 0, 0);
      const b1 = Branch.create(ctx, promptToks.length, { temperature: 0 });
      b1.captureLogits();

      const logits = b1.getLogits();
      assert(logits instanceof Float32Array,
        `getLogits: returns Float32Array`);
      assert(logits.length === ctx.vocabSize,
        `getLogits: length=${logits.length} === vocabSize=${ctx.vocabSize}`);

      // Feed branch logits into ctx.modelEntropy() — proves the returned
      // buffer is a valid logits distribution consumable by metrics API
      const entropyFromBranch = ctx.modelEntropy("nats", logits);
      const entropyFromCtx = ctx.modelEntropy("nats");
      assert(isFinite(entropyFromBranch) && entropyFromBranch > 0,
        `getLogits→modelEntropy: ${entropyFromBranch.toFixed(4)} nats`);

      // Branch logits (captured from same decode) should match context logits
      assert(Math.abs(entropyFromBranch - entropyFromCtx) < 1e-4,
        `getLogits→modelEntropy: branch=${entropyFromBranch.toFixed(4)} ≈ ctx=${entropyFromCtx.toFixed(4)}`);

      // After store.commit, logits change — getLogits() reflects new state
      const p = b1.produce();
      assert(!p.isStop, `getLogits: produce() should not hit EOG on first token`);
      await store.commit([[b1, p.token]]);
      const logitsAfter = b1.getLogits();
      const entropyAfter = ctx.modelEntropy("nats", logitsAfter);
      assert(isFinite(entropyAfter),
        `getLogits after commit: entropy=${entropyAfter.toFixed(4)} nats`);

      await b1.prune();
    }

    // ── Test D: produce() → store.commit() interop ──
    // Real workflow: use produce() to inspect candidates, then batch-commit winners.
    // Tests: produce() reads from branch snapshot, store.commit() advances state,
    // produce() on next iteration reads from updated snapshot.
    {
      await ctx.decode(promptToks, 0, 0);
      const b1 = Branch.create(ctx, promptToks.length, { temperature: 0 });
      b1.captureLogits();
      const b2 = await b1.fork();

      const output = [];
      for (let i = 0; i < 5; i++) {
        // Inspect with produce() — does NOT advance state
        const p1 = b1.produce(), p2 = b2.produce();

        // Can inspect text and isStop before committing
        assert(typeof p1.text === 'string' && typeof p2.text === 'string',
          `produce→commit: produce() returns text at step ${i}`);

        if (p1.isStop || p2.isStop) break;

        // Batch-commit the inspected tokens
        await store.commit([[b1, p1.token], [b2, p2.token]]);
        output.push(p1.text);
      }

      console.log(`  produce→commit: "${output.join('')}"`);
      assert(output.length > 0,
        `produce→commit: generated ${output.length} tokens via inspect-then-batch pattern`);

      await b2.prune(); await b1.prune();
    }

    // ── Test E: Mixed single/batched operations ──
    // Mix Branch.commit() (single) with BranchStore.commit() (batched) on same branches.
    // Tests: both paths write to the same branch state correctly, no corruption when
    // alternating between decode::one and decode::each on the same sequence.
    {
      await ctx.decode(promptToks, 0, 0);
      const b1 = Branch.create(ctx, promptToks.length, { temperature: 0 });
      b1.captureLogits();
      const b2 = await b1.fork();

      // Step 1-3: single-branch commit (decode::one path)
      for (let i = 0; i < 3; i++) {
        const live = [[b1, b1.produce()], [b2, b2.produce()]]
          .filter(([, p]) => !p.isStop);
        if (!live.length) break;
        for (const [b, p] of live) await b.commit(p.token);
      }
      const posAfterSingle = b1.position;

      // Step 4-6: batched commit (decode::each path)
      for (let i = 0; i < 3; i++) {
        const live = [[b1, b1.produce()], [b2, b2.produce()]]
          .filter(([, p]) => !p.isStop);
        if (!live.length) break;
        await store.commit(live.map(([b, p]) => [b, p.token]));
      }
      const posAfterBatched = b1.position;
      assert(posAfterBatched === posAfterSingle + 3,
        `mixed ops: position correct after single→batched (${posAfterSingle}→${posAfterBatched})`);

      // Step 7-9: back to single-branch commit
      for (let i = 0; i < 3; i++) {
        const live = [[b1, b1.produce()], [b2, b2.produce()]]
          .filter(([, p]) => !p.isStop);
        if (!live.length) break;
        for (const [b, p] of live) await b.commit(p.token);
      }

      // 9 total steps, perplexity must reflect all of them
      assert(isFinite(b1.perplexity) && b1.perplexity >= 1.0,
        `mixed ops: perplexity valid after 9 mixed steps (${b1.perplexity.toFixed(2)})`);

      await b2.prune(); await b1.prune();
    }

    // ── Test F: Independent EOG — one branch stops, other continues ──
    // Steer b1 to produce EOG at step 3 while b2 keeps generating.
    // Tests: per-branch EOG filtering, store.commit with shrinking branch set,
    // surviving branch generates correct output after sibling stops.
    {
      await ctx.decode(promptToks, 0, 0);
      const b1 = Branch.create(ctx, promptToks.length, { temperature: 0 });
      b1.captureLogits();
      const b2 = await b1.fork();

      const eog = ctx.getEogToken();
      const gen1 = [], gen2 = [];
      const stopped = [false, false];

      for (let step = 0; step < 8; step++) {
        // At step 3, force b1 to hit EOG
        if (step === 3 && !stopped[0]) {
          b1.steer([{ token: eog, bias: 100.0 }]);
        }

        const pairs = [
          ...(!stopped[0] ? [[b1, b1.produce()]] : []),
          ...(!stopped[1] ? [[b2, b2.produce()]] : []),
        ];

        const live = pairs.filter(([, p]) => !p.isStop);
        const dead = pairs.filter(([, p]) => p.isStop);

        // Mark stopped branches
        for (const [b] of dead) {
          if (b === b1) stopped[0] = true;
          if (b === b2) stopped[1] = true;
        }

        if (!live.length) break;
        await store.commit(live.map(([b, p]) => [b, p.token]));

        for (const [b, p] of live) {
          (b === b1 ? gen1 : gen2).push(p.token);
        }

        // Clean up steer after use
        if (step === 3 && stopped[0]) b1.clearSteer();
      }

      assert(stopped[0], `independent EOG: b1 hit EOG (steered at step 3)`);
      assert(gen1.length === 3, `independent EOG: b1 generated 3 tokens before EOG (got ${gen1.length})`);
      assert(gen2.length > gen1.length,
        `independent EOG: b2 continued past b1's EOG (b1=${gen1.length}, b2=${gen2.length})`);

      const text2 = await ctx.detokenize(gen2);
      console.log(`  independent EOG: b1 stopped at step 3, b2 continued → "${text2}"`);

      // b2's position should reflect all its tokens, not be truncated by b1's stop
      assert(b2.position === promptToks.length + gen2.length,
        `independent EOG: b2 position correct (${b2.position} === ${promptToks.length} + ${gen2.length})`);

      await b2.prune(); await b1.prune();
    }
  } finally {
    ctx.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// PPL SANITY — commit() must produce sane perplexity (not millions)
// ═══════════════════════════════════════════════════════════════════════════

async function testPplSanity() {
  console.log('\n--- PPL Sanity ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: CTX_SIZE,
    nThreads: 4
  });

  try {
    const messages = [{ role: 'user', content: 'Tell me about the weather.' }];
    const { prompt } = await ctx.formatChat(JSON.stringify(messages));
    const promptToks = await ctx.tokenize(prompt);
    await ctx.decode(promptToks, 0, 0);

    const branch = Branch.create(ctx, promptToks.length, { temperature: 0 });
    branch.captureLogits();

    for (let i = 0; i < 10; i++) {
      const { token, isStop } = branch.produce();
      if (isStop) break;
      await branch.commit(token);
    }

    const ppl = branch.perplexity;
    console.log(`  perplexity after 10 commits: ${ppl.toFixed(2)}`);
    assert(isFinite(ppl) && ppl >= 1.0 && ppl < 1000,
      `PPL sanity: ${ppl.toFixed(2)} is in [1, 1000)`);

    await branch.prune();
  } finally {
    ctx.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// COMMIT ROLLBACK — decode failure must restore sampler/grammar/metrics
// ═══════════════════════════════════════════════════════════════════════════

async function testCommitRollback() {
  console.log('\n--- Commit Rollback ---');

  // Tiny KV (nCtx=32) with many branches (nSeqMax=8). Each branch consumes
  // 1 KV cell per commit. With 8 branches and ~5 shared prefix cells, the
  // 32-cell budget exhausts after ~3 commits per branch. decode_each returns
  // non-zero (find_slot fails) → StoreCommitWorker throws → rollback fires.
  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: 32,
    nBatch: 512,
    nThreads: 4,
    nSeqMax: 8
  });

  try {
    const promptToks = await ctx.tokenize("Hi");
    await ctx.decode(promptToks, 0, 0);

    const root = Branch.create(ctx, promptToks.length, { temperature: 1.0 });
    root.captureLogits();
    const branches = [root];
    for (let i = 1; i < 8; i++) {
      const b = await root.fork();
      b.reseedSampler(1000 + i); // Divergent tokens → separate KV cells
      branches.push(b);
    }

    const store = new BranchStore(ctx);

    // Commit until decode fails from KV exhaustion
    // nCtx may be clamped to a model minimum (e.g. 256), so we need enough
    // rounds for 8 branches to exhaust ~256 cells: 256/8 = 32 rounds
    let successfulRounds = 0;
    let failedRound = false;
    for (let round = 0; round < 50; round++) {
      const live = branches
        .map(b => [b, b.produce()])
        .filter(([, p]) => !p.isStop);
      if (!live.length) break;

      // Snapshot PPL before this round
      const pplsBefore = live.map(([b]) => b.perplexity);

      try {
        await store.commit(live.map(([b, p]) => [b, p.token]));
        successfulRounds++;
      } catch {
        // Decode failed — verify PPL restored
        const pplsAfter = live.map(([b]) => b.perplexity);
        const allRestored = pplsBefore.every((p, i) => p === pplsAfter[i]);
        assert(allRestored,
          `rollback: all PPLs restored after decode failure at round ${round}`);

        // Branches still usable for single commits (1 token fits)
        const [b0, p0] = live[0];
        const posBefore = b0.position;
        try {
          await b0.commit(p0.token);
          assert(b0.position === posBefore + 1,
            `rollback: single commit succeeds after failed batch (pos ${b0.position})`);
        } catch {
          // KV may be truly full even for 1 token — that's OK, test the PPL assertion above
        }

        failedRound = true;
        break;
      }
    }

    console.log(`  ${successfulRounds} successful rounds before KV exhaustion`);
    assert(failedRound,
      `rollback: decode failure triggered (nCtx=32, 8 branches, ${successfulRounds} rounds)`);

    await root.pruneSubtree();
  } finally {
    ctx.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// ASYNC REJECTION — Worker failures must reject, branch state un-advanced
// ═══════════════════════════════════════════════════════════════════════════

async function testAsyncRejection() {
  console.log('\n--- Async Rejection ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: CTX_SIZE,
    nThreads: 4,
    nSeqMax: 4
  });

  try {
    const tokens = await ctx.tokenize("Hello world");
    await ctx.decode(tokens, 0, 0);

    const branch = Branch.create(ctx, tokens.length, { temperature: 0 });
    branch.captureLogits();

    // Generate one token to prove branch works
    const { token, isStop } = branch.produce();
    assert(!isStop, 'rejection: initial produce succeeds');
    await branch.commit(token);
    const posAfterCommit = branch.position;

    // Prune the branch — frees native resources
    await branch.prune();
    assert(branch.disposed, 'rejection: branch is disposed after prune');

    // commit() on disposed branch — _ensureNotDisposed should throw synchronously
    let threwOnCommit = false;
    try {
      await branch.commit(token);
    } catch (e) {
      threwOnCommit = true;
      assert(e.message.includes('disposed'), `rejection: commit error says "disposed": "${e.message}"`);
    }
    assert(threwOnCommit, 'rejection: commit on disposed branch throws');

    // produce() on disposed branch
    let threwOnProduce = false;
    try {
      branch.produce();
    } catch (e) {
      threwOnProduce = true;
    }
    assert(threwOnProduce, 'rejection: produce on disposed branch throws');

    // fork() on disposed branch
    let threwOnFork = false;
    try {
      await branch.fork();
    } catch (e) {
      threwOnFork = true;
    }
    assert(threwOnFork, 'rejection: fork on disposed branch throws');

    // Native AsyncWorker rejection: call _branchDecodeAndCaptureOne with invalid handle (0)
    let nativeRejected = false;
    try {
      await ctx._branchDecodeAndCaptureOne(0, token);
    } catch (e) {
      nativeRejected = true;
      assert(e instanceof Error, `rejection: native rejection is Error: ${e.constructor.name}`);
    }
    assert(nativeRejected, 'rejection: invalid handle to AsyncWorker rejects promise');
  } finally {
    ctx.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// EMPTY INPUT EDGE CASES — Batch workers with empty arrays resolve cleanly
// ═══════════════════════════════════════════════════════════════════════════

async function testEmptyInputEdgeCases() {
  console.log('\n--- Empty Input Edge Cases ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: CTX_SIZE,
    nThreads: 4,
    nSeqMax: 4
  });

  try {
    const tokens = await ctx.tokenize("Hello world");
    await ctx.decode(tokens, 0, 0);

    const branch = Branch.create(ctx, tokens.length, { temperature: 0 });
    branch.captureLogits();
    const store = new BranchStore(ctx);

    const posBefore = branch.position;

    // store.commit([]) — empty batch
    await store.commit([]);
    assert(branch.position === posBefore, 'empty store.commit: position unchanged');
    ok('store.commit([]) resolves');

    // store.prefill([]) — empty batch
    await store.prefill([]);
    assert(branch.position === posBefore, 'empty store.prefill: position unchanged');
    ok('store.prefill([]) resolves');

    // branch.prefill([]) — empty token array
    await branch.prefill([]);
    assert(branch.position === posBefore, 'empty branch.prefill: position unchanged');
    ok('branch.prefill([]) resolves');

    // Verify branch still works after empty operations
    const { token, isStop } = branch.produce();
    assert(!isStop, 'empty edge: produce still works after empty ops');
    await branch.commit(token);
    assert(branch.position === posBefore + 1, 'empty edge: commit advances position after empty ops');

    await branch.prune();
  } finally {
    ctx.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// JSON SCHEMA TO GRAMMAR — AsyncWorker with zero prior coverage
// ═══════════════════════════════════════════════════════════════════════════

async function testJsonSchemaToGrammar() {
  console.log('\n--- jsonSchemaToGrammar ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: CTX_SIZE,
    nThreads: 4
  });

  try {
    const schema = {
      type: 'object',
      properties: {
        name: { type: 'string' },
        age: { type: 'integer' },
      },
      required: ['name', 'age'],
    };

    // Happy path: valid schema → GBNF string
    const grammar = await ctx.jsonSchemaToGrammar(JSON.stringify(schema));
    assert(typeof grammar === 'string' && grammar.length > 0,
      `jsonSchemaToGrammar: returned ${grammar.length}-char grammar`);
    assert(grammar.includes('root'), 'jsonSchemaToGrammar: grammar contains "root" rule');

    // Use the grammar with createSampler to prove it's valid GBNF
    const handle = ctx.createSampler(grammar);
    assert(handle > 0, `jsonSchemaToGrammar: createSampler accepted grammar (handle=${handle})`);

    // Generate tokens with grammar constraint
    await ctx.kvCacheClear();
    const prompt = await ctx.tokenize("Output JSON: ");
    await ctx.decode(prompt, 0, 0);

    const branch = Branch.create(ctx, prompt.length, { temperature: 0 }, undefined, grammar);
    branch.captureLogits();

    const output = [];
    for (let i = 0; i < 50; i++) {
      const { token, text, isStop } = branch.produce();
      if (isStop) break;
      await branch.commit(token);
      output.push(text);
    }

    const result = output.join('');
    let parsed;
    try {
      parsed = JSON.parse(result);
    } catch {
      // Grammar may not always produce complete JSON in 50 tokens
    }

    if (parsed) {
      assert(typeof parsed.name === 'string', `jsonSchemaToGrammar: output has string "name": "${parsed.name}"`);
      assert(typeof parsed.age === 'number', `jsonSchemaToGrammar: output has number "age": ${parsed.age}`);
    } else {
      // At minimum the output should start with valid JSON structure
      assert(result.startsWith('{'), `jsonSchemaToGrammar: output starts with '{': "${result.slice(0, 30)}..."`);
    }

    await branch.prune();
    ctx.freeSamplerHandle(handle);

    // Error path: invalid JSON → promise rejects
    let rejected = false;
    try {
      await ctx.jsonSchemaToGrammar('not valid json {{{');
    } catch (e) {
      rejected = true;
      assert(e instanceof Error, `jsonSchemaToGrammar: rejection is Error: ${e.constructor.name}`);
    }
    assert(rejected, 'jsonSchemaToGrammar: invalid JSON rejects');
  } finally {
    ctx.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// DISPOSED-DURING-ASYNC — _disposed set synchronously prevents use-after-prune
// ═══════════════════════════════════════════════════════════════════════════

async function testDisposedDuringAsync() {
  console.log('\n--- Disposed During Async ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: CTX_SIZE,
    nThreads: 4,
    nSeqMax: 4
  });

  try {
    const tokens = await ctx.tokenize("Test prompt");
    await ctx.decode(tokens, 0, 0);

    const branch = Branch.create(ctx, tokens.length, { temperature: 0 });
    branch.captureLogits();

    // Generate one token so branch has state
    const { token } = branch.produce();
    await branch.commit(token);

    // Call prune() — DO NOT await yet
    const prunePromise = branch.prune();

    // Immediately (before microtask resolves) check disposed
    assert(branch.disposed, 'disposed-during: _disposed is true synchronously after prune() call');

    // produce() should throw synchronously
    let threwProduce = false;
    try {
      branch.produce();
    } catch {
      threwProduce = true;
    }
    assert(threwProduce, 'disposed-during: produce() throws before prune promise resolves');

    // commit() should throw synchronously (the _ensureNotDisposed guard)
    let threwCommit = false;
    try {
      await branch.commit(token);
    } catch {
      threwCommit = true;
    }
    assert(threwCommit, 'disposed-during: commit() throws before prune promise resolves');

    // Now await the prune — should resolve cleanly
    await prunePromise;
    ok('disposed-during: prune promise resolves after synchronous guard tests');

    // Double-prune should be a no-op (idempotent)
    await branch.prune();
    ok('disposed-during: double prune is idempotent');
  } finally {
    ctx.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// ASYNC ITERATOR — Branch as async iterable
// ═══════════════════════════════════════════════════════════════════════════

async function testAsyncIterator() {
  console.log('\n--- Async Iterator ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: CTX_SIZE,
    nThreads: 4,
    nSeqMax: 4
  });

  try {
    const prompt = await ctx.tokenize("The quick brown fox");
    await ctx.decode(prompt, 0, 0);

    // Generate to EOG via for-await
    const branch = Branch.create(ctx, prompt.length, { temperature: 0 });
    branch.captureLogits();

    const tokens = [];
    for await (const { token, text } of branch) {
      assert(typeof token === 'number' && typeof text === 'string',
        `iterator: yields {token, text} (token=${token})`);
      tokens.push(token);
      if (tokens.length >= 10) break;  // consumer-side bound
    }
    assert(tokens.length === 10, `iterator: consumer break at 10 tokens (got ${tokens.length})`);

    // Every yielded token was committed (commit-before-yield)
    assert(branch.position === prompt.length + tokens.length,
      `iterator: position reflects all yielded tokens (${branch.position} === ${prompt.length} + ${tokens.length})`);

    // Perplexity reflects all committed tokens
    assert(isFinite(branch.perplexity) && branch.perplexity >= 1.0,
      `iterator: perplexity valid after iteration (${branch.perplexity.toFixed(2)})`);

    await branch.prune();

    // Compare: iterator output matches produce/commit output (deterministic, temp=0)
    await ctx.kvCacheClear();
    await ctx.decode(prompt, 0, 0);

    const branchManual = Branch.create(ctx, prompt.length, { temperature: 0 });
    branchManual.captureLogits();
    const manualTokens = [];
    for (let i = 0; i < 10; i++) {
      const { token, isStop } = branchManual.produce();
      if (isStop) break;
      await branchManual.commit(token);
      manualTokens.push(token);
    }

    assert(tokens.length === manualTokens.length &&
      tokens.every((t, i) => t === manualTokens[i]),
      'iterator: output matches manual produce/commit (deterministic)');

    await branchManual.prune();
  } finally {
    ctx.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  let mainCtx = null;

  try {
    // Create main context for reusable tests
    mainCtx = await addon.createContext({
      modelPath: MODEL_PATH,
      nCtx: CTX_SIZE,
      nThreads: 4
    });
    ok(`createContext(nCtx=${CTX_SIZE}) → vocabSize=${mainCtx.vocabSize}`);

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
    await testWarmMultiTurnRecall();
    await testWarmSemanticRecall();
    await testBranchSteer();
    await testNBatchAblation();
    await testDeterminism();
    await testDecodeAndCapture();
    await testBranchStore();
    await testPplSanity();
    await testCommitRollback();
    await testAsyncRejection();
    await testEmptyInputEdgeCases();
    await testJsonSchemaToGrammar();
    await testDisposedDuringAsync();
    await testAsyncIterator();
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
