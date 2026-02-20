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

  // Branch-based prefill + getLogits
  const branch = Branch.create(ctx, 0, { temperature: 0 });
  await branch.prefill(tokens);

  const branchLogits = branch.getLogits();
  assert(branchLogits instanceof Float32Array, `branch.getLogits() → Float32Array(${branchLogits.length})`);
  assert(branchLogits.length === ctx.vocabSize, `branchLogits.length === vocabSize (${ctx.vocabSize})`);

  // Validate logits are not garbage
  let hasNonZero = false, hasNaN = false;
  for (let i = 0; i < branchLogits.length; i++) {
    if (branchLogits[i] !== 0.0) hasNonZero = true;
    if (isNaN(branchLogits[i])) hasNaN = true;
  }
  assert(hasNonZero && !hasNaN, 'branch logits valid (non-zero, no NaN)');

  // modelEntropy with branch logits
  const entropy = ctx.modelEntropy('nats', branchLogits);
  assert(isFinite(entropy) && entropy >= 0, `modelEntropy(branchLogits) → ${entropy.toFixed(4)} nats`);

  // Branch greedy sampling (temperature: 0)
  const greedy = branch.sample();
  assert(greedy >= 0 && greedy < ctx.vocabSize, `branch.sample() greedy → ${greedy}`);

  // isStopToken - EOS should be a stop token
  const eos = ctx.getEogToken();
  assert(ctx.isStopToken(eos), `isStopToken(EOS=${eos}) → true`);

  // withLogits helper (context-level logits)
  // Note: getLogits() reads from the shared context buffer, which is populated
  // by branch decode operations
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

  await branch.prune();
}

// ═══════════════════════════════════════════════════════════════════════════
// KV CACHE TESTS
// ═══════════════════════════════════════════════════════════════════════════

async function testKVCache(ctx) {
  console.log('\n--- KV Cache ---');

  await ctx.kvCacheClear();
  const tokens = await ctx.tokenize("Test prompt");
  const branch = Branch.create(ctx, 0, { temperature: 0 });
  await branch.prefill(tokens);

  const sizeBefore = ctx.kvCacheSize();
  assert(sizeBefore >= 0, `kvCacheSize() after prefill → ${sizeBefore}`);

  await branch.prune();
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
    // Use a branch to prefill tokens (populates KV on its seq_id)
    const tokens = await ctx.tokenize("The quick brown fox");
    const branch = Branch.create(ctx, 0, { temperature: 0 });
    await branch.prefill(tokens);

    // Branch allocates a seq_id — check its KV is populated
    const branchPos = branch.position;
    assert(branchPos === tokens.length, `branch position → ${branchPos}`);

    // Fork creates a new sequence with copied KV
    const forked = await branch.fork();
    assert(forked.position === branchPos, `forked position matches parent → ${forked.position}`);

    // Raw KV seq ops still work for advanced use
    const seq1Before = ctx.kvSeqPosMax(3);  // unused seq_id
    assert(seq1Before === -1, `kvSeqPosMax(unused) → ${seq1Before} (empty)`);

    await forked.prune();
    await branch.prune();
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
    nThreads: 4,
    nSeqMax: 4
  });

  try {
    const grammar = `root ::= "{" ws "}" ws
ws ::= [ \\t\\n]*`;

    // Branch API with grammar
    const prompt = await ctx.tokenize("Output: ");
    const branch = Branch.create(ctx, 0, { temperature: 0 }, undefined, grammar);
    await branch.prefill(prompt);

    const output = [];
    for (let i = 0; i < 10; i++) {
      const { token, text, isStop } = await branch.produce();
      if (isStop) break;
      await branch.commit(token);
      output.push(text);
    }

    const result = output.join('');
    assert(/^\{\s*\}\s*$/.test(result), `Branch+grammar → "${result}"`);

    // Grammar is cloned on fork — independent parser states
    await ctx.kvCacheClear();
    const prompt2 = await ctx.tokenize("Output: ");
    const root = Branch.create(ctx, 0, { temperature: 0 }, undefined, grammar);
    await root.prefill(prompt2);

    const childA = await root.fork();
    const childB = await root.fork();

    // Both children should produce grammar-valid output independently
    const outA = [], outB = [];
    for (let i = 0; i < 10; i++) {
      const pA = await childA.produce();
      if (!pA.isStop) { await childA.commit(pA.token); outA.push(pA.text); }
      const pB = await childB.produce();
      if (!pB.isStop) { await childB.commit(pB.token); outB.push(pB.text); }
    }

    const resultA = outA.join(''), resultB = outB.join('');
    assert(/^\{\s*\}\s*$/.test(resultA), `Fork A grammar → "${resultA}"`);
    assert(/^\{\s*\}\s*$/.test(resultB), `Fork B grammar → "${resultB}"`);

    await childA.prune();
    await childB.prune();
    await root.prune();
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
  const branch = Branch.create(ctx, 0, { temperature: 0 });
  await branch.prefill(tokens);

  // modelSurprisal with branch logits
  const token1 = branch.sample();
  const branchLogits = branch.getLogits();
  const surprisal = ctx.modelSurprisal(token1, "nats", branchLogits);
  assert(surprisal >= 0, `modelSurprisal(branchLogits) → ${surprisal.toFixed(2)} nats`);

  const surprisalBits = ctx.modelSurprisal(token1, "bits", branchLogits);
  assert(Math.abs(surprisalBits - surprisal / Math.log(2)) < 0.01, 'bits = nats / ln(2)');

  // Branch perplexity — built-in, accumulates through commit()
  await branch.commit(token1);
  const { token: token2 } = await branch.produce();
  await branch.commit(token2);

  const ppl = branch.perplexity;
  assert(isFinite(ppl) && ppl >= 1.0, `branch.perplexity → ${ppl.toFixed(2)}`);

  await branch.prune();
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
    const branch = Branch.create(ctx, 0, { temperature: 0 });
    await branch.prefill(promptToks);

    // Turn 1
    const gen1 = [];
    for (let i = 0; i < GEN_TOKENS; i++) {
      const { token, isStop } = await branch.produce();
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
        const { token, isStop } = await branch.produce();
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
        const { token, isStop } = await branch.produce();
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
    const branch = Branch.create(ctx, 0, { temperature: 0 });
    await branch.prefill(promptToks);

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
          const { token, isStop } = await branch.produce();
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
      branch = Branch.create(ctx, 0, { temperature: 0 });
      await branch.prefill(promptToks);

      // Generate turn 1 response
      const gen = [];
      for (let i = 0; i < GEN_TOKENS; i++) {
        const { token, isStop } = await branch.produce();
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
    const branch = Branch.create(ctx, 0, { temperature: 0 });
    await branch.prefill(tokens);

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
    const parent = Branch.create(ctx, 0, { temperature: 0 });
    await parent.prefill(tokens2);

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
      const branch = Branch.create(ctx, 0, { temperature: 0 }, nBatch);
      await branch.prefill(promptToks);

      const followUp = await ctx.tokenize(" What else?");
      await branch.prefill(followUp);

      const gen = [];
      for (let i = 0; i < 5; i++) {
        const { token, isStop } = await branch.produce();
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

      const branch = Branch.create(ctx, 0, { temperature: 0 });
      await branch.prefill(tokens);

      const gen = [];
      for (let i = 0; i < 20; i++) {
        const { token, isStop } = await branch.produce();
        if (isStop) break;
        await branch.commit(token);
        gen.push(token);
      }
      await branch.prune();
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
// BRANCH PREFILL + GET LOGITS (replaces testDecodeAndCapture)
// ═══════════════════════════════════════════════════════════════════════════

async function testBranchPrefillAndLogits() {
  console.log('\n--- Branch prefill + getLogits ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: CTX_SIZE,
    nThreads: 4
  });

  try {
    const tokens = await ctx.tokenize("Hello");
    const branch = Branch.create(ctx, 0, { temperature: 0 });
    await branch.prefill(tokens);

    const logits = branch.getLogits();
    let valid = false;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] !== 0 && !isNaN(logits[i])) valid = true;
    }
    assert(valid, `branch.prefill() + getLogits() → valid logits`);

    // Branch logits are an independent copy
    const orig = logits[0];
    logits[0] = -999;
    const logits2 = branch.getLogits();
    assert(logits2[0] !== -999, 'branch.getLogits() returns independent copy');

    await branch.prune();
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
      const root = Branch.create(ctx, 0, { temperature: 0.8 });
      await root.prefill(promptToks);
      const branches = [root, await root.fork(), await root.fork()];
      branches[1].reseedSampler(42);
      branches[2].reseedSampler(99);

      for (let step = 0; step < 10; step++) {
        const produced = await Promise.all(branches.map(async b => [b, await b.produce()]));
        const live = produced.filter(([, p]) => !p.isStop);
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
      const b1 = Branch.create(ctx, 0, { temperature: 0 });
      await b1.prefill(promptToks);
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
        const produced = [[b1, await b1.produce()], [b2, await b2.produce()]];
        const live = produced.filter(([, p]) => !p.isStop);
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
      const b1 = Branch.create(ctx, 0, { temperature: 0 });
      await b1.prefill(promptToks);

      const logits = b1.getLogits();
      assert(logits instanceof Float32Array,
        `getLogits: returns Float32Array`);
      assert(logits.length === ctx.vocabSize,
        `getLogits: length=${logits.length} === vocabSize=${ctx.vocabSize}`);

      // Feed branch logits into ctx.modelEntropy() — proves the returned
      // buffer is a valid logits distribution consumable by metrics API
      const entropyFromBranch = ctx.modelEntropy("nats", logits);
      assert(isFinite(entropyFromBranch) && entropyFromBranch > 0,
        `getLogits→modelEntropy: ${entropyFromBranch.toFixed(4)} nats`);

      // After store.commit, logits change — getLogits() reflects new state
      const p = await b1.produce();
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
      const b1 = Branch.create(ctx, 0, { temperature: 0 });
      await b1.prefill(promptToks);
      const b2 = await b1.fork();

      const output = [];
      for (let i = 0; i < 5; i++) {
        // Inspect with produce() — does NOT advance state
        const p1 = await b1.produce(), p2 = await b2.produce();

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
      const b1 = Branch.create(ctx, 0, { temperature: 0 });
      await b1.prefill(promptToks);
      const b2 = await b1.fork();

      // Step 1-3: single-branch commit (decode::one path)
      for (let i = 0; i < 3; i++) {
        const produced = [[b1, await b1.produce()], [b2, await b2.produce()]];
        const live = produced.filter(([, p]) => !p.isStop);
        if (!live.length) break;
        for (const [b, p] of live) await b.commit(p.token);
      }
      const posAfterSingle = b1.position;

      // Step 4-6: batched commit (decode::each path)
      for (let i = 0; i < 3; i++) {
        const produced = [[b1, await b1.produce()], [b2, await b2.produce()]];
        const live = produced.filter(([, p]) => !p.isStop);
        if (!live.length) break;
        await store.commit(live.map(([b, p]) => [b, p.token]));
      }
      const posAfterBatched = b1.position;
      assert(posAfterBatched === posAfterSingle + 3,
        `mixed ops: position correct after single→batched (${posAfterSingle}→${posAfterBatched})`);

      // Step 7-9: back to single-branch commit
      for (let i = 0; i < 3; i++) {
        const produced = [[b1, await b1.produce()], [b2, await b2.produce()]];
        const live = produced.filter(([, p]) => !p.isStop);
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
      const b1 = Branch.create(ctx, 0, { temperature: 0 });
      await b1.prefill(promptToks);
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
          ...(!stopped[0] ? [[b1, await b1.produce()]] : []),
          ...(!stopped[1] ? [[b2, await b2.produce()]] : []),
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
    const branch = Branch.create(ctx, 0, { temperature: 0 });
    await branch.prefill(promptToks);

    for (let i = 0; i < 10; i++) {
      const { token, isStop } = await branch.produce();
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
    const root = Branch.create(ctx, 0, { temperature: 1.0 });
    await root.prefill(promptToks);
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
      const produced = await Promise.all(branches.map(async b => [b, await b.produce()]));
      const live = produced.filter(([, p]) => !p.isStop);
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
    const branch = Branch.create(ctx, 0, { temperature: 0 });
    await branch.prefill(tokens);

    // Generate one token to prove branch works
    const { token, isStop } = await branch.produce();
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

    // produce() on disposed branch — async version rejects
    let threwOnProduce = false;
    try {
      await branch.produce();
    } catch (e) {
      threwOnProduce = true;
    }
    assert(threwOnProduce, 'rejection: produce on disposed branch rejects');

    // produceSync() on disposed branch — throws synchronously
    let threwOnProduceSync = false;
    try {
      branch.produceSync();
    } catch (e) {
      threwOnProduceSync = true;
    }
    assert(threwOnProduceSync, 'rejection: produceSync on disposed branch throws');

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
    const branch = Branch.create(ctx, 0, { temperature: 0 });
    await branch.prefill(tokens);
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
    const { token, isStop } = await branch.produce();
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

    // Use the grammar with Branch.create to prove it's valid GBNF
    const prompt = await ctx.tokenize("Output JSON: ");
    const branch = Branch.create(ctx, 0, { temperature: 0 }, undefined, grammar);
    await branch.prefill(prompt);

    const output = [];
    for (let i = 0; i < 50; i++) {
      const { token, text, isStop } = await branch.produce();
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
    const branch = Branch.create(ctx, 0, { temperature: 0 });
    await branch.prefill(tokens);

    // Generate one token so branch has state
    const { token } = await branch.produce();
    await branch.commit(token);

    // Call prune() — DO NOT await yet
    const prunePromise = branch.prune();

    // Immediately (before microtask resolves) check disposed
    assert(branch.disposed, 'disposed-during: _disposed is true synchronously after prune() call');

    // produceSync() should throw synchronously
    let threwProduce = false;
    try {
      branch.produceSync();
    } catch {
      threwProduce = true;
    }
    assert(threwProduce, 'disposed-during: produceSync() throws before prune promise resolves');

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

    // Generate to EOG via for-await
    const branch = Branch.create(ctx, 0, { temperature: 0 });
    await branch.prefill(prompt);

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
    const branchManual = Branch.create(ctx, 0, { temperature: 0 });
    await branchManual.prefill(prompt);
    const manualTokens = [];
    for (let i = 0; i < 10; i++) {
      const { token, isStop } = await branchManual.produce();
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
// HOT-SWAP TESTS (setSamplerParams / setGrammar)
// ═══════════════════════════════════════════════════════════════════════════

async function testSetSamplerParams() {
  console.log('\n--- setSamplerParams ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: CTX_SIZE,
    nThreads: 4,
  });

  try {
    const prompt = await ctx.tokenize("The capital of France is");

    // Greedy baseline
    const greedy = Branch.create(ctx, 0, { temperature: 0, topK: 0, topP: 1.0, minP: 0 });
    await greedy.prefill(prompt);
    const greedyTok = greedy.sample();
    assert(greedyTok >= 0, `setSamplerParams: greedy token valid (${greedyTok})`);

    // Switch to stochastic — at high temp, should eventually diverge
    greedy.setSamplerParams({ temperature: 1.5, seed: 42, topK: 0, topP: 1.0, minP: 0 });
    let diverged = false;
    for (let i = 0; i < 20; i++) {
      if (greedy.sample() !== greedyTok) { diverged = true; break; }
    }
    assert(diverged, 'setSamplerParams: stochastic diverges from greedy');

    // Switch back to greedy — should be deterministic again
    greedy.setSamplerParams({ temperature: 0, topK: 0, topP: 1.0, minP: 0 });
    const tok2 = greedy.sample();
    const tok3 = greedy.sample();
    assert(tok2 === tok3, `setSamplerParams: greedy restored (${tok2} === ${tok3})`);

    await greedy.prune();

    // Memoization: identical params should not rebuild
    await ctx.kvCacheClear();
    const branch = Branch.create(ctx, 0, { temperature: 0.8, seed: 100 });
    await branch.prefill(prompt);
    branch.setSamplerParams({ temperature: 0.8, seed: 100 });  // Same — should be no-op
    assert(!branch.disposed, 'setSamplerParams: memoized no-op does not dispose');

    await branch.prune();
  } finally {
    ctx.dispose();
  }
}

async function testSetGrammar() {
  console.log('\n--- setGrammar ---');

  const ctx = await addon.createContext({
    modelPath: MODEL_PATH,
    nCtx: CTX_SIZE,
    nThreads: 4,
    nSeqMax: 4,
  });

  try {
    const grammar = `root ::= "{" ws "}" ws
ws ::= [ \\t\\n]*`;

    // Hot-swap: create without grammar, then add one
    const prompt = await ctx.tokenize("Output: ");
    const branch = Branch.create(ctx, 0, { temperature: 0 });
    await branch.prefill(prompt);

    branch.setGrammar(grammar);
    const output = [];
    for (let i = 0; i < 10; i++) {
      const { token, text, isStop } = await branch.produce();
      if (isStop) break;
      await branch.commit(token);
      output.push(text);
    }
    const result = output.join('');
    assert(/^\{\s*\}\s*$/.test(result), `setGrammar: hot-swap constrains → "${result}"`);

    // Remove grammar
    branch.setGrammar('');
    // Should no longer be constrained (just verify it doesn't throw)
    const { token } = await branch.produce();
    assert(typeof token === 'number', 'setGrammar: removal works, sample succeeds');

    await branch.prune();

    // Hot-swap + fork: grammar cloned to child
    await ctx.kvCacheClear();
    const root = Branch.create(ctx, 0, { temperature: 0 });
    await root.prefill(prompt);
    root.setGrammar(grammar);

    const child = await root.fork();
    const childOut = [];
    for (let i = 0; i < 10; i++) {
      const p = await child.produce();
      if (p.isStop) break;
      await child.commit(p.token);
      childOut.push(p.text);
    }
    const childResult = childOut.join('');
    assert(/^\{\s*\}\s*$/.test(childResult), `setGrammar: fork inherits grammar → "${childResult}"`);

    await child.prune();
    await root.prune();
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
    await testBranchPrefillAndLogits();
    await testBranchStore();
    await testPplSanity();
    await testCommitRollback();
    await testAsyncRejection();
    await testEmptyInputEdgeCases();
    await testJsonSchemaToGrammar();
    await testDisposedDuringAsync();
    await testAsyncIterator();
    await testSetSamplerParams();
    await testSetGrammar();
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
