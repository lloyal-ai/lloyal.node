# lloyal.node

**Research-grade edge inference for Node.js**

Inference with forkable state — KV cache, grammar, metrics all clone atomically. Entropy and surprisal mid-generation. Multi-sequence parallel exploration. The control surface llama.cpp exposes, in TypeScript.

```bash
npm install @lloyal-labs/lloyal.node
```

Prebuilt binaries for 13 platforms:

| Platform | Arch  | Acceleration        |
| -------- | ----- | ------------------- |
| macOS    | arm64 | Metal               |
| macOS    | x64   | CPU                 |
| Linux    | x64   | CPU / CUDA / Vulkan |
| Linux    | arm64 | CPU / CUDA / Vulkan |
| Windows  | x64   | CPU / CUDA / Vulkan |
| Windows  | arm64 | CPU / Vulkan        |

GPU selection happens at runtime, not install time. See [DISTRIBUTION.md](./docs/DISTRIBUTION.md) for details.

---

## Forkable Inference

KV cache, grammar parser, and perplexity trackers all live behind handles. Handles clone atomically. This is the primitive that makes branching possible.

Consider speculative decoding. You draft tokens speculatively, then need to fork at the draft boundary to explore continuations. Each fork carries independent state. When branches collapse, the rejected ones disappear cleanly.

```typescript
import { createContext } from '@lloyal-labs/lloyal.node';

const ctx = await createContext({ modelPath, nCtx: 4096, nSeqMax: 4 });

// Prefill prompt on seq 0
const prompt = await ctx.tokenize(text);
await ctx.decode(prompt, 0, 0);

// Draft 4 speculative tokens
const drafts: number[] = [];
let pos = prompt.length;

for (let i = 0; i < 4; i++) {
  const token = ctx.sample({ temperature: 0.0 });
  drafts.push(token);
  await ctx.decode([token], pos++, 0);
}

// Fork to seq 1 for main model pass
ctx.kvSeqCopy(0, 1);

// ... run main model on seq 1, find accepted prefix length ...

// Collapse: keep seq 1, discard seq 0's speculative tail
ctx.kvSeqKeep(1);
```

No state leaks. No manual cleanup. Fork, explore, collapse.

---

## Best-of-N Sampling

Generate multiple completions. Pick the best. Simple strategy, surprisingly effective — but only if each completion is truly independent.

Multi-sequence support lets you run N completions in parallel, each with its own KV cache state. Perplexity trackers clone atomically, so each branch accumulates its own quality score.

```typescript
const ctx = await createContext({ modelPath, nCtx: 4096, nSeqMax: 8 });

const prompt = await ctx.tokenize(text);
await ctx.decode(prompt, 0, 0);

// Generate N independent completions
const N = 4;
const completions = await Promise.all(
  Array.from({ length: N }, async (_, i) => {
    const seqId = i + 1;
    ctx.kvSeqCopy(0, seqId);  // Fork KV cache

    const tracker = ctx.createPerplexityTracker();
    const tokens: number[] = [];
    let pos = prompt.length;

    while (tokens.length < 100) {
      const token = ctx.sample({ temperature: 0.9 });
      if (ctx.isStopToken(token)) break;

      const surprisal = ctx.modelSurprisal(token);
      ctx.addSurprisal(tracker, surprisal);

      tokens.push(token);
      await ctx.decode([token], pos++, seqId);
    }

    const ppl = ctx.getPerplexity(tracker);
    ctx.freePerplexityTracker(tracker);

    return { seqId, tokens, ppl };
  })
);

// Pick lowest perplexity
const best = completions.reduce((a, b) => (a.ppl < b.ppl ? a : b));
ctx.kvSeqKeep(best.seqId);

console.log(`Best (PPL ${best.ppl.toFixed(2)}):`, await ctx.detokenize(best.tokens));
```

Four completions. Four independent KV states. Four independent perplexity scores. One winner.

---

## Context Window Dynamics

Context windows have limits. Documents don't. `clearAndReseed()` enables infinite streaming via BlinkKV — clear the cache, re-decode sinks and tail at contiguous positions, continue. No attention surgery required.

Perplexity tracking lets you measure quality as you stream. Does your eviction strategy degrade output? The instrumentation tells you.

```typescript
const nCtx = 512;
const ctx = await createContext({ modelPath, nCtx });

const allTokens = await ctx.tokenize(longDocument);
const sinks = allTokens.slice(0, 4);  // First 4 tokens of document
const tailSize = nCtx - sinks.length - 8;  // Leave headroom

const tracker = ctx.createPerplexityTracker();
let cachePos = 0;

for (let t = 0; t < allTokens.length; t++) {
  const token = allTokens[t];

  // Measure surprisal before decoding
  if (cachePos > 0) {
    const surprisal = ctx.modelSurprisal(token);
    ctx.addSurprisal(tracker, surprisal);
  }

  await ctx.decode([token], cachePos++, 0);

  // Cache full? Reseed at boundary
  if (cachePos >= nCtx) {
    const tail = allTokens.slice(t - tailSize + 1, t + 1);
    await ctx.clearAndReseed(sinks, tail);
    cachePos = sinks.length + tail.length;

    console.log(`Reseeded at token ${t}. PPL so far: ${ctx.getPerplexity(tracker)}`);
  }
}

console.log(`Final perplexity: ${ctx.getPerplexity(tracker)}`);
ctx.freePerplexityTracker(tracker);
```

This is how BlinkKV measured that position contiguity matters more than attention sinks — PPL 9.7 with proper reseeding vs 280 with naive eviction. The same primitives enable whatever you want to study.

---

## Uncertainty as Control Flow

The model knows when it's guessing. Entropy measures how flat the distribution is. High entropy means the model sees many plausible continuations — it's uncertain. Low entropy means one token dominates — it's confident.

Use this mid-generation to change behavior:

```typescript
const ctx = await createContext({ modelPath, nCtx: 4096 });

const prompt = await ctx.tokenize(text);
await ctx.decode(prompt, 0, 0);

let pos = prompt.length;
const output: number[] = [];

while (output.length < 200) {
  const entropy = ctx.modelEntropy('bits');

  if (entropy > 6.0) {
    // Model is guessing — inject retrieved context
    const retrieved = await retrieveRelevant(ctx.detokenize(output.slice(-50)));
    const retrievedTokens = await ctx.tokenize(`\n[Context: ${retrieved}]\n`);
    await ctx.decode(retrievedTokens, pos, 0);
    pos += retrievedTokens.length;
    continue;
  }

  if (entropy < 1.0) {
    // Distribution collapsed — might be stuck in a loop
    const token = ctx.sample({ temperature: 1.2 });
    output.push(token);
    await ctx.decode([token], pos++, 0);
    continue;
  }

  // Normal sampling
  const token = ctx.sample({ temperature: 0.7 });
  if (ctx.isStopToken(token)) break;

  output.push(token);
  await ctx.decode([token], pos++, 0);
}
```

Entropy isn't just a metric — it's a control signal. Retrieval triggers, temperature adaptation, early stopping. The model tells you what it needs.

---

## Persistent Sampling State

Repeat penalties need history. The sampler must know what tokens appeared earlier to penalize their recurrence. But when you reseed the KV cache during infinite streaming, that history could vanish.

The sampler chain persists independently. It tracks all accepted tokens across the entire generation — before and after `clearAndReseed()`. Penalties stay coherent even as the KV cache resets.

```typescript
const ctx = await createContext({ modelPath, nCtx: 2048 });

const prompt = await ctx.tokenize(text);
await ctx.decode(prompt, 0, 0);

// First generation phase
let pos = prompt.length;
const allTokens: number[] = [];

while (allTokens.length < 500) {
  // Sampler chain persists penalty history across all calls
  const token = ctx.sample({
    temperature: 0.8,
    repeatPenalty: 1.1,
    repeatLastN: 256,  // Look back 256 tokens in penalty history
  });

  if (ctx.isStopToken(token)) break;
  allTokens.push(token);
  await ctx.decode([token], pos++, 0);

  // KV cache filled? Reseed.
  if (pos >= 2000) {
    const sinks = prompt.slice(0, 4);
    const tail = allTokens.slice(-256);
    await ctx.clearAndReseed(sinks, tail);
    pos = sinks.length + tail.length;

    // Penalty history survives — sampler still knows about
    // ALL tokens generated, not just the 256 in the tail
  }
}
```

The penalty sampler accumulates history via `accept()` internally. Parameters rebuild the chain only when they change — same temperature and penalties reuse the existing state. Different parameters trigger a rebuild, but new chains inherit no history.

This is why repetition penalties work across context boundaries. The KV cache knows attention. The sampler knows history. Both are necessary for quality.

---

## Grammar-Constrained Branching

JSON schema constraints narrow the token space. But different valid continuations exist — `{"name":` could continue with any string. When you're exploring alternatives, the grammar state needs to fork with the KV cache.

Handle-based grammar samplers make this possible:

```typescript
const ctx = await createContext({ modelPath, nCtx: 2048, nSeqMax: 4 });

const schema = {
  type: 'object',
  properties: {
    name: { type: 'string' },
    age: { type: 'number' },
    city: { enum: ['NYC', 'LA', 'Chicago'] },
  },
  required: ['name', 'age', 'city'],
};

const grammar = ctx.jsonSchemaToGrammar(JSON.stringify(schema));
const grammarHandle = ctx.createSampler(grammar);

const prompt = await ctx.tokenize('Generate a person:\n');
await ctx.decode(prompt, 0, 0);

let pos = prompt.length;
const partial: number[] = [];

// Generate until we approach a choice point
while (partial.length < 30) {
  const buffer = ctx.getTokenScores();
  ctx.applySampler(grammarHandle, buffer);

  const token = ctx.sample({ temperature: 0.7 });
  ctx.acceptSamplerToken(grammarHandle, token);
  partial.push(token);
  await ctx.decode([token], pos++, 0);

  if (ctx.isStopToken(token)) break;
}

// Fork: save state and explore different continuations
const kvSnapshot = await ctx.kvCacheSave(0);
const grammarClone = ctx.cloneSampler(grammarHandle);

for (const city of ['NYC', 'LA', 'Chicago']) {
  // Restore to decision point
  await ctx.kvCacheLoad(0, kvSnapshot);
  const branchGrammar = ctx.cloneSampler(grammarClone);

  const cityTokens = await ctx.tokenize(`"${city}"`);
  let branchPos = pos;

  for (const t of cityTokens) {
    ctx.acceptSamplerToken(branchGrammar, t);
    await ctx.decode([t], branchPos++, 0);
  }

  const text = await ctx.detokenize([...partial, ...cityTokens]);
  console.log(`Branch ${city}:`, text);

  ctx.freeSamplerHandle(branchGrammar);
}

ctx.freeSamplerHandle(grammarHandle);
ctx.freeSamplerHandle(grammarClone);
```

Grammar narrows the space. Forking explores within that space. Both states travel together.

---

## Who This Is For

You're implementing speculative decoding and need to fork KV state at the draft boundary, run multiple continuations, then collapse to the accepted prefix.

You're building best-of-N sampling and need independent perplexity tracking per completion — each branch accumulates its own quality score.

You're studying what happens inside the context window — how attention patterns shift when you evict tokens, whether sinks matter, what position encodings actually do. `clearAndReseed()`, `kvCacheRemove()`, per-token surprisal. The instrumentation is here.

You're adding entropy-triggered retrieval and need `modelEntropy()` mid-generation to detect when the model is guessing.

You're constraining output to a JSON schema and need the grammar state to fork with KV — because each branch explores different valid completions.

You're streaming beyond the context window with `clearAndReseed()` and need repeat penalties to remember what happened before the reseed — because the sampler maintains its own history.

You need the decode loop, not a completion endpoint.

---

## API Reference

### Context Creation

```typescript
const ctx = await createContext({
  modelPath: string,       // Path to .gguf file (required)
  nCtx?: number,           // Context size (default: 2048)
  nThreads?: number,       // CPU threads (default: 4)
  embeddings?: boolean,    // Enable embedding mode (default: false)
  poolingType?: number,    // 0=NONE, 1=MEAN, 2=CLS, 3=LAST
  nSeqMax?: number,        // Max parallel sequences (default: 1)
});
```

### Core Methods

| Method                        | Returns             | Description                       |
| ----------------------------- | ------------------- | --------------------------------- |
| `tokenize(text)`              | `Promise<number[]>` | Text → token IDs                  |
| `detokenize(tokens)`          | `Promise<string>`   | Token IDs → text                  |
| `tokenToText(token)`          | `string`            | Single token → text (streaming)   |
| `decode(tokens, pos, seqId?)` | `Promise<void>`     | Forward pass, updates KV cache    |
| `sample(params?)`             | `number`            | Sample next token                 |
| `isStopToken(token)`          | `boolean`           | Check for EOS token               |
| `getLogits()`                 | `Float32Array`      | Raw logits (zero-copy, read-only) |
| `getTokenScores()`            | `Buffer`            | Token scores (zero-copy, mutable) |

### KV Cache

| Method                             | Returns           | Description                    |
| ---------------------------------- | ----------------- | ------------------------------ |
| `kvCacheSize(seqId?)`              | `number`          | Tokens in cache                |
| `kvCacheClear()`                   | `Promise<void>`   | Clear all sequences            |
| `kvCacheRemove(seqId, start, end)` | `Promise<void>`   | Remove token range             |
| `kvCacheSave(seqId?)`              | `Promise<Buffer>` | Snapshot state                 |
| `kvCacheLoad(seqId, state)`        | `Promise<void>`   | Restore state                  |
| `kvSeqCopy(src, dst)`              | `void`            | Copy sequence (tag copy, O(1)) |
| `kvSeqKeep(seqId)`                 | `void`            | Keep only one sequence         |
| `clearAndReseed(sinks, tail)`      | `Promise<void>`   | BlinkKV pattern                |

### Grammar (Handle-Based, Forkable)

| Method                           | Returns  | Description                  |
| -------------------------------- | -------- | ---------------------------- |
| `jsonSchemaToGrammar(schema)`    | `string` | Schema → GBNF                |
| `createSampler(grammarStr)`      | `number` | Create grammar handle        |
| `cloneSampler(handle)`           | `number` | Clone grammar state          |
| `applySampler(handle, buffer)`   | `void`   | Apply constraints to logits  |
| `acceptSamplerToken(handle, id)` | `void`   | Advance parser state         |
| `freeSamplerHandle(handle)`      | `void`   | Release grammar handle       |

### Grammar (Internal State)

| Method                  | Returns | Description                     |
| ----------------------- | ------- | ------------------------------- |
| `initGrammar(gbnf)`     | `void`  | Initialize internal parser      |
| `applyGrammar(buffer)`  | `void`  | Apply constraints to logits     |
| `acceptToken(token)`    | `void`  | Advance internal parser         |
| `resetGrammar()`        | `void`  | Reset to initial state          |
| `freeGrammar()`         | `void`  | Release internal grammar        |

### Metrics

| Method                           | Returns         | Description                      |
| -------------------------------- | --------------- | -------------------------------- |
| `modelEntropy(base?)`            | `number`        | Distribution entropy (bits/nats) |
| `modelSurprisal(token, base?)`   | `number`        | Token surprisal                  |
| `createPerplexityTracker()`      | `TrackerHandle` | Create tracker (forkable)        |
| `clonePerplexityTracker(handle)` | `TrackerHandle` | Clone tracker state              |
| `addSurprisal(handle, value)`    | `void`          | Add to tracker                   |
| `getPerplexity(handle)`          | `number`        | Get current PPL                  |
| `freePerplexityTracker(handle)`  | `void`          | Release tracker                  |

### Embeddings

| Method                      | Returns         | Description                 |
| --------------------------- | --------------- | --------------------------- |
| `encode(tokens)`            | `Promise<void>` | Forward pass for embeddings |
| `getEmbeddings(normalize?)` | `Float32Array`  | Extract embedding vector    |
| `getEmbeddingDimension()`   | `number`        | Vector dimension            |

### Lifecycle

| Method      | Description                          |
| ----------- | ------------------------------------ |
| `dispose()` | Free native resources (**required**) |

---

## Examples

Working examples in [`examples/`](./examples/):

| Example | Description |
|---------|-------------|
| [`chat/`](./examples/chat/) | Interactive chat with streaming output |
| [`embed/`](./examples/embed/) | Text embeddings |
| [`speculative/`](./examples/speculative/) | Forkable KV state for speculative decoding |
| [`best-of-n/`](./examples/best-of-n/) | Parallel completions with perplexity selection |
| [`streaming/`](./examples/streaming/) | Infinite context via BlinkKV |
| [`entropy/`](./examples/entropy/) | Entropy-triggered control flow |
| [`grammar/`](./examples/grammar/) | JSON schema constraints with forkable grammar state |

```bash
node examples/chat/chat.mjs /path/to/model.gguf
```

---

## Ecosystem

| Package                                                 | Runtime      | Description                       |
| ------------------------------------------------------- | ------------ | --------------------------------- |
| [liblloyal](https://github.com/lloyal-ai/liblloyal)     | C++          | Header-only inference kernel      |
| **lloyal.node**                                         | Node.js      | This package                      |
| [nitro-llama](https://github.com/lloyal-ai/nitro-llama) | React Native | Mobile bindings via Nitro Modules |
| [tsampler](https://github.com/lloyal-ai/tsampler)       | TypeScript   | Reference sampler implementation  |

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and release process.

## License

Apache 2.0 — See [LICENSE](./LICENSE) for details.
