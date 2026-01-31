# lloyal.node

**Advanced edge inference for Node.js**

A llama.cpp control surface in TypeScript with atomic inference state forking. Real time rolling perplexity/entropy/surprisal and multi-sequence parallel exploration primitives.

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

GPU selection happens at runtime, not install time. See [distribution.md](docs/distribution.md) for details.

---

## Examples

Working examples demonstrate each capability:

| Example                                   | What It Demonstrates                                                          |
| ----------------------------------------- | ----------------------------------------------------------------------------- |
| [`best-of-n/`](./examples/best-of-n/)     | Branch API parallel generation, PPL selection, fork/produce/commit            |
| [`speculative/`](./examples/speculative/) | Branch API fork/prune, draft/verify/accept/reject, bonus token sampling       |
| [`entropy/`](./examples/entropy/)         | Entropy Decision Tree — `modelEntropy()` mid-generation as control signal     |
| [`grammar/`](./examples/grammar/)         | Pull loop with generators, JSON schema constraints, KV + grammar branching    |
| [`streaming/`](./examples/streaming/)     | Infinite context via BlinkKV, `clearAndReseed`, perplexity tracking           |
| [`chat/`](./examples/chat/)               | Interactive streaming chat                                                    |
| [`embed/`](./examples/embed/)             | Text embeddings extraction                                                    |

```bash
node examples/best-of-n/best-of-n.mjs
node examples/speculative/speculative.mjs
node examples/entropy/entropy.mjs
node examples/grammar/grammar.mjs
```

Each example has a README explaining the pattern in depth.

---

## Core Patterns

### Branch API

`Branch` is the primary API for parallel generation. Each branch owns a KV cache sequence, sampler chain, logits snapshot, and perplexity tracker. Fork a branch to explore alternatives, compare by perplexity, prune losers.

```javascript
import { createContext, Branch } from '@lloyal-labs/lloyal.node';

const ctx = await createContext({ modelPath: './model.gguf', nSeqMax: 8 });
const tokens = await ctx.tokenize('Once upon a time');
await ctx.decode(tokens, 0, 0);

// Create root branch, capture logits from prefill
const root = Branch.create(ctx, 0, tokens.length, { temperature: 0.8 });
root.captureLogits();

// Fork N candidates — each gets copied KV, logits, sampler, perplexity
const candidates = [1, 2, 3, 4, 5].map((seqId, i) => {
  const branch = root.fork(seqId);
  branch.reseedSampler(1000 + i); // Unique PRNG per branch
  return branch;
});

// Generate in parallel (interleaved round-robin)
for (let t = 0; t < 50; t++) {
  for (const branch of candidates) {
    const { token, isStop } = branch.produce(); // Sample (no KV write)
    if (isStop) continue;
    branch.commit(token); // Accept + decode + capture
  }
}

// Select best by perplexity, prune losers
const best = candidates.reduce((a, b) => a.perplexity < b.perplexity ? a : b);
for (const c of candidates) { if (c !== best) c.prune(); }
```

**What `fork()` clones:** KV cache sequence, logits snapshot, sampler chain (penalties + PRNG), perplexity tracker. Under unified KV (the default), forking is a metadata-only operation — no KV tensor buffers are copied.

**Use cases:** Best-of-N sampling, speculative decoding, MCTS/LATS tree search, beam search.

See [`examples/best-of-n/`](./examples/best-of-n/) and [`examples/speculative/`](./examples/speculative/) for complete patterns.

### Low-Level Forking

For fine-grained control without the Branch wrapper, raw KV and state operations are available:

| Approach             | Method                            | Use Case                                     |
| -------------------- | --------------------------------- | -------------------------------------------- |
| **Tag copy**         | `kvSeqCopy(src, dst)`             | Parallel branches with different seqIds      |
| **Snapshot/restore** | `kvCacheSave()` / `kvCacheLoad()` | Sequential exploration, return to checkpoint |

[`examples/grammar/`](./examples/grammar/) uses snapshot/restore — save state, explore branches sequentially, restore between each:

```javascript
const snapshot = await ctx.kvCacheSave(0); // Save checkpoint
// ... explore branch ...
await ctx.kvCacheLoad(0, snapshot); // Return to checkpoint
```

### Entropy as Control Signal

Model uncertainty mid-generation enables dynamic behavior:

```javascript
const entropy = ctx.modelEntropy('bits');

if (entropy > 4.0) {
  // High uncertainty — model is guessing
  // Trigger retrieval, reduce temperature, or branch
}
```

See [`examples/entropy/`](./examples/entropy/) for entropy-triggered sampling strategies.

### Pull Loop with Generators

For branching mid-generation, generators provide natural backpressure:

```javascript
function* tokenGenerator(ctx, grammarHandle) {
  while (true) {
    const logits = ctx.getLogits();
    ctx.applySampler(grammarHandle, logits);
    const token = ctx.sample({ temperature: 0.7 });
    if (ctx.isStopToken(token)) return;
    ctx.acceptSamplerToken(grammarHandle, token);
    yield { token, text: ctx.tokenToText(token) };
  }
}

// Consumer controls pace — stop at branch point
for (const { token, text } of gen) {
  if (accumulated.includes('"city"')) break; // Pause here, branch
}
```

See [`examples/grammar/`](./examples/grammar/) for the full pull loop pattern.

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

| Method                        | Returns             | Description                     |
| ----------------------------- | ------------------- | ------------------------------- |
| `tokenize(text)`              | `Promise<number[]>` | Text → token IDs                |
| `detokenize(tokens)`          | `Promise<string>`   | Token IDs → text                |
| `tokenToText(token)`          | `string`            | Single token → text (streaming) |
| `decode(tokens, pos, seqId?)` | `Promise<void>`     | Forward pass, updates KV cache  |
| `sample(params?)`             | `number`            | Sample next token               |
| `isStopToken(token)`          | `boolean`           | Check for EOS token             |
| `getLogits()`                 | `Float32Array`      | Raw logits (zero-copy view)     |

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

### Grammar (Handle-Based)

| Method                           | Returns  | Description                 |
| -------------------------------- | -------- | --------------------------- |
| `jsonSchemaToGrammar(schema)`    | `string` | Schema → GBNF               |
| `createSampler(grammarStr)`      | `number` | Create grammar handle       |
| `cloneSampler(handle)`           | `number` | Clone grammar state         |
| `applySampler(handle, logits)`   | `void`   | Apply constraints to logits |
| `acceptSamplerToken(handle, id)` | `void`   | Advance parser state        |
| `freeSamplerHandle(handle)`      | `void`   | Release grammar handle      |

### Metrics

| Method                                  | Returns         | Description                                |
| --------------------------------------- | --------------- | ------------------------------------------ |
| `modelEntropy(base?, logits?)`          | `number`        | Distribution entropy (bits/nats)           |
| `modelSurprisal(token, base?, logits?)` | `number`        | Token surprisal (supports captured logits) |
| `createPerplexityTracker()`             | `TrackerHandle` | Create tracker (forkable)                  |
| `clonePerplexityTracker(handle)`        | `TrackerHandle` | Clone tracker state                        |
| `addSurprisal(handle, value)`           | `void`          | Add to tracker                             |
| `getPerplexity(handle)`                 | `number`        | Get current PPL                            |
| `freePerplexityTracker(handle)`         | `void`          | Release tracker                            |

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
