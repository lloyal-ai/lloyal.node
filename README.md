# lloyal.node

Node.js bindings for [liblloyal](https://github.com/lloyal-ai/liblloyal)—the inference kernel that orchestrates llama.cpp in-process for agentic inference patterns.

## Installation

```bash
npm install lloyal.node
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

Falls back to source build if your platform isn't covered.

```bash
LLOYAL_GPU=cuda npm install    # NVIDIA
LLOYAL_GPU=vulkan npm install  # AMD/Intel
LLOYAL_GPU=cpu npm install     # Force CPU
```

See [DISTRIBUTION.md](./docs/DISTRIBUTION.md) for package details.

## Quick Start

Complete example with greedy sampling:

```typescript
import { createContext } from 'lloyal.node';

async function generate(prompt: string, maxTokens = 100): Promise<string> {
  const ctx = await createContext({
    modelPath: './model.gguf',
    nCtx: 2048,
    nThreads: 4,
  });

  try {
    const tokens = await ctx.tokenize(prompt);
    await ctx.decode(tokens, 0);

    const output: number[] = [];
    let pos = tokens.length;

    for (let i = 0; i < maxTokens; i++) {
      const token = ctx.greedySample();
      if (token < 0) break; // EOS

      output.push(token);
      await ctx.decode([token], pos++);
    }

    return ctx.detokenize(output);
  } finally {
    ctx.dispose();
  }
}

const response = await generate('The capital of France is');
console.log(response);
```

## Test-Time Alignment

TTA is token-level test-time alignment by exposing logits so TypeScript can apply stateful policies/constraints to the full next-token distribution before sampling—no retraining. Enabling fusion of application state with sampling strategy at every token step. Instead of generating text and validating after, you enforce constraints _during_ generation.

This requires two things:

1. **Raw logits** — the probability distribution over all possible next tokens
2. **TypeScript sampling** — so your app logic can modify probabilities before selection

lloyal.node provides the logits. [tsampler](https://github.com/lloyal-ai/tsampler) provides the sampling:

```typescript
import { createContext } from 'lloyal.node';
import {
  sampleWithStrategy,
  computeModelEntropy,
  TokenHistoryTracker,
  SamplerWorkspace,
  Xoroshiro128Plus,
} from '@lloyal/tsampler';

const ctx = await createContext({ modelPath: './model.gguf' });
const prng = new Xoroshiro128Plus(42); // Deterministic PRNG
const tokenHistory = new TokenHistoryTracker(64); // For repetition penalties
const workspace = new SamplerWorkspace(256); // Pre-allocated, zero-alloc hot path

const tokens = await ctx.tokenize(prompt);
await ctx.decode(tokens, 0);

let pos = tokens.length;
const output: number[] = [];

while (output.length < maxTokens) {
  const logits = ctx.getLogits();

  // === YOUR STEERING LOGIC HERE ===

  // Enforce domain rules
  if (currency === 'JPY') {
    logits[DECIMAL_TOKEN] = -Infinity; // JPY has no decimal subdivision
  }

  // Adapt to model confidence
  const entropy = computeModelEntropy(logits);
  const params =
    entropy < 2.0
      ? { topK: 256, temperature: 1.5 } // Low confidence → explore more
      : { topK: 40, temperature: 0.8 }; // High confidence → stay focused

  // === END STEERING LOGIC ===

  const token = sampleWithStrategy(logits, {
    tokenHistory,
    params,
    workspace,
    prng,
  });

  if (token < 0) break;

  tokenHistory.accept(token);
  output.push(token);
  await ctx.decode([token], pos++);
}
```

### Domain Constraints

```typescript
// Financial: JPY has no decimal subdivision
if (currency === 'JPY' && parsingAmount) {
  logits[DECIMAL_TOKEN] = -Infinity;
  DIGIT_TOKENS.forEach((id) => (logits[id] += 2.0));
}

// Legal: Boost required terminology
if (contractType === 'NDA') {
  CONFIDENTIALITY_TOKENS.forEach((id) => (logits[id] += 5.0));
}

// Medical: Enforce terminology based on actual lab values
if (glucoseLevel > normalMax) {
  ELEVATED_TOKENS.forEach((id) => (logits[id] += 10.0));
  NORMAL_TOKENS.forEach((id) => (logits[id] = -Infinity));
}
```

### Quality Gates

```typescript
import { computeModelSurprisal, RollingPerplexity } from '@lloyal/tsampler';

const ppl = new RollingPerplexity();

while (generating) {
  const logits = ctx.getLogits();
  const token = sampleWithStrategy(logits, {
    tokenHistory,
    params,
    workspace,
    prng,
  });

  const surprisal = computeModelSurprisal(logits, token);
  ppl.addSurprisal(surprisal);

  if (ppl.ppl() > 50) {
    // Generation quality degrading — options:
    // 1. Trigger RAG retrieval for more context
    // 2. Prune KV cache (evict stale context)
    // 3. Early stop and retry with different prompt
  }

  // ...
}
```

### Entropy-Adaptive Retrieval

```typescript
import { computeModelEntropy } from '@lloyal/tsampler';

while (generating) {
  const logits = ctx.getLogits();
  const entropy = computeModelEntropy(logits);

  if (entropy > 5.0) {
    // Model is uncertain — retrieve relevant context
    const context = await rag.retrieve(currentQuery);
    await injectContext(ctx, context);
    continue; // Re-evaluate with new context
  }

  const token = sampleWithStrategy(logits, {
    tokenHistory,
    params,
    workspace,
    prng,
  });
  // ...
}
```

## Why TypeScript Sampling?

|                         | Native C++   | TypeScript (tsampler) |
| ----------------------- | ------------ | --------------------- |
| Speed                   | ~0.3ms/token | ~3-5ms/token          |
| Overhead vs 50ms decode | —            | ~6-10%                |
| Logit steering          | ❌           | ✅                    |
| Adaptive strategies     | ❌           | ✅                    |
| OTA updates             | Rebuild app  | Ship new JS           |
| Debugging               | printf       | Full inspect          |

The overhead is imperceptible. A 50ms decode dominates; 3ms sampling is noise.

### tsampler Capabilities

[tsampler](https://github.com/lloyal-ai/tsampler) provides llama.cpp sampling parity in pure TypeScript:

**Sampling methods:** greedy, top-k, top-p, min-p, typical-p, top-n-sigma, temperature, mirostat v1/v2

**Penalties:** repetition, frequency, presence (exact llama.cpp formulas)

**Infrastructure:**

- `Xoroshiro128Plus` — deterministic PRNG, reproducible generations
- `TokenHistoryTracker` — sliding window for penalty calculations
- `SamplerWorkspace` — pre-allocated buffers, zero-alloc hot path
- `computeModelEntropy()` — Shannon entropy in nats
- `computeModelSurprisal()` — per-token surprisal
- `RollingPerplexity` — streaming perplexity tracking

### Native References

lloyal.node includes native C++ implementations for validation:

```typescript
// TypeScript implementation
const tsEntropy = computeModelEntropy(logits);

// Native reference (C++)
const nativeEntropy = ctx.computeEntropy();

// Should match within float precision
console.assert(Math.abs(tsEntropy - nativeEntropy) < 1e-5);
```

Available references:

- `ctx.computeEntropy()` — Shannon entropy in nats
- `ctx.greedySample()` — argmax token ID

Build with confidence. Validate against native. Deploy TypeScript.

## Embeddings

lloyal.node supports embedding extraction with configurable pooling:

```typescript
import { createContext } from 'lloyal.node';

const ctx = await createContext({
  modelPath: './nomic-embed-text.gguf',
  embeddings: true,
  poolingType: 1, // 0=NONE, 1=MEAN, 2=CLS, 3=LAST
});

async function embed(text: string): Promise<Float32Array> {
  const tokens = await ctx.tokenize(text);
  await ctx.encode(tokens);

  const embedding = ctx.getEmbeddings(true); // L2-normalized
  await ctx.kvCacheClear(); // Reset for next text

  return embedding;
}

const vec = await embed('Document to embed');
console.log(`Dimension: ${ctx.getEmbeddingDimension()}`); // e.g., 768
```

## API Reference

**📖 [Full API Documentation](https://lloyal-ai.github.io/lloyal.node)** - Complete reference with examples and type definitions

### Context Creation

```typescript
const ctx = await createContext({
  modelPath: string,       // Path to .gguf file (required)
  nCtx?: number,           // Context size (default: 2048)
  nThreads?: number,       // CPU threads (default: 4)
  nGpuLayers?: number,     // Layers to offload to GPU (default: 0)
  embeddings?: boolean,    // Enable embedding mode (default: false)
  poolingType?: number     // 0=NONE, 1=MEAN, 2=CLS, 3=LAST (default: 0)
});
```

### Inference

| Method                     | Returns             | Description                                           |
| -------------------------- | ------------------- | ----------------------------------------------------- |
| `tokenize(text)`           | `Promise<number[]>` | Text → token IDs                                      |
| `detokenize(tokens)`       | `Promise<string>`   | Token IDs → text                                      |
| `decode(tokens, position)` | `Promise<void>`     | Forward pass, populates KV cache                      |
| `getLogits()`              | `Float32Array`      | Vocabulary-sized probability distribution (zero-copy) |

### Native References

| Method             | Returns  | Description             |
| ------------------ | -------- | ----------------------- |
| `greedySample()`   | `number` | Argmax token ID         |
| `computeEntropy()` | `number` | Shannon entropy in nats |

### Embeddings

| Method                      | Returns         | Description                                |
| --------------------------- | --------------- | ------------------------------------------ |
| `encode(tokens)`            | `Promise<void>` | Forward pass for embedding extraction      |
| `getEmbeddings(normalize?)` | `Float32Array`  | Embedding vector, optionally L2-normalized |
| `getEmbeddingDimension()`   | `number`        | Vector dimension                           |
| `kvCacheClear()`            | `Promise<void>` | Clear KV cache between texts               |

### Lifecycle

| Method      | Description                                           |
| ----------- | ----------------------------------------------------- |
| `dispose()` | Free native resources. **Required** — call when done. |

## LLoyal Ecosystem

| Package                                                 | Language     | What it does                                                                                                                                                                                             |
| ------------------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [liblloyal](https://github.com/lloyal-ai/liblloyal)     | C++          | Inference kernel. Orchestrates llama.cpp with composable primitives: tokenization, decoding, KV cache, sampling chains, metrics, embeddings. Plus `branch.hpp` / `lease.hpp` for state forking and SMMA. |
| **lloyal.node**                                         | N-API        | Node.js bindings. Zero-copy logits, native references for validation.                                                                                                                                    |
| [tsampler](https://github.com/lloyal-ai/tsampler)       | TypeScript   | Sampling library with llama.cpp parity. All filters, penalties, entropy metrics. Plugin for lloyal.node—consumes logits, returns tokens.                                                                 |
| [nitro-llama](https://github.com/lloyal-ai/nitro-llama) | React Native | Mobile bindings via Nitro Modules. Same liblloyal primitives on iOS/Android.                                                                                                                             |

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup, build instructions, and release process.

## License

Apache 2.0 — See [LICENSE](./LICENSE) for details.
