# lloyal.node

**Covalent inference for Node.js**

Forkable inference state for llama.cpp — Branch a generation into a tree — prefix sharing is the bond across branches while each owns its own machinery (sampler chain, seed, grammar, logits snapshot, perplexity tracker) enabling controlled divergence at decode time.

## Branch API

Fork from root for best-of-N, fork from children for MCTS/beam search, fork from a draft for speculative decoding. The produce/commit protocol separates sampling from state advancement — sample without writing to KV, inspect the result, then decide whether to commit.

```javascript
import { createContext, Branch } from "@lloyal-labs/lloyal.node";

const ctx = await createContext({ modelPath: "./model.gguf", nSeqMax: 8 });
const tokens = await ctx.tokenize("Once upon a time");
await ctx.decode(tokens, 0, 0);

// Create root branch, freeze logits from prefill
const root = Branch.create(ctx, 0, tokens.length, { temperature: 0.8 });
root.captureLogits();

// Fork N candidates — KV prefix shared, sampler/grammar/logits/perplexity cloned
const candidates = [1, 2, 3, 4, 5].map((seqId, i) => {
  const branch = root.fork(seqId);
  branch.reseedSampler(1000 + i);
  return branch;
});

// Generate (interleaved round-robin)
for (let t = 0; t < 50; t++) {
  for (const branch of candidates) {
    const { token, isStop } = branch.produce(); // Sample, no KV write
    if (isStop) continue;
    branch.commit(token); // Accept + forward pass + capture
  }
}

// Select by perplexity, prune losers
const best = candidates.reduce((a, b) => (a.perplexity < b.perplexity ? a : b));
for (const c of candidates) {
  if (c !== best) c.prune();
}
```

**What `fork()` shares:** KV cache prefix (metadata-only under unified KV — no tensor buffers copied).

**What `fork()` clones:** Logits snapshot, sampler chain (penalties + PRNG), grammar state, logit bias, perplexity tracker.

**Key methods:**

- `produce()` / `commit()` — two-phase: sample without KV write, then advance
- `prune()` — discard loser and its divergent KV entries
- `destroy()` — release handle, keep KV (for winners continuing with raw ops)
- `reseedSampler()` — unique PRNG per fork for stochastic diversity
- `perplexity` — rolling PPL per branch for quality-based selection

---

## Install

```bash
npm install @lloyal-labs/lloyal.node
```

Prebuilt binaries for 13 platform/GPU combinations. GPU selection at runtime, not install time.

| Platform | Arch  | Acceleration        |
| -------- | ----- | ------------------- |
| macOS    | arm64 | Metal               |
| macOS    | x64   | CPU                 |
| Linux    | x64   | CPU / CUDA / Vulkan |
| Linux    | arm64 | CPU / CUDA / Vulkan |
| Windows  | x64   | CPU / CUDA / Vulkan |
| Windows  | arm64 | CPU / Vulkan        |

CI integration testing (real inference):

| Architecture | Test Model     | Template |
| ------------ | -------------- | -------- |
| Llama        | Llama 3.2 1B   | llama3   |
| Phi          | Phi 3.5 Mini   | phi3     |
| Qwen         | Qwen 3 1.7B    | chatml   |
| Gemma        | Gemma 3 1B     | gemma    |
| SmolLM       | SmolLM2 1.7B   | chatml   |
| TinyLlama    | TinyLlama 1.1B | zephyr   |

See [distribution.md](docs/distribution.md) for details.

---

## Examples

| Example                                   | Pattern                                                                    |
| ----------------------------------------- | -------------------------------------------------------------------------- |
| [`best-of-n/`](./examples/best-of-n/)     | Branch API: fork, produce/commit, perplexity selection                     |
| [`speculative/`](./examples/speculative/) | Branch API: draft/verify, fork/prune, bonus token sampling                 |
| [`streaming/`](./examples/streaming/)     | Infinite context via BlinkKV reseeding with sidecar summarization          |
| [`entropy/`](./examples/entropy/)         | `modelEntropy()` mid-generation as control signal                          |
| [`grammar/`](./examples/grammar/)         | Pull loop with generators, JSON schema constraints, KV + grammar branching |
| [`chat/`](./examples/chat/)               | Interactive streaming chat                                                 |
| [`embed/`](./examples/embed/)             | Text embeddings extraction                                                 |

```bash
node examples/best-of-n/best-of-n.mjs
node examples/speculative/speculative.mjs
```

Each example has a README explaining the pattern.

---

## Other Patterns

### Entropy as Control Signal

Model uncertainty mid-generation enables dynamic behavior:

```javascript
const entropy = ctx.modelEntropy("bits");

if (entropy > 4.0) {
  // High uncertainty — model is guessing
  // Trigger retrieval, reduce temperature, or branch
}
```

See [`examples/entropy/`](./examples/entropy/) for entropy-triggered sampling strategies.

### Low-Level KV Operations

For fine-grained control without Branch:

| Approach             | Method                            | Use Case                                     |
| -------------------- | --------------------------------- | -------------------------------------------- |
| **Sequence copy**    | `kvSeqCopy(src, dst)`             | Share prefix across sequences                |
| **Snapshot/restore** | `kvCacheSave()` / `kvCacheLoad()` | Sequential exploration, return to checkpoint |

### Grammar-Constrained Generation

```javascript
const grammar = ctx.jsonSchemaToGrammar(schema);
const handle = ctx.createSampler(grammar);
// Pull loop — consumer controls pace, can branch at any point
```

See [`examples/grammar/`](./examples/grammar/) for the full pull loop pattern.

---

## API Reference

Full API documentation: **[lloyal-ai.github.io/lloyal.node](https://lloyal-ai.github.io/lloyal.node/)**

Generated from [`lib/index.d.ts`](./lib/index.d.ts) with TypeDoc.

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
