# lloyal.node

[![Build & Test](https://github.com/lloyal-ai/lloyal.node/actions/workflows/tests.yml/badge.svg)](https://github.com/lloyal-ai/lloyal.node/actions/workflows/tests.yml)
[![GPU Tests](https://github.com/lloyal-ai/lloyal.node/actions/workflows/gpu-test.yml/badge.svg)](https://github.com/lloyal-ai/lloyal.node/actions/workflows/gpu-test.yml)
[![npm](https://img.shields.io/npm/v/@lloyal-labs/lloyal.node.svg)](https://www.npmjs.com/package/@lloyal-labs/lloyal.node)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-b8087-green.svg)](https://github.com/ggml-org/llama.cpp/releases/tag/b8087)

**Covalent Inference for Node.js**

Composable inference primitives for forkable decode state, shared-prefix KV branching, and continuous tree batching. Branches share a KV prefix while keeping independent machinery — sampler chain, grammar, logits snapshot, perplexity tracker — for controlled divergence at decode time. `BranchStore` packs tokens from N branches (each at a different position, different seq_id, each needing independent logits captured) into a single `llama_batch` and dispatches once. `kv::tenancy` manages seq_id leases automatically — acquired on `create()`/`fork()`, evicted on `prune()`, rebuilt on `retainOnly()`.

Built on [liblloyal](https://github.com/lloyal-ai/liblloyal), a header-only C++20 inference kernel for llama.cpp.

## The Branch API

```javascript
import { createContext, Branch, BranchStore } from "@lloyal-labs/lloyal.node";

const ctx = await createContext({ modelPath: "./model.gguf", nSeqMax: 6 });
const store = new BranchStore(ctx);

// Shared prompt: "Explain quantum entanglement"
const prompt = await ctx.tokenize("Explain quantum entanglement");
await ctx.decode(prompt, 0, 0);

const root = Branch.create(ctx, prompt.length, { temperature: 0.8 });
root.captureLogits();

// Fork 4 branches — each gets a different reasoning prefix
const analogy  = await root.fork();
const formal   = await root.fork();
const socratic = await root.fork();
const visual   = await root.fork();

// Scatter-prefill: inject divergent prefixes in one batched dispatch
// 4 branches × variable lengths → auto bin-packed into minimal GPU calls
await store.prefill([
  [analogy,  await ctx.tokenize("Think of it like two coins...")],    // 12 tokens
  [formal,   await ctx.tokenize("In quantum mechanics, the...")],     // 8 tokens
  [socratic, await ctx.tokenize("What happens when you measure...")], // 10 tokens
  [visual,   await ctx.tokenize("Imagine two particles...")],         // 7 tokens
]);

// Generate — all 4 in lockstep, 1 GPU call per step
const branches = [analogy, formal, socratic, visual];
for (;;) {
  const live = branches.filter(b => !b.disposed);
  if (!live.length) break;
  const produced = live.map(b => ({ b, ...b.produce() }));

  // Prune branches that hit stop tokens
  for (const p of produced.filter(p => p.isStop)) await p.b.prune();

  // Commit survivors — accept + decode in one GPU dispatch
  const items = produced
    .filter(p => !p.isStop)
    .map(p => { p.b.accept(p.token); return [p.b, p.token]; });
  await store.commit(items);
}

// Winner takes all — one seq_keep pass, losers vaporized
const winner = branches
  .filter(b => !b.disposed)
  .reduce((a, b) => (a.perplexity < b.perplexity ? a : b));
await store.retainOnly(winner);
// store.available === nSeqMax - 1 — all leases recovered
```

Or for single-branch generation, Branch is an async iterable — generate until EOG:

```javascript
for await (const { token, text } of branch) {
  process.stdout.write(text);
}
```

## Continuous Tree Batching

Tree search with N branches means N calls to `llama_decode()` — each paying GPU dispatch overhead, memory barriers, and PCIe round-trips. `BranchStore` eliminates this: tokens from N branches — each at a different position, different seq_id, each needing independent logits captured — are packed into a single `llama_batch` and dispatched once. N branches, 1 GPU call.

Two packing strategies for different access patterns:

```javascript
// commit: 1 token per branch — one GPU dispatch for N branches
await store.commit([[branch1, tok1], [branch2, tok2], [branch3, tok3]]);

// prefill: variable tokens per branch — asymmetric injection
await store.prefill([
  [branchA, systemTokens],  // 200 tokens
  [branchB, queryTokens],   //  12 tokens
  [branchC, docTokens],     // 800 tokens
]);
// Greedy bin-packed into ceil(total / nBatch) dispatches
```

## KV Tenancy

Two resources, two scales. Slots (65K) are how many branches can *exist* — cheap CPU state. Leases (`nSeqMax`) are how many can *decode* — scarce KV cache residency. Tenancy manages the scarce resource automatically: leases are acquired on `create()`/`fork()`, evicted on `prune()`, rebuilt on `retainOnly()`. No manual seq_id tracking, ever.

```javascript
store.available;                   // leases remaining — use for width/depth budget
await store.retainOnly(winner);    // nuclear: 1 seq_keep, rebuild vacancy
```

The turn lifecycle: search is surgical (N × `prune()`), promotion is nuclear (1 × `retainOnly()`). Per turn, fork → expand → evaluate → prune losers → repeat. Between turns, promote winner → tree is gone → next turn starts fresh.

## Topology

Parent/child edges are always-on. Simple chat → best-of-N → deep search is one continuum.

```javascript
branch.parent;       // handle or null if root
branch.children;     // child handles
branch.isLeaf;       // no children?
branch.isActive;     // holds a KV lease?
```

| Method | FK analogy | Behavior |
|--------|-----------|----------|
| `prune()` | RESTRICT | Throws if children exist |
| `pruneSubtree()` | CASCADE | Iterative post-order traversal |

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
| Ministral    | Ministral 3B   | mistral  |

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
const grammar = await ctx.jsonSchemaToGrammar(schema);
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
