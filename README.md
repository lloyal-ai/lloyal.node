# lloyal.node

[![Build & Test](https://github.com/lloyal-ai/lloyal.node/actions/workflows/tests.yml/badge.svg)](https://github.com/lloyal-ai/lloyal.node/actions/workflows/tests.yml)
[![GPU Tests](https://github.com/lloyal-ai/lloyal.node/actions/workflows/gpu-test.yml/badge.svg)](https://github.com/lloyal-ai/lloyal.node/actions/workflows/gpu-test.yml)
[![npm](https://img.shields.io/npm/v/@lloyal-labs/lloyal.node.svg)](https://www.npmjs.com/package/@lloyal-labs/lloyal.node)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-b8087-green.svg)](https://github.com/ggml-org/llama.cpp/releases/tag/b8087)

**Native backend for the lloyal inference platform.**

Prebuilt llama.cpp binaries for 13 platform/GPU combinations, exposing a `SessionContext` that powers the [`@lloyal-labs/sdk`](https://github.com/lloyal-ai/sdk) inference primitives (Branch, BranchStore, Session, Rerank) and [`@lloyal-labs/lloyal-agents`](https://github.com/lloyal-ai/sdk/tree/main/packages/agents) multi-agent framework. Built on [liblloyal](https://github.com/lloyal-ai/liblloyal), a header-only C++20 inference kernel for llama.cpp.

All SDK and agent exports are re-exported from this package for convenience — `import { Branch, runAgents } from "@lloyal-labs/lloyal.node"` works out of the box.

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

## Quick Start

```javascript
import { createContext } from "@lloyal-labs/lloyal.node";
import { Branch, BranchStore } from "@lloyal-labs/sdk";

const ctx = await createContext({ modelPath: "./model.gguf", nSeqMax: 4 });
const store = new BranchStore(ctx);

const root = Branch.create(ctx, 0, { temperature: 0.8 });
await root.prefill(await ctx.tokenize("Explain quantum entanglement"));

// Fork and generate — all branches in lockstep, 1 GPU call per step
const branches = await Promise.all([root.fork(), root.fork(), root.fork()]);
for (;;) {
  const live = branches.filter((b) => !b.disposed);
  if (!live.length) break;
  const produced = live.map((b) => ({ b, ...b.produce() }));
  for (const p of produced.filter((p) => p.isStop)) await p.b.prune();
  const items = produced
    .filter((p) => !p.isStop)
    .map((p) => {
      p.b.accept(p.token);
      return [p.b, p.token];
    });
  await store.commit(items);
}
```

Or for single-branch generation, Branch is an async iterable:

```javascript
for await (const { token, text } of branch) {
  process.stdout.write(text);
}
```

See [`@lloyal-labs/sdk`](https://github.com/lloyal-ai/sdk) for the full Branch API, continuous tree batching, KV tenancy, and topology documentation.

### Without the SDK

`createContext` returns a `SessionContext` — the native interface to llama.cpp. You can use it directly without the SDK's Branch/BranchStore layer:

```javascript
import { createContext } from "@lloyal-labs/lloyal.node";

const ctx = await createContext({ modelPath: "./model.gguf", nSeqMax: 4 });

// Chat templates — model-agnostic formatting + tool calling
const { prompt, grammar, format } = await ctx.formatChat(messages, {
  addGenerationPrompt: true,
  tools: [{ type: "function", function: { name: "search", parameters: schema } }],
});
const { content, toolCalls } = await ctx.parseChatOutput(output, format);

// Branch primitives — what the SDK's Branch class wraps
const handle = ctx._branchCreate(0, samplerParams);
await ctx._branchPrefill(handle, tokens);
const token = ctx._branchSample(handle);
const text = ctx.tokenToText(token);
const isStop = ctx.isStopToken(token);
ctx._branchAccept(handle, token);
const logits = ctx._branchGetLogits(handle);     // Float32Array(vocabSize)
const entropy = ctx._branchModelEntropy(handle);
const child = ctx._branchFork(handle);

// Store primitives — what the SDK's BranchStore wraps
await ctx._storeCommit([handle1, handle2], [tok1, tok2]);  // N branches, 1 GPU call
await ctx._storePrefill([handle], [tokens]);
await ctx._storeRetainOnly(winner);
const available = ctx._storeAvailable();

// KV cache — snapshot, copy, persist
await ctx.kvSeqCopy(0, 1);                      // share prefix across sequences
await ctx.kvCacheSave();                         // snapshot for rollback
await ctx.kvCacheLoad();                         // restore checkpoint
await ctx.kvCacheWriteFile("cache.bin");         // persist to disk

// Embeddings
const embeddings = await ctx.encode("query text");
const dim = ctx.getEmbeddingDimension();

// Grammar + tokenizer
const grammar = await ctx.jsonSchemaToGrammar(schema);
const tokens = await ctx.tokenize("Hello world");
const sep = await ctx.getTurnSeparator();
```

## What This Package Provides

**Native-only** (not in SDK):

- `createContext(options)` — load a GGUF model, return a `SessionContext`
- `loadBinary(options?)` — explicit GPU variant selection with automatic fallback
- Prebuilt binaries for 13 platform/GPU combinations

**Re-exported from [`@lloyal-labs/sdk`](https://github.com/lloyal-ai/sdk):**

- `Branch`, `BranchStore`, `Session`, `Rerank`
- Per-token metrics: `modelEntropy()`, `modelSurprisal()`, `samplingPerplexity`
- Chat formatting: `formatChat()`, `parseChatOutput()`
- Grammar: `jsonSchemaToGrammar()`, `setGrammar()`

**Re-exported from [`@lloyal-labs/lloyal-agents`](https://github.com/lloyal-ai/sdk/tree/main/packages/agents):**

- `runAgents`, `useAgentPool`, `generate`, `diverge`, `createToolkit`
- Structured concurrency DAG via Effection generators
- In-loop orchestration: agents as branches of a single running process

## GPU Variant Selection

```javascript
import { loadBinary, createContext } from "@lloyal-labs/lloyal.node";

// Automatic — uses Metal on macOS, CPU elsewhere
const ctx = await createContext({ modelPath: "./model.gguf" });

// Explicit CUDA
const binding = loadBinary({ gpuVariant: "cuda" });
const ctx = await binding.createContext({ modelPath: "./model.gguf" });
// Falls back to CPU with a warning if CUDA runtime not available
```

## Examples

| Example                           | Pattern                                           |
| --------------------------------- | ------------------------------------------------- |
| [`entropy/`](./examples/entropy/) | `modelEntropy()` mid-generation as control signal |
| [`chat/`](./examples/chat/)       | Interactive streaming chat                        |
| [`embed/`](./examples/embed/)     | Text embeddings extraction                        |

```bash
npx tsx examples/best-of-n/best-of-n.ts
npx tsx examples/chat/chat.ts ./model.gguf
```

## CI Testing

Integration tests run real inference across architectures:

| Architecture | Test Model   | Template |
| ------------ | ------------ | -------- |
| Llama        | Llama 3.2 1B | llama3   |
| Phi          | Phi 3.5 Mini | phi3     |
| Qwen         | Qwen 3 1.7B  | chatml   |
| Gemma        | Gemma 3 1B   | gemma    |
| SmolLM       | SmolLM2 1.7B | chatml   |
| Ministral    | Ministral 3B | mistral  |

See [distribution.md](docs/distribution.md) for details.

## Ecosystem

| Package                                                                                    | Description                                                                  |
| ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| [`@lloyal-labs/sdk`](https://github.com/lloyal-ai/sdk)                                     | Backend-agnostic inference primitives (Branch, BranchStore, Session, Rerank) |
| [`@lloyal-labs/lloyal-agents`](https://github.com/lloyal-ai/sdk/tree/main/packages/agents) | Multi-agent framework — in-loop orchestration via structured concurrency     |
| [liblloyal](https://github.com/lloyal-ai/liblloyal)                                        | Header-only C++20 inference kernel for llama.cpp                             |
| **lloyal.node**                                                                            | This package — native backend + prebuilt binaries                            |
| [nitro-llama](https://github.com/lloyal-ai/nitro-llama)                                    | React Native backend via Nitro Modules                                       |
| [tsampler](https://github.com/lloyal-ai/tsampler)                                          | Reference sampler implementation                                             |

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and release process.

## License

Apache 2.0 — See [LICENSE](./LICENSE) for details.
