# liblloyal-node

Thin N-API wrapper over [liblloyal](https://github.com/lloyal-ai/liblloyal) for Node.js - raw llama.cpp inference primitives.

## Features

- **Raw & Thin**: Direct access to llama.cpp primitives via liblloyal
- **Zero-Copy Logits**: `getLogits()` returns Float32Array pointing to llama.cpp memory
- **Native Reference**: Includes native entropy/greedy implementations for testing
- **TypeScript**: Full type definitions included

## Primary Use Case

Integration testing for [tsampler](../tsampler) - validates TypeScript sampler implementations against native references.

## Installation

```bash
npm install liblloyal-node
```

### Prerequisites

- Node.js ≥18
- C++20 compiler
- Pre-built llama.cpp library (see below)

## Building from Source

```bash
# 1. Clone with submodules
git clone --recursive https://github.com/lloyal-ai/liblloyal-node.git
cd liblloyal-node

# 2. Build llama.cpp (one-time setup)
cd llama.cpp
cmake -B build-macos -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-macos -j
cd ..

# This builds multiple static libraries:
# - libllama.a (3.4M) - main llama.cpp library
# - libggml-base.a (806K) - base GGML operations
# - libggml-cpu.a (831K) - CPU-specific implementations
# - libggml-metal.a (728K) - Metal GPU support (macOS)

# 3. Build N-API addon
npm install  # Automatically runs setup-headers.js to create symlinks
npm run build
```

### How Include Paths Work

liblloyal expects `#include <llama/llama.h>`, but llama.cpp provides headers at `include/llama.h`.

**Solution:** The `npm install` script automatically creates a symlink structure:
- `include/llama/` → `llama.cpp/include/*.h`
- `include/ggml/` → `llama.cpp/ggml/include/*.h`

These symlinks are **gitignored** and regenerated on each `npm install`. This approach:
- Respects liblloyal's include path expectations (external package boundary)
- Doesn't modify llama.cpp submodule structure
- Works across platforms (Node.js handles symlinks portably)
- Zero disk overhead (symlinks, not copies)

**Note:** The package uses git submodules for `liblloyal` and `llama.cpp`. If you already cloned without `--recursive`, run:

```bash
git submodule update --init --recursive
```

## Usage

```typescript
import { createContext } from 'liblloyal-node';

const ctx = await createContext({
  modelPath: './model.gguf',
  nCtx: 2048,
  nThreads: 4
});

try {
  // Tokenize
  const tokens = await ctx.tokenize("The capital of France is");

  // Decode (forward pass)
  await ctx.decode(tokens, 0);

  // Get raw logits (zero-copy!)
  const logits = ctx.getLogits();  // Float32Array

  // Native reference implementations (for testing)
  const entropy = ctx.computeEntropy();  // nats
  const token = ctx.greedySample();      // token ID

  console.log(`Entropy: ${entropy.toFixed(3)} nats`);
  console.log(`Greedy token: ${token}`);
} finally {
  ctx.dispose();  // Free native resources
}
```

## API

### `createContext(options)`

Creates a new inference context.

**Options:**
- `modelPath: string` - Path to .gguf model file (required)
- `nCtx?: number` - Context size (default: 2048)
- `nThreads?: number` - Number of threads (default: 4)

**Returns:** `Promise<SessionContext>`

### `SessionContext`

#### Core Primitives

- **`getLogits(): Float32Array`** - Get raw logits (zero-copy, valid until next decode)
- **`decode(tokens: number[], position: number): Promise<void>`** - Decode tokens through model
- **`tokenize(text: string): Promise<number[]>`** - Tokenize text to token IDs
- **`detokenize(tokens: number[]): Promise<string>`** - Detokenize tokens to text

#### Native References (for testing)

- **`computeEntropy(): number`** - Native entropy computation (nats)
- **`greedySample(): number`** - Native greedy sampling

#### Lifecycle

- **`dispose(): void`** - Free native resources

#### Properties

- **`vocabSize: number`** - Model vocabulary size (readonly)

## Example: Testing TS Sampler

```typescript
import { createContext } from 'liblloyal-node';
import { computeModelEntropy } from '../tsampler';

const ctx = await createContext({ modelPath: './model.gguf' });

const tokens = await ctx.tokenize("Once upon a time");
await ctx.decode(tokens, 0);

const logits = ctx.getLogits();

// TS implementation
const tsEntropy = computeModelEntropy(logits);

// Native reference
const nativeEntropy = ctx.computeEntropy();

// Should match within float precision
assert(Math.abs(tsEntropy - nativeEntropy) < 1e-5);

ctx.dispose();
```

## Architecture

```
┌─────────────────────────────────────┐
│  JavaScript (lib/index.js)          │
│  - createContext()                  │
│  - SessionContext                   │
└──────────────┬──────────────────────┘
               │
               │ N-API
               │
┌──────────────▼──────────────────────┐
│  C++ (src/SessionContext.cpp)       │
│  - Napi::ObjectWrap                 │
│  - Async workers for I/O ops        │
└──────────────┬──────────────────────┘
               │
               │ uses
               │
┌──────────────▼──────────────────────┐
│  liblloyal (header-only)            │
│  - decoder, sampler, tokenizer      │
└──────────────┬──────────────────────┘
               │
               │ wraps
               │
┌──────────────▼──────────────────────┐
│  llama.cpp                          │
│  - libllama.a, libggml.a            │
└─────────────────────────────────────┘
```

## Development

```bash
# Clean build
npm run clean

# Debug build (with symbols)
npm run build:debug

# Run tests (once implemented)
npm test
```

## License

MIT
