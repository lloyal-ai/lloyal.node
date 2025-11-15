# liblloyal-node

Thin N-API wrapper over [liblloyal](https://github.com/lloyal-ai/liblloyal) for Node.js - raw llama.cpp inference primitives.

## Features

- **Raw & Thin**: Direct access to llama.cpp primitives via liblloyal
- **Zero-Copy Logits**: `getLogits()` returns Float32Array pointing to llama.cpp memory
- **Native Reference**: Includes native entropy/greedy implementations for testing
- **TypeScript**: Full type definitions included

## Use Cases

A minimal Node.js binding for llama.cpp inference, suitable for:
- **Testing & Validation**: Compare TypeScript implementations against native references
- **Serverless Deployments**: Lightweight footprint for edge compute and Lambda-style functions
- **Automation & CI**: Build deterministic test suites for LLM-powered workflows
- **Research & Prototyping**: Direct access to llama.cpp primitives without framework overhead

## Installation

```bash
npm install liblloyal-node
```

**Note:** This package includes vendored sources for `liblloyal` and `llama.cpp` dependencies. First install will compile C++ code and take 5-15 minutes.

### Prerequisites

- Node.js ≥18
- C++20 compiler (GCC, Clang, or MSVC)
- CMake (for building llama.cpp)
- node-gyp build tools

## Building from Source

```bash
# Clone with submodules
git clone --recursive https://github.com/lloyal-ai/lloyal.node.git
cd lloyal.node

# Build
npm install
npm run build
```

The `npm install` step automatically:
- **Linux**: Builds llama.cpp as a single shared library (`.so`)
- **macOS**: Builds llama.cpp XCFramework (uses upstream script)
- **Windows**: Builds static libraries and links into N-API module

**Why single combined library?** N-API requires llama.cpp to be built as a single shared library rather than separate static libraries. This ensures proper symbol resolution and prevents initialization issues with static globals.

**For Development:**
If you're actively developing and updating submodules:
```bash
git submodule update --remote    # Update to latest
npm run clean                    # Clean build artifacts
npm install                      # Rebuild
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

**Note for Contributors:** The package uses git submodules for `liblloyal` and `llama.cpp` during development. npm users get vendored sources automatically. If you cloned the repo without `--recursive`:

```bash
git submodule update --init --recursive
```

### Test Models (Git LFS)

The test suite uses [Git LFS](https://git-lfs.com/) to track the SmolLM2 model (~1GB). Install Git LFS before cloning:

```bash
# Install Git LFS (one-time setup)
brew install git-lfs  # macOS
# or: sudo apt-get install git-lfs  # Linux

# Initialize Git LFS
git lfs install

# Clone with LFS files
git clone --recursive https://github.com/lloyal-ai/lloyal.node.git
```

If you already cloned without LFS, pull the model:

```bash
git lfs pull
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

# Run tests
npm test              # Run all tests (API + E2E)
npm run test:api      # API functionality and benchmarks
npm run test:e2e      # Correctness and determinism validation
```

### Tests

- **`test/api.js`**: API functionality tests and performance benchmarks
- **`test/e2e.js`**: End-to-end validation with deterministic output checks

Tests use SmolLM2-1.7B-Instruct with chat templates to simulate real-world usage patterns.

## License

MIT
