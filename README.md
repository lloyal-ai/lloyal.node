# lloyal.node

Thin N-API wrapper over [liblloyal](https://github.com/lloyal-ai/liblloyal) for Node.js - raw llama.cpp inference primitives.

## Features

- **Prebuilt Binaries**: Install in <1 minute on 7 common platforms (macOS, Linux, Windows)
- **Raw & Thin**: Direct access to llama.cpp primitives via liblloyal
- **Zero-Copy Logits**: `getLogits()` returns Float32Array pointing to llama.cpp memory
- **GPU Acceleration**: Metal (macOS), CUDA, and Vulkan support with dedicated prebuilts
- **BYO llama.cpp**: Swap `libllama.dylib` for custom builds (dynamic linking)
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
npm install lloyal.node
```

### Prebuilt Binaries (Recommended)

lloyal.node ships with **prebuilt binaries** for common platforms. Installation takes **<1 minute**:

| Platform | Architecture | GPU | Package | Install Time |
|----------|--------------|-----|---------|--------------|
| **macOS** | Apple Silicon (arm64) | Metal | `@lloyal/lloyal.node-darwin-arm64` | <1 min ⚡ |
| **macOS** | Intel (x64) | CPU | `@lloyal/lloyal.node-darwin-x64` | <1 min ⚡ |
| **Linux** | x64 | CPU | `@lloyal/lloyal.node-linux-x64` | <1 min ⚡ |
| **Linux** | x64 | CUDA | `@lloyal/lloyal.node-linux-x64-cuda` | <1 min ⚡ |
| **Linux** | x64 | Vulkan | `@lloyal/lloyal.node-linux-x64-vulkan` | <1 min ⚡ |
| **Windows** | x64 | CPU | `@lloyal/lloyal.node-win32-x64` | <1 min ⚡ |
| **Windows** | x64 | CUDA | `@lloyal/lloyal.node-win32-x64-cuda` | <1 min ⚡ |

**How it works:**
- npm automatically downloads the correct prebuilt for your platform
- Platform packages are listed as `optionalDependencies`
- Falls back to building from source if your platform isn't covered

### Building from Source (Fallback)

If no prebuilt is available for your platform, lloyal.node builds from **vendored sources** (5-15 minutes):

**Prerequisites:**
- Node.js ≥18
- C++20 compiler (GCC, Clang, or MSVC)
- CMake ≥3.14
- node-gyp build tools

**Supported platforms:**
- Any platform with a C++20 compiler and CMake
- GPU backends require additional dependencies (see GPU Acceleration section)

## Using in Your Project

Simply add lloyal.node as a dependency:

```json
{
  "dependencies": {
    "lloyal.node": "^0.1.0"
  }
}
```

Then import and use:

```javascript
const { createContext } = require('lloyal.node');

const ctx = await createContext({
  modelPath: './model.gguf'
});
```

**That's it!** npm handles downloading prebuilts or building from source automatically.

## Development & Contributing

**Clone the repository:**

```bash
# Clone with submodules
git clone --recursive https://github.com/lloyal-ai/lloyal.node.git
cd lloyal.node

# Build from source
npm install
npm run build
```

**Build process:**
- **Linux**: Builds llama.cpp as a single shared library (`.so`) with `-DCMAKE_POSITION_INDEPENDENT_CODE=ON`
- **macOS**: Creates universal binary (arm64+x86_64) `libllama.dylib` with Metal/Accelerate support
- **Windows**: Builds DLLs for llama.cpp + ggml

**Why single combined library?** Dynamic linking to `libllama.so`/`.dylib` enables the "bring your own llama.cpp" pattern while avoiding ODR violations.

**Active development workflow:**
```bash
git submodule update --remote    # Update submodules
npm run clean                    # Clean build artifacts
npm run build                    # Rebuild
```

### GPU Acceleration

By default, lloyal.node auto-detects the best backend for your platform:

| Platform | Default Backend | GPU Support |
|----------|----------------|-------------|
| **macOS (local)** | Metal | ✅ GPU acceleration |
| **macOS (CI)** | CPU | ⚠️ No GPU (virtualized) |
| **Linux** | CPU | Manual via `LLOYAL_GPU` |
| **Windows** | CPU | Manual via `LLOYAL_GPU` |

**Override with `LLOYAL_GPU` environment variable:**

```bash
# Force CPU-only build (disables all GPU backends)
LLOYAL_GPU=cpu npm install

# Enable CUDA (Linux/Windows with NVIDIA GPU)
LLOYAL_GPU=cuda npm install

# Enable Vulkan (Linux/Windows)
LLOYAL_GPU=vulkan npm install

# Enable Metal (macOS only, default on local builds)
LLOYAL_GPU=metal npm install
```

**Requirements by Backend:**

- **CPU**: No additional dependencies (works everywhere)
- **Metal**: macOS only (built-in, requires physical GPU)
- **CUDA**: NVIDIA GPU + [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed
- **Vulkan**: [Vulkan SDK](https://vulkan.lunarg.com/) installed

**⚠️ Runtime Dependencies & Dynamic Linking:**

lloyal.node uses **dynamic linking** to a bundled `libllama.so`/`libllama.dylib`:

```
node_modules/lloyal.node/build/Release/
├── lloyal.node              # N-API wrapper (links to libllama via @rpath)
└── libllama.dylib           # llama.cpp + ggml (bundled, but swappable!)
```

**Batteries included, BYO supported:** The bundled llama library ships with the package, but you can replace it with your own build (same ABI required).

GPU backends introduce **additional runtime dependencies**:

| Backend | Bundled in Package | External Runtime Dependencies | Portable? |
|---------|-------------------|-------------------------------|-----------|
| **CPU** | `libllama.so` only | None | ✅ Yes |
| **Metal** | `libllama.dylib` + Metal framework | macOS frameworks (always available) | ✅ Yes (macOS only) |
| **CUDA** | `libllama.so` + CUDA code | `libcudart.so`, `libcublas.so`, etc. | ❌ No - requires CUDA runtime |
| **Vulkan** | `libllama.so` + Vulkan code | `libvulkan.so` | ❌ No - requires Vulkan drivers |

**CUDA/Vulkan builds are NOT portable** - they require the same GPU libraries at runtime:

```bash
# Build on machine with CUDA
LLOYAL_GPU=cuda npm install  # ✅ Links against CUDA libs

# Deploy to production without CUDA
node app.js  # ❌ Error: libcudart.so.12 not found

# Solution: Install CUDA runtime on production, or rebuild with CPU
LLOYAL_GPU=cpu npm install  # ✅ Portable to any Linux machine
```

Check dynamic dependencies with:
```bash
# Linux
ldd build/Release/lloyal.node

# macOS
otool -L build/Release/lloyal.node
```

**Bring Your Own llama.cpp:**

Advanced users can replace the bundled llama library with a custom build:

```bash
# Build your custom llama.cpp (must match ABI)
cd /path/to/your/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON ...
cmake --build build

# Replace bundled library (AFTER npm install)
cp /path/to/your/llama.cpp/build/libllama.so \
   node_modules/lloyal.node/build/Release/libllama.so

# Verify it loads
node -e "require('lloyal.node').createContext({modelPath: './model.gguf'})"
```

**⚠️ ABI Compatibility Requirements:**
- Same llama.cpp commit/version (API signatures must match)
- Same backend (CPU/CUDA/Vulkan/Metal)
- Same architecture (x86_64/arm64)
- Mismatches cause runtime crashes or undefined behavior

**Use cases:**
- Custom llama.cpp patches
- Organization-specific builds
- Testing upstream llama.cpp changes
- Optimized builds for specific hardware

**Examples:**

```bash
# Deploy to AWS Lambda (CPU-only for compatibility)
LLOYAL_GPU=cpu npm install

# Development on Linux workstation with NVIDIA GPU
LLOYAL_GPU=cuda npm install

# Rebuild with different backend
npm run clean
LLOYAL_GPU=vulkan npm install
```

**Note:** The backend is determined at **build time**, not runtime. To switch backends, you must rebuild with `npm run clean && LLOYAL_GPU=<backend> npm install`.

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
import { createContext } from 'lloyal.node';

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
import { createContext } from 'lloyal.node';
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

## Distribution & Releases

### Platform Package Architecture

lloyal.node uses the **industry-standard prebuilt pattern** (same as sharp, sqlite3, canvas):

```
lloyal.node (main package)
├── optionalDependencies
│   ├── @lloyal/lloyal.node-darwin-arm64
│   ├── @lloyal/lloyal.node-darwin-x64
│   ├── @lloyal/lloyal.node-linux-x64
│   ├── @lloyal/lloyal.node-linux-x64-cuda
│   ├── @lloyal/lloyal.node-linux-x64-vulkan
│   ├── @lloyal/lloyal.node-win32-x64
│   └── @lloyal/lloyal.node-win32-x64-cuda
└── install script (prebuilt or fallback to source)
```

**Platform packages contain:**
```
@lloyal/lloyal.node-darwin-arm64/
├── bin/
│   ├── lloyal.node           # N-API binary
│   └── libllama.dylib        # Shared library
├── index.js                  # Exports path to binary
└── package.json              # os: ["darwin"], cpu: ["arm64"]
```

### Release Process

**For maintainers:**

```bash
# 1. Update vendored sources (if needed)
npm run update-vendors

# 2. Bump version (triggers sync-versions.js)
npm version minor  # or major/patch

# 3. Tag and push
git push origin main --tags

# GitHub Actions automatically:
# - Builds 7 platform packages
# - Publishes to npm as @lloyal/lloyal.node-*
# - Publishes main package with updated optionalDependencies
```

**CI Pipeline:**
- `.github/workflows/release.yml` builds on tag push
- 7 parallel jobs for each platform/GPU variant
- Installs platform dependencies (CUDA toolkit, Vulkan SDK)
- Packages binaries to `bin/` directory
- Publishes all packages with synchronized versions

### Vendoring Strategy

**For npm registry distribution:**
- llama.cpp and liblloyal sources vendored to `vendor/`
- Run `npm run update-vendors` before publishing
- Vendored sources enable source builds for unsupported platforms

**For development:**
- Use git submodules (`git clone --recursive`)
- Update with `git submodule update --remote`

## License

MIT
