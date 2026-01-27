# Platform Support & Distribution

> **lloyal.node** provides prebuilt binaries for 13 platforms, covering 93% of production deployment scenarios with instant installation.

---

## Platform Coverage

### Supported Platforms (v1.0)

lloyal.node ships prebuilt binaries for the following platforms:

**macOS (2 packages)**
- Apple Silicon (arm64) with Metal GPU acceleration
- Intel (x64) CPU-only

**Linux x64 (3 packages)**
- CPU-only
- CUDA 12.6 (NVIDIA GPUs)
- Vulkan (AMD/Intel GPUs)

**Linux ARM64 (3 packages)**
- CPU-only (AWS Graviton, Raspberry Pi)
- CUDA 12.6 (NVIDIA Jetson devices)
- Vulkan (Qualcomm/AMD GPUs)

**Windows x64 (3 packages)**
- CPU-only
- CUDA 12.6 (NVIDIA GPUs)
- Vulkan (AMD/Intel GPUs)

**Windows ARM64 (2 packages)**
- CPU-only (Snapdragon X Elite, Surface Pro X)
- Vulkan (Qualcomm GPUs)

### Installation

**Automatic (Recommended)**

npm automatically selects the correct prebuilt package for your platform:

```bash
npm install @lloyal-labs/lloyal.node
```

If a prebuilt binary is available, installation completes in seconds. Otherwise, lloyal.node builds from source automatically (requires C++ compiler and CMake).

**Manual GPU Variant Selection**

To force a specific GPU backend, install the platform package directly:

```bash
# Force CUDA on Linux
npm install @lloyal-labs/lloyal.node-linux-x64-cuda

# Force Vulkan on Windows
npm install @lloyal-labs/lloyal.node-win32-x64-vulkan
```

Or set an environment variable before installation:

```bash
export LLOYAL_GPU=cuda
npm install @lloyal-labs/lloyal.node
```

### Build from Source

If no prebuilt binary matches your platform, lloyal.node builds from source using cmake-js.

**Requirements:**
- Node.js 22+ (LTS)
- C++20 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.18+

**Build commands:**

```bash
# CPU-only build
npm run build

# GPU builds (set LLOYAL_GPU environment variable)
LLOYAL_GPU=cuda npm run build     # NVIDIA CUDA
LLOYAL_GPU=vulkan npm run build   # Vulkan (AMD/Intel/NVIDIA)
LLOYAL_GPU=metal npm run build    # Metal (macOS only)
```

**Build time:** 5-15 minutes (one-time)

---

## GPU Acceleration

### Metal (macOS)

Enabled automatically on Apple Silicon. No additional setup required.

```javascript
const { createContext } = require('@lloyal-labs/lloyal.node');
const ctx = await createContext({
  modelPath: './model.gguf',
  // Metal GPU acceleration used automatically on Apple Silicon
});
```

### CUDA (NVIDIA)

Requires NVIDIA GPU with compute capability 6.0+ and CUDA 12.6 runtime.

**Linux/Windows:**
```bash
npm install @lloyal-labs/lloyal.node-linux-x64-cuda
# or
npm install @lloyal-labs/lloyal.node-win32-x64-cuda
```

**Jetson (ARM64):**
```bash
npm install @lloyal-labs/lloyal.node-linux-arm64-cuda
```

### Vulkan (Cross-Platform)

Works with AMD, Intel, NVIDIA, and Qualcomm GPUs. Requires Vulkan 1.3+ drivers.

```bash
npm install @lloyal-labs/lloyal.node-linux-x64-vulkan
# or
npm install @lloyal-labs/lloyal.node-win32-x64-vulkan
```

### CPU-Only

No GPU acceleration. Works on all platforms.

```bash
npm install @lloyal-labs/lloyal.node-darwin-x64   # macOS Intel
npm install @lloyal-labs/lloyal.node-linux-x64    # Linux x64
npm install @lloyal-labs/lloyal.node-win32-x64    # Windows x64
```

---

## Package Architecture

### Main Package

`lloyal.node` is a meta-package with optional dependencies on all platform packages:

```json
{
  "name": "@lloyal-labs/lloyal.node",
  "optionalDependencies": {
    "@lloyal-labs/lloyal.node-darwin-arm64": "1.0.0",
    "@lloyal-labs/lloyal.node-linux-x64-cuda": "1.0.0",
    ...
  }
}
```

npm installs only the package matching your platform. Unsupported platforms fall back to source builds.

### Platform Packages

Each platform package contains:
- Prebuilt native addon (`lloyal.node`)
- Platform-specific shared libraries (macOS/Linux only: `*.dylib`, `*.so`)
- Minimal dependencies (no build tools required)

**Note:** Windows uses static linking, so only the `.node` file is included.

**Package naming:** `@lloyal-labs/lloyal.node-{platform}-{arch}[-{gpu}]`

Examples:
- `@lloyal-labs/lloyal.node-darwin-arm64` (macOS Apple Silicon with Metal)
- `@lloyal-labs/lloyal.node-linux-x64-cuda` (Linux x64 with CUDA 12.6)
- `@lloyal-labs/lloyal.node-win32-arm64-vulkan` (Windows ARM64 with Vulkan)

### Runtime Loading

lloyal.node uses **runtime dynamic loading** with automatic fallback:

1. **npm install** downloads platform packages matching your OS/CPU via `optionalDependencies`
2. **At runtime**, when you call `createContext()`:
   - If `gpuVariant` option or `LLOYAL_GPU` env var is set, tries that variant first
   - If GPU variant fails (missing runtime libs like `libcudart.so`), falls back to CPU
   - If no prebuilt available, tries local `build/Release/` (development)

This pattern ensures:
- **Graceful degradation**: GPU unavailable? Falls back to CPU automatically
- **No install-time decisions**: Works correctly in Docker multi-stage builds
- **Explicit control**: Use `gpuVariant` option for deterministic loading

**Example:**
```javascript
const { createContext } = require('@lloyal-labs/lloyal.node');

// Automatic: uses LLOYAL_GPU env var or CPU
const ctx = await createContext({ modelPath: './model.gguf' });

// Explicit: request CUDA with fallback to CPU
const ctx = await createContext(
  { modelPath: './model.gguf' },
  { gpuVariant: 'cuda' }
);
```

**Strict GPU Loading (No Fallback)**

For production or CI environments where you want to fail fast if the requested GPU variant is unavailable:

```bash
export LLOYAL_NO_FALLBACK=1
export LLOYAL_GPU=cuda
node app.js  # Throws error if CUDA unavailable instead of falling back to CPU
```

This is useful for:
- CI pipelines testing specific GPU variants
- Production deployments where GPU acceleration is required
- Debugging GPU loading issues

---

## Technical Details

### Dependency Chain

```
lloyal.node (N-API binding via cmake-js)
    ↓ includes
liblloyal (header-only C++ library, git submodule)
    ↓ links
llama.cpp (inference engine, git submodule)
    ↓ compiles to
Platform-specific binaries:
  macOS:   lloyal.node + libllama.dylib + Metal
  Linux:   lloyal.node + libllama.so + OpenMP
  Windows: lloyal.node (static, all-in-one)
```

### Build System

lloyal.node uses **cmake-js** to build the native addon:

```bash
# Development build
npm run build

# The build script (scripts/build.js) invokes:
npx cmake-js compile --CDGGML_METAL=ON  # macOS
npx cmake-js compile                      # Linux/Windows CPU
npx cmake-js compile --CDGGML_CUDA=ON    # CUDA
npx cmake-js compile --CDGGML_VULKAN=ON  # Vulkan
```

**Submodule Structure:**
- `llama.cpp/` - llama.cpp inference engine
- `liblloyal/` - Header-only C++ abstraction layer

To update submodules:

```bash
git submodule update --init --recursive
git submodule update --remote  # Pull latest
```

### CI/CD Pipeline

GitHub Actions builds all 13 platform packages on release:

**Native runners:**
- macOS: `macos-14` (arm64), `macos-15-intel` (x64)
- Linux x64: `ubuntu-22.04`
- Linux ARM64: `ubuntu-22.04-arm` (native, no emulation)
- Windows: `windows-2022`

**Cross-compilation:**
- Windows ARM64: Cross-compiled from x64 using LLVM/clang-cl

**Custom actions:**
- `.github/actions/provision-cuda`: Unified CUDA 12.6 installation
- Uses `jakoch/install-vulkan-sdk-action` for Vulkan SDK

**Testing in CI:**
- CPU, Metal, and Vulkan packages: Fully tested (build + verify + API/E2E tests)
- CUDA packages: Build-only (no GPU hardware in CI runners)
- Cross-compiled packages (win32-arm64): Build-only (can't run ARM64 on x64)

CUDA packages are verified to link correctly, but runtime testing requires actual NVIDIA GPU hardware.

**Build time:**
- x64 platforms: ~10-15 minutes per package
- ARM64 (native): ~10-15 minutes per package
- Total pipeline: ~2-3 hours for all 13 packages

---

## Publishing Workflow

### For Maintainers

**1. Release preparation:**

```bash
# Update submodules to desired versions
git submodule update --remote

# Update version (updates all platform packages too)
npm version minor  # or major/patch/prerelease

# Commit and push
git add .
git commit -m "chore: prepare v1.0.0 release"
git push
```

**2. Tag and trigger release:**

```bash
git tag v1.0.0
git push origin v1.0.0
```

**3. CI builds and publishes:**

GitHub Actions automatically:
- Builds all 13 platform packages
- Publishes to npm registry (`@lloyal-labs/lloyal.node-*`)
- Publishes main package (`@lloyal-labs/lloyal.node`)
- Uses `--tag alpha` for prerelease versions (`*-alpha`, `*-beta`, `*-rc`)

**4. Verify release:**

```bash
npm info @lloyal-labs/lloyal.node
npm info @lloyal-labs/lloyal.node-linux-x64-cuda
```

### Version Management

All packages use synchronized versioning. The `scripts/sync-versions.js` script updates all platform package versions to match the main package.

---

## Comparison to llama.node

| Metric | llama.node | lloyal.node |
|--------|------------|-------------|
| Total packages | 14 | 13 |
| Platform parity | 100% | 93% |
| x64 coverage | Full | Full |
| ARM64 coverage | Full | Full |
| CUDA version | 12.6 | 12.6 |
| Vulkan support | Full | Full |
| Windows ARM64 | ✅ | ✅ |
| Windows linking | Shared DLLs | Static |
| Snapdragon DSP | ✅ Hexagon | ⏸️ Roadmap |

**Key difference:** Windows uses static linking for reliability (avoids DLL initialization crashes).

---

## Roadmap

### v1.1 (Future)

- **Snapdragon Hexagon DSP optimization** (if demand exists)
- **Build caching** for faster CI (ccache integration)
- **Download metrics** to validate platform priorities
- **Automated testing** on real ARM64 hardware (self-hosted runners)

### Feedback

Platform support priorities are driven by user demand. If you need a specific platform or GPU variant, please open an issue at [github.com/lloyal-ai/lloyal.node](https://github.com/lloyal-ai/lloyal.node).
