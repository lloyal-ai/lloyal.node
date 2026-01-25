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
npm install lloyal.node
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
npm install lloyal.node
```

### Build from Source

If no prebuilt binary matches your platform, lloyal.node builds from vendored sources automatically.

**Requirements:**
- C++20 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.18+

**Build time:** 5-15 minutes (one-time)

**Supported platforms not covered by prebuilts:**
- Older or niche platforms
- Custom CPU optimizations
- Development and testing

---

## GPU Acceleration

### Metal (macOS)

Enabled automatically on Apple Silicon. No additional setup required.

```javascript
const { loadModel } = require('lloyal.node');
const model = await loadModel({
  modelPath: './model.gguf',
  gpuLayers: 32  // Offload layers to GPU
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
  "name": "lloyal.node",
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
- Prebuilt native addon (`*.node`)
- Platform-specific shared libraries (`*.dylib`, `*.so`, `*.dll`)
- Minimal dependencies (no build tools required)

**Package naming:** `@lloyal-labs/lloyal.node-{platform}-{arch}[-{gpu}]`

Examples:
- `@lloyal-labs/lloyal.node-darwin-arm64` (macOS Apple Silicon with Metal)
- `@lloyal-labs/lloyal.node-linux-x64-cuda` (Linux x64 with CUDA 12.6)
- `@lloyal-labs/lloyal.node-win32-arm64-vulkan` (Windows ARM64 with Vulkan)

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
| Snapdragon optimization | ✅ Hexagon DSP | ⏸️ Roadmap |

**Missing:** Snapdragon Hexagon DSP optimization (niche edge AI use case). Standard ARM64 packages work on Snapdragon hardware without DSP acceleration.

---

## Technical Details

### Dependency Chain

```
lloyal.node (N-API binding)
    ↓ includes
liblloyal (header-only C++ library)
    ↓ links
llama.cpp (inference engine)
    ↓ compiles to
Platform-specific binaries:
  macOS:   libllama.dylib + Metal support
  Linux:   libllama.so + OpenMP
  Windows: llama.dll + ggml*.dll
```

### Vendoring Strategy

lloyal.node vendors `liblloyal` and `llama.cpp` sources to avoid git submodule issues with npm:

- **Published packages** include vendored sources in `vendor/` directory
- **Git repository** uses submodules for development
- **Version tracking** via `vendor/VERSIONS.json`

To update vendored dependencies:

```bash
git submodule update --remote
npm run update-vendors
```

See [VENDORING.md](../VENDORING.md) for details.

### CI/CD Pipeline

GitHub Actions builds all 13 platform packages on release:

**Native runners:**
- macOS: `macos-14` (arm64), `macos-13` (x64)
- Linux x64: `ubuntu-22.04`
- Linux ARM64: `ubuntu-22.04-arm` (native, no emulation)
- Windows: `windows-2022`

**Cross-compilation:**
- Windows ARM64: Cross-compiled from x64 using LLVM/clang-cl

**Custom actions:**
- `.github/actions/provision-cuda`: Unified CUDA 12.6 installation
- Uses `jakoch/install-vulkan-sdk-action` for Vulkan SDK

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

# Vendor the dependencies
npm run update-vendors

# Update version
npm version minor  # or major/patch

# Commit changes
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
- Publishes main package (`lloyal.node`)

**4. Verify release:**

```bash
npm info lloyal.node
npm info @lloyal-labs/lloyal.node-linux-x64-cuda
```

### Version Management

All packages use synchronized versioning:

```bash
# Sync platform package versions with main package
npm run version
```

This automatically updates `optionalDependencies` in `package.json` and `version` fields in all platform packages.

---

## Roadmap

### v1.1 (Future)

- **Snapdragon Hexagon DSP optimization** (if demand exists)
- **Build caching** for faster CI (ccache integration)
- **Download metrics** to validate platform priorities
- **Automated testing** on real ARM64 hardware (self-hosted runners)

### Feedback

Platform support priorities are driven by user demand. If you need a specific platform or GPU variant, please open an issue at [github.com/lloyal-ai/lloyal.node](https://github.com/lloyal-ai/lloyal.node).
