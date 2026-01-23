# Distribution Strategy for lloyal.node

> **Purpose:** This document outlines how to package, publish, and distribute lloyal.node as a native Node.js module with complex dependencies.

---

## Table of Contents

- [The Distribution Challenge](#the-distribution-challenge)
- [Strategy Overview](#strategy-overview)
- [Phase 1: Build from Source](#phase-1-build-from-source)
- [Phase 2: Core Platform Prebuilts](#phase-2-core-platform-prebuilts)
- [Phase 3: Full Platform Matrix](#phase-3-full-platform-matrix)
- [Implementation Guide](#implementation-guide)
- [Publishing Workflow](#publishing-workflow)
- [Version Management](#version-management)

---

## The Distribution Challenge

### Dependency Structure

lloyal.node is an N-API binding with a complex dependency chain:

```
lloyal.node (N-API binding)
    ↓ C++ includes
liblloyal (header-only library, vendored from git submodule)
    ↓ links against
llama.cpp (C++ inference engine, vendored from git submodule)
    ↓ compiles to (platform-specific)
macOS:   libllama.dylib (shared library with Metal support)
Linux:   libllama.so (shared library with OpenMP)
Windows: llama.dll + ggml*.dll (multiple DLLs)
```

### Key Problems

**1. Git Submodules & npm**

npm does not initialize git submodules when installing packages:
- Installing from GitHub: `npm install github:org/repo` clones the repo but ignores `.gitmodules`
- Installing from npm registry: No `.git` directory exists at all
- Result: `liblloyal/` and `llama.cpp/` directories are empty, build fails

**2. Build Complexity**

Users must compile C++ on installation:
- Requires: C++20 compiler, CMake, node-gyp, Python
- Platform-specific toolchains (MSVC on Windows, GCC/Clang on Linux/macOS)
- Build time: 5-15 minutes on first install
- High failure rate on non-standard environments

**3. Platform & GPU Fragmentation**

llama.cpp supports multiple acceleration backends:
- **Metal** (macOS): Built-in GPU acceleration
- **CUDA** (NVIDIA): Requires CUDA toolkit
- **Vulkan** (cross-platform): Requires Vulkan SDK
- **CPU-only**: No dependencies, slower inference

Each backend requires a different build, can't ship a single binary supporting all.

**4. Build Environment Variations**

Even on the same OS/arch, builds vary:
- Different compiler versions (GCC 9 vs 13)
- Different CUDA versions (11.x vs 12.x)
- Different CPU features (AVX2, AVX-512, NEON)
- Different system libraries (glibc versions on Linux)

---

## Strategy Overview

### Three-Phase Approach

| Phase | Status | Distribution | User Install Time | GPU Support |
|-------|--------|--------------|-------------------|-------------|
| **1: Source** | ✅ **COMPLETE** (v0.1.x) | Vendored sources on npm | 5-15 minutes | Auto-detect (Metal/CPU) |
| **2: Core Prebuilts** | 📋 Planned (v0.5.x+) | 3 common platforms | <1 minute | CPU + Metal |
| **3: Full Matrix** | 📋 Future (v1.x+) | 10+ platform/GPU packages | <1 minute | All variants |

### Design Principles

1. **Progressive Enhancement**: Start simple, add complexity only when justified
2. **Graceful Degradation**: Prebuilts fail → fallback to source build
3. **Platform Detection**: Use npm's `os` and `cpu` fields for automatic selection
4. **Version Synchronization**: All platform packages match main package version

---

## Phase 1: Build from Source (Vendored) ✅ COMPLETE

### Overview

**Status:** ✅ Implemented and tested (v0.1.0)
**Audience:** Early adopters, developers, contributors
**Timeline:** v0.1.0 - v0.4.x
**Distribution:** npm registry with vendored submodule sources

**Verified Platforms:**
- ✅ Linux (ubuntu-latest) - Node 18, 20, 22
- ✅ macOS (macos-14) - Node 18, 20, 22
- ✅ Windows (windows-latest) - Node 18, 20, 22

**Test Coverage:** 15 tests per platform (11 API + 4 E2E validation tests)

### The Git Submodules Problem

lloyal.node uses git submodules for dependencies (liblloyal, llama.cpp). **npm does not and will not support git submodules:**

- Installing from npm: Package is a tarball, no `.git` directory
- Installing from GitHub: npm clones repo but ignores `.gitmodules`
- Result: Submodule directories are empty, build fails

**Attempted Solution (Doesn't Work):**
Adding a `preinstall` script to run `git submodule update --init --recursive` fails because:
1. npm cache copies files to temp directory before install scripts
2. Submodules aren't copied, so directories are empty
3. Script runs but has no effect

### Solution: Vendor Submodule Sources

**Include submodule source code directly in npm package:**

```json
{
  "name": "lloyal.node",
  "version": "0.1.0",
  "main": "lib/index.js",
  "gypfile": true,
  "scripts": {
    "prepare": "bash scripts/build-llama.sh",
    "install": "bash scripts/build-llama.sh && node scripts/setup-headers.js && node-gyp rebuild"
  },
  "files": [
    "lib/",
    "src/",
    "scripts/",
    "binding.gyp",
    "vendor/"
  ]
}
```

### How It Works

**When end users install from npm:**

```bash
npm install lloyal.node

# Only the 'install' script runs:
install → Build llama.cpp + Setup headers + node-gyp rebuild
```

**When developers work locally or before publishing:**

```bash
npm install  # In the package directory itself

# Both scripts run:
1. prepare → Build llama.cpp for platform (bash scripts/build-llama.sh)
2. install → Build llama.cpp + Setup headers + node-gyp rebuild
```

**Note:** The `prepare` script is kept for Phase 2 (prebuilt binaries). In CI/CD, it will build llama.cpp before packaging prebuilt binaries. For Phase 1, only the `install` script matters for end users.

**User workflow:**
1. npm downloads tarball (~50MB with vendored sources)
2. npm extracts to node_modules/lloyal.node
3. `install` script builds llama.cpp static libraries/frameworks
4. `install` script creates header symlinks and compiles N-API binding
5. Total time: 5-15 minutes

### Publishing Workflow

**Before publishing, sync submodules:**

```bash
# Update submodules to latest
git submodule update --remote

# Or update to specific commits
cd liblloyal && git checkout <commit> && cd ..
cd llama.cpp && git checkout <commit> && cd ..

# Commit submodule updates
git add liblloyal llama.cpp
git commit -m "chore: update submodules"

# Pack to verify contents
npm pack
tar -tzf lloyal.node-*.tgz | grep -E "(liblloyal|llama.cpp)"
# Should show vendored source files

# Publish
npm publish
```

**Important:** Vendored sources are a **snapshot** of submodules at publish time. Users get the exact versions you tested.

### Pros & Cons

**Pros:**
- Simple to implement (no CI/CD needed)
- Supports all platforms/architectures (if they can compile)
- GPU auto-detection works (Metal, CUDA if installed)
- Full control over build flags

**Cons:**
- Slow install (5-15 min compilation)
- High failure rate (missing compilers, toolchains)
- Requires build tools on user machine
- Poor developer experience

### When to Use

- Development and testing
- Early alpha/beta releases
- Platforms without prebuilt support
- Users needing custom build flags

---

## Phase 2: Core Platform Prebuilts ✅ COMPLETE

### Overview

**Status:** ✅ Implemented (v0.1.0)
**Audience:** Production users on common x64 platforms
**Distribution:** 7 npm packages covering 80%+ of developers

### Platform Packages (Implemented)

| Package | Platform | Arch | GPU | Status |
|---------|----------|------|-----|--------|
| `@lloyal/lloyal.node-darwin-arm64` | macOS | arm64 | Metal | ✅ Working |
| `@lloyal/lloyal.node-darwin-x64` | macOS | x64 | CPU | ✅ Working |
| `@lloyal/lloyal.node-linux-x64` | Linux | x64 | CPU | ✅ Working |
| `@lloyal/lloyal.node-linux-x64-cuda` | Linux | x64 | CUDA 12.2 | ✅ Working |
| `@lloyal/lloyal.node-linux-x64-vulkan` | Linux | x64 | Vulkan | ✅ Working |
| `@lloyal/lloyal.node-win32-x64` | Windows | x64 | CPU | ✅ Working |
| `@lloyal/lloyal.node-win32-x64-cuda` | Windows | x64 | CUDA 12.2 | ✅ Working |

**Total coverage:** ~80% of developers with instant install

**Note:** Original Phase 2 plan was 3 packages, but we exceeded expectations by implementing 7 packages including GPU variants.

### Architecture

**Main Package (`lloyal.node`):**
```json
{
  "name": "lloyal.node",
  "version": "0.5.0",
  "optionalDependencies": {
    "@lloyal/lloyal.node-darwin-arm64": "0.5.0",
    "@lloyal/lloyal.node-linux-x64": "0.5.0",
    "@lloyal/lloyal.node-win32-x64": "0.5.0"
  },
  "scripts": {
    "install": "node scripts/install.js"
  }
}
```

**Platform Package (`@lloyal/lloyal.node-darwin-arm64`):**
```json
{
  "name": "@lloyal/lloyal.node-darwin-arm64",
  "version": "0.5.0",
  "os": ["darwin"],
  "cpu": ["arm64"],
  "main": "index.node",
  "files": [
    "index.node",
    "*.dylib"
  ]
}
```

### Install Flow

```javascript
// scripts/install.js
const platform = `${process.platform}-${process.arch}`;
const prebuiltPackage = `@lloyal/lloyal.node-${platform}`;

try {
  // Check if platform-specific package is installed
  require.resolve(prebuiltPackage);
  console.log(`✓ Using prebuilt binary for ${platform}`);
  process.exit(0);
} catch {
  // Fallback to source build
  console.log(`⚠ No prebuilt for ${platform}, building from source...`);
  console.log(`This will take 5-15 minutes.`);

  // Initialize submodules (if git repo)
  require('./init-submodules.js');

  // Build llama.cpp
  require('./build-llama.js');

  // Setup headers and compile N-API binding
  execSync('node scripts/setup-headers.js && node-gyp rebuild', {
    stdio: 'inherit'
  });
}
```

### CI/CD Pipeline

**Workflow:** `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags:
      - v*

jobs:
  build-prebuilts:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        include:
          - os: macos-14
            arch: arm64
            platform: darwin-arm64
          - os: ubuntu-22.04
            arch: x64
            platform: linux-x64
          - os: windows-latest
            arch: x64
            platform: win32-x64

    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          registry-url: 'https://registry.npmjs.org'

      - name: Install dependencies
        run: npm install

      - name: Build native module
        run: npm run build

      - name: Package prebuilt
        run: |
          mkdir -p prebuilds/${{ matrix.platform }}
          cp build/Release/*.node prebuilds/${{ matrix.platform }}/
          if [ "${{ runner.os }}" = "macOS" ]; then
            cp build/Release/*.dylib prebuilds/${{ matrix.platform }}/ || true
          fi

      - name: Create platform package
        run: |
          node scripts/create-platform-package.js \
            ${{ matrix.platform }} \
            ${{ github.ref_name }}

      - name: Publish platform package
        working-directory: packages/lloyal.node-${{ matrix.platform }}
        run: npm publish --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}

  publish-main:
    needs: build-prebuilts
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          registry-url: 'https://registry.npmjs.org'

      - name: Update package versions
        run: node scripts/sync-versions.js

      - name: Publish main package
        run: npm publish
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

### Helper Scripts

**scripts/create-platform-package.js:**
```javascript
const fs = require('fs');
const path = require('path');

const [platform, version] = process.argv.slice(2);

const packageJson = {
  name: `@lloyal/lloyal.node-${platform}`,
  version: version.replace('v', ''),
  description: `Native module for lloyal.node (${platform})`,
  main: 'index.node',
  os: [platform.split('-')[0]],
  cpu: [platform.split('-')[1]],
  repository: {
    type: 'git',
    url: 'https://github.com/lloyal-ai/lloyal.node.git'
  },
  license: 'MIT',
  files: ['index.node', '*.dylib', '*.so', '*.dll']
};

const pkgDir = path.join('packages', `lloyal.node-${platform}`);
fs.mkdirSync(pkgDir, { recursive: true });
fs.writeFileSync(
  path.join(pkgDir, 'package.json'),
  JSON.stringify(packageJson, null, 2)
);

// Copy built binary
const prebuiltDir = path.join('prebuilds', platform);
fs.readdirSync(prebuiltDir).forEach(file => {
  fs.copyFileSync(
    path.join(prebuiltDir, file),
    path.join(pkgDir, file)
  );
});

console.log(`✓ Created package: ${packageJson.name}@${packageJson.version}`);
```

**scripts/sync-versions.js:**
```javascript
const fs = require('fs');
const path = require('path');

const mainPkg = require('../package.json');
const version = mainPkg.version;

// Update optionalDependencies to match current version
if (mainPkg.optionalDependencies) {
  Object.keys(mainPkg.optionalDependencies).forEach(dep => {
    mainPkg.optionalDependencies[dep] = version;
  });

  fs.writeFileSync(
    'package.json',
    JSON.stringify(mainPkg, null, 2)
  );

  console.log(`✓ Synced all package versions to ${version}`);
}
```

### Pros & Cons

**Pros:**
- Fast install for 70% of users (<1 minute)
- Lower failure rate (no compilation needed)
- Better developer experience
- Still supports all platforms via fallback

**Cons:**
- More complex CI/CD (3 build jobs)
- Multiple npm packages to maintain
- Version synchronization required
- Storage costs for prebuilt binaries

### When to Use

- Production releases (v1.0.0+)
- Public npm distribution
- Targeting broad developer audience

---

## Phase 3: Full Platform Matrix ⚙️ IN PROGRESS (v1.0)

### Overview

**Status:** ⚙️ Implementing (target: v1.0.0)
**Audience:** All users, all platforms, all GPU variants
**Timeline:** v1.0.0
**Distribution:** 10 platform/GPU packages covering 95%+ deployments

### Platform Packages (v1.0 Target)

**Already Implemented (7 packages from Phase 2+):**
- ✅ `@lloyal/lloyal.node-darwin-arm64` (macOS Apple Silicon, Metal)
- ✅ `@lloyal/lloyal.node-darwin-x64` (macOS Intel, CPU)
- ✅ `@lloyal/lloyal.node-linux-x64` (Linux x64, CPU)
- ✅ `@lloyal/lloyal.node-linux-x64-cuda` (Linux x64 + CUDA 12.2)
- ✅ `@lloyal/lloyal.node-linux-x64-vulkan` (Linux x64 + Vulkan)
- ✅ `@lloyal/lloyal.node-win32-x64` (Windows x64, CPU)
- ✅ `@lloyal/lloyal.node-win32-x64-cuda` (Windows x64 + CUDA 12.2)

**New for v1.0 (3 packages):**
- 🔄 `@lloyal/lloyal.node-linux-arm64` (Linux ARM64 - AWS Graviton, Raspberry Pi)
- 🔄 `@lloyal/lloyal.node-linux-arm64-cuda` (Linux ARM64 + CUDA - NVIDIA Jetson)
- 🔄 `@lloyal/lloyal.node-win32-x64-vulkan` (Windows x64 + Vulkan - AMD/Intel GPU)

**Deferred to v1.1+ (2 packages):**
- ⏸️ `@lloyal/lloyal.node-win32-arm64` (Windows ARM64 - awaiting GitHub Actions ARM64 Windows runners)
- ⏸️ `@lloyal/lloyal.node-darwin-x64-vulkan` (macOS Intel + eGPU - negligible use case)

### What Changed from Original Plan

**Original Phase 3 (docs):** 12 packages including win32-arm64, darwin-x64-vulkan

**Actual v1.0 Phase 3:** 10 packages

**Rationale:** 10 packages cover 95%+ of real-world usage. Remaining 2 packages require infrastructure not yet available (win32-arm64) or serve minimal users (darwin-x64-vulkan).

### GPU Variant Installation

**Option 1: Manual Selection**

Users explicitly install GPU variant:

```bash
# Default (CPU or auto-GPU)
npm install lloyal.node

# Force CUDA
npm install lloyal.node
npm install @lloyal/lloyal.node-linux-x64-cuda --save-optional

# Force Vulkan
npm install lloyal.node
npm install @lloyal/lloyal.node-linux-x64-vulkan --save-optional
```

**Option 2: Environment Variable**

```bash
# User sets preference
export LLOYAL_GPU=cuda
npm install lloyal.node

# scripts/install.js reads env var and selects variant
```

**Option 3: Runtime Detection**

```javascript
// On first use, detect available GPU
const gpu = detectGPU(); // 'cuda', 'vulkan', 'metal', 'cpu'

if (!hasVariant(gpu)) {
  console.log(`Installing optimized build for ${gpu}...`);
  await installVariant(gpu);
}
```

### CI/CD Implementation (v1.0)

Build matrix with 10 jobs (see `.github/workflows/release.yml`):

```yaml
strategy:
  matrix:
    include:
      # macOS (2 jobs)
      - { os: macos-14, arch: arm64, gpu: metal, package: darwin-arm64 }
      - { os: macos-13, arch: x64, gpu: cpu, package: darwin-x64 }

      # Linux x64 (3 jobs)
      - { os: ubuntu-22.04, arch: x64, gpu: cpu, package: linux-x64 }
      - { os: ubuntu-22.04, arch: x64, gpu: cuda, package: linux-x64-cuda }
      - { os: ubuntu-22.04, arch: x64, gpu: vulkan, package: linux-x64-vulkan }

      # Linux ARM64 (2 jobs - Docker + QEMU)
      - { os: ubuntu-22.04, arch: arm64, gpu: cpu, package: linux-arm64, docker_platform: linux/arm64 }
      - { os: ubuntu-22.04, arch: arm64, gpu: cuda, package: linux-arm64-cuda, docker_image: nvcr.io/nvidia/l4t-cuda:12.6-devel }

      # Windows (3 jobs)
      - { os: windows-2022, arch: x64, gpu: cpu, package: win32-x64 }
      - { os: windows-2022, arch: x64, gpu: cuda, package: win32-x64-cuda, cuda_version: 12.2.0 }
      - { os: windows-2022, arch: x64, gpu: vulkan, package: win32-x64-vulkan }
```

**Key Implementation Details:**
- **ARM64 builds:** Use Docker + QEMU for cross-compilation (GitHub Actions has no native ARM64 Linux runners)
- **CUDA ARM64:** Use NVIDIA L4T (Linux for Tegra) Docker image for Jetson compatibility
- **Vulkan Windows:** Install LunarG Vulkan SDK during CI build step

### Pros & Cons (v1.0 Implementation)

**Pros:**
- Excellent user experience (instant install + optimal performance)
- Covers 95%+ of real-world deployments
- GPU acceleration out of box (CUDA, Vulkan, Metal)
- ARM64 support (AWS Graviton, Jetson, Raspberry Pi)
- Professional distribution

**Cons:**
- Moderate CI/CD complexity (10 jobs, cross-compilation, GPU toolchains)
- Maintenance burden (10 packages to version/publish)
- Storage/bandwidth costs (50-150MB per package)
- Platform-specific bugs to debug (especially ARM64 QEMU builds)
- Cannot fully test all platforms in CI (no ARM64 hardware runners)

### Success Metrics

**Phase 3 v1.0 considered successful when:**
- All 10 platform packages build successfully in CI
- All 10 packages published to npm registry
- `npm install lloyal.node` works on all 10 platforms
- Community validation on ARM64 hardware (Graviton, Raspberry Pi, Jetson)
- No regression in existing 7 packages
- Commercial product expectations

---

## Implementation Guide

### Setup for Phase 2

**1. Create scripts directory structure:**

```
scripts/
├── init-submodules.js       # Initialize git submodules
├── build-llama.sh           # Build llama.cpp for platform
├── setup-headers.js         # Symlink headers for liblloyal
├── install.js               # Prebuilt or fallback to source
├── create-platform-package.js  # Generate platform package
├── sync-versions.js         # Update all package versions
└── publish-if-need.js       # Conditional publish
```

**2. Update package.json:**

```json
{
  "name": "lloyal.node",
  "version": "0.5.0",
  "optionalDependencies": {
    "@lloyal/lloyal.node-darwin-arm64": "0.5.0",
    "@lloyal/lloyal.node-linux-x64": "0.5.0",
    "@lloyal/lloyal.node-win32-x64": "0.5.0"
  },
  "scripts": {
    "preinstall": "node scripts/init-submodules.js",
    "install": "node scripts/install.js",
    "build": "node-gyp rebuild",
    "sync-versions": "node scripts/sync-versions.js",
    "publish-if-need": "node scripts/publish-if-need.js"
  }
}
```

**3. Create GitHub workflow:**

Copy the Phase 2 CI/CD pipeline to `.github/workflows/release.yml`

**4. Test locally:**

```bash
# Simulate prebuilt install
npm pack
mkdir test-install && cd test-install
npm install ../lloyal.node-*.tgz

# Should either:
# - Use prebuilt (if on supported platform)
# - Build from source (if not)
```

**5. Configure npm token:**

```bash
# Add NPM_TOKEN to GitHub secrets
# Settings → Secrets and variables → Actions → New repository secret
# Name: NPM_TOKEN
# Value: npm_xxxxxxxxxxxxxxxxxxxx
```

---

## Publishing Workflow

### Phase 1: Manual Source Publish

```bash
# 1. Test build locally
npm run build

# 2. Bump version
npm version patch  # or minor/major

# 3. Publish to npm
npm publish

# 4. Tag and push
git push origin main --tags
```

### Phase 2: Automated Prebuilt Publish

```bash
# 1. Commit changes
git add .
git commit -m "feat: add feature X"

# 2. Bump version (triggers sync-versions)
npm version minor  # 0.5.0 → 0.6.0

# 3. Push tag (triggers CI)
git push origin main --tags

# CI automatically:
# - Builds 3 platform packages
# - Publishes each platform package
# - Updates main package optionalDependencies
# - Publishes main package
```

### Pre-Publish Checklist

- [ ] All tests pass
- [ ] Submodules are up to date
- [ ] CHANGELOG.md updated
- [ ] Version bumped in package.json
- [ ] README.md reflects changes
- [ ] Breaking changes documented
- [ ] Platform packages tested locally
- [ ] npm token configured (CI)

### Post-Publish Verification

```bash
# Check npm registry
npm view lloyal.node

# Test installation on different platforms
docker run -it node:20 sh -c "npm install lloyal.node"

# Verify platform packages published
npm view @lloyal/lloyal.node-darwin-arm64
npm view @lloyal/lloyal.node-linux-x64
npm view @lloyal/lloyal.node-win32-x64
```

---

## Version Management

### Semantic Versioning

Follow [semver 2.0.0](https://semver.org/):

- **MAJOR** (0.x → 1.0, 1.x → 2.0): Breaking API changes
- **MINOR** (0.1 → 0.2, 1.0 → 1.1): New features, backwards compatible
- **PATCH** (0.1.0 → 0.1.1): Bug fixes, backwards compatible

### Version Synchronization

**Rule:** All platform packages MUST match main package version

**Enforcement:**

```javascript
// scripts/sync-versions.js (run via npm version hook)
const mainVersion = require('./package.json').version;

// Update optionalDependencies
pkg.optionalDependencies = Object.keys(pkg.optionalDependencies).reduce((acc, dep) => {
  acc[dep] = mainVersion;
  return acc;
}, {});

// Update platform packages (if they exist)
const packages = fs.readdirSync('packages');
packages.forEach(pkg => {
  const pkgJson = require(`./packages/${pkg}/package.json`);
  pkgJson.version = mainVersion;
  fs.writeFileSync(
    `./packages/${pkg}/package.json`,
    JSON.stringify(pkgJson, null, 2)
  );
});
```

**package.json hook:**
```json
{
  "scripts": {
    "version": "npm run sync-versions && git add ."
  }
}
```

### Dependency Updates

**When llama.cpp updates:**

1. Update submodule: `git submodule update --remote llama.cpp`
2. Test build on all platforms
3. If compatible → PATCH version bump
4. If breaking changes → MAJOR version bump
5. Document changes in CHANGELOG.md

**When liblloyal updates:**

1. Update submodule: `git submodule update --remote liblloyal`
2. Test API compatibility
3. Bump version accordingly
4. Update documentation

---

## Best Practices

### 1. Fail Fast, Fail Loudly

```javascript
// In install scripts, detect issues early
if (!hasCompiler()) {
  console.error('ERROR: C++ compiler not found');
  console.error('Install build tools: https://...');
  process.exit(1);
}
```

### 2. Clear Error Messages

```javascript
try {
  require.resolve(prebuiltPackage);
} catch {
  console.log('');
  console.log('⚠ No prebuilt binary available for your platform');
  console.log(`Platform: ${platform}`);
  console.log('');
  console.log('Building from source (5-15 minutes)...');
  console.log('Requirements: C++20 compiler, CMake, node-gyp');
  console.log('Troubleshooting: https://github.com/lloyal-ai/lloyal.node#building');
  console.log('');
}
```

### 3. Provide Escape Hatches

Allow users to force source build:

```bash
# Skip prebuilt, always build from source
npm install lloyal.node --build-from-source

# Or via environment variable
LLOYAL_BUILD_FROM_SOURCE=1 npm install lloyal.node
```

```javascript
// scripts/install.js
if (process.env.LLOYAL_BUILD_FROM_SOURCE === '1' ||
    process.argv.includes('--build-from-source')) {
  console.log('Forcing build from source...');
  buildFromSource();
  process.exit(0);
}
```

### 4. Document Platform Support

**README.md:**
```markdown
## Platform Support

| Platform | Architecture | Support | Install Time | GPU |
|----------|--------------|---------|--------------|-----|
| macOS | Apple Silicon (arm64) | ⚡ Prebuilt | <1 min | Metal |
| macOS | Intel (x64) | 🔨 Source | 5-15 min | CPU |
| Linux | x64 | ⚡ Prebuilt | <1 min | CPU |
| Linux | ARM64 | 🔨 Source | 5-15 min | CPU |
| Windows | x64 | ⚡ Prebuilt | <1 min | CPU |
| Windows | ARM64 | 🔨 Source | 5-15 min | CPU |

⚡ Prebuilt = Download binary
🔨 Source = Compile on install
```

### 5. Test on Real Platforms

Don't rely on CI alone:
- Test prebuilt install on actual macOS/Linux/Windows machines
- Test fallback to source build
- Test with different Node.js versions (18, 20, 22)
- Test with different compilers (GCC, Clang, MSVC)

---

## Troubleshooting

### Common Issues

**Issue:** `Error: Module did not self-register`

**Cause:** Binary compiled for different Node.js version

**Solution:**
```bash
# Rebuild for your Node.js version
npm rebuild lloyal.node
```

**Issue:** `Error: llama.cpp/include not found`

**Cause:** Submodules not initialized

**Solution:**
```bash
git submodule update --init --recursive
npm install
```

**Issue:** Prebuilt fails to load with "symbol not found"

**Cause:** Platform mismatch or incompatible system libraries

**Solution:**
```bash
# Force source build
npm install lloyal.node --build-from-source
```

---

## References

### Similar Projects

Native Node.js modules with prebuilt strategies:
- **sharp**: Image processing (libvips)
- **better-sqlite3**: SQLite bindings
- **canvas**: Cairo canvas API
- **bcrypt**: Password hashing
- **node-sass**: Sass compiler

### Useful Links

- [npm optionalDependencies docs](https://docs.npmjs.com/cli/v10/configuring-npm/package-json#optionaldependencies)
- [node-gyp documentation](https://github.com/nodejs/node-gyp)
- [N-API best practices](https://nodejs.org/api/n-api.html)
- [GitHub Actions matrix builds](https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs)

---

**Document Version:** 1.1
**Last Updated:** 2025-01-16
**Maintainer:** lloyal.node team
