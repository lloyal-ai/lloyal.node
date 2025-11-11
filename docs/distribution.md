# Distribution Strategy for liblloyal-node

> **Purpose:** This document outlines how to package, publish, and distribute liblloyal-node as a native Node.js module with complex dependencies.

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

liblloyal-node is an N-API binding with a complex dependency chain:

```
liblloyal-node (N-API binding)
    ↓ C++ includes
liblloyal (header-only library, git submodule)
    ↓ links against
llama.cpp (C++ inference engine, git submodule)
    ↓ compiles to
libllama.a + libggml.a (static libraries)
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
| **1: Source** | v0.1.x | Source-only on npm | 5-15 minutes | Auto-detect |
| **2: Core Prebuilts** | v0.5.x+ | 3 common platforms | <1 minute | CPU + Metal |
| **3: Full Matrix** | v1.x+ | 10+ platform/GPU packages | <1 minute | All variants |

### Design Principles

1. **Progressive Enhancement**: Start simple, add complexity only when justified
2. **Graceful Degradation**: Prebuilts fail → fallback to source build
3. **Platform Detection**: Use npm's `os` and `cpu` fields for automatic selection
4. **Version Synchronization**: All platform packages match main package version

---

## Phase 1: Build from Source

### Overview

**Audience:** Early adopters, developers, contributors
**Timeline:** v0.1.0 - v0.4.x
**Distribution:** npm registry, source-only package

### How It Works

```bash
npm install liblloyal-node

# Triggers lifecycle scripts:
1. preinstall  → Initialize git submodules (if .git exists)
2. prepare     → Build llama.cpp for platform
3. install     → Setup headers + node-gyp rebuild
```

### Implementation

**package.json:**
```json
{
  "name": "liblloyal-node",
  "version": "0.1.0",
  "main": "lib/index.js",
  "gypfile": true,
  "scripts": {
    "preinstall": "node scripts/init-submodules.js",
    "prepare": "bash scripts/build-llama.sh",
    "install": "node scripts/setup-headers.js && node-gyp rebuild"
  },
  "files": [
    "lib/",
    "src/",
    "scripts/",
    "binding.gyp",
    ".gitmodules",
    "liblloyal/",
    "llama.cpp/"
  ]
}
```

**scripts/init-submodules.js:**
```javascript
#!/usr/bin/env node
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const isGitRepo = fs.existsSync(path.join(ROOT, '.git'));

if (!isGitRepo) {
  console.log('[init-submodules] Not a git repository, skipping');
  process.exit(0);
}

const llamaExists = fs.existsSync(path.join(ROOT, 'llama.cpp/.git'));
const libloyalExists = fs.existsSync(path.join(ROOT, 'liblloyal/.git'));

if (llamaExists && libloyalExists) {
  console.log('[init-submodules] ✓ Submodules already initialized');
  process.exit(0);
}

console.log('[init-submodules] Initializing git submodules...');
try {
  execSync('git submodule update --init --recursive', {
    cwd: ROOT,
    stdio: 'inherit'
  });
  console.log('[init-submodules] ✓ Submodules initialized');
} catch (error) {
  console.error('[init-submodules] Failed:', error.message);
  console.error('Please run manually: git submodule update --init --recursive');
  process.exit(1);
}
```

### Limitations

**npm Registry Distribution:**

When published to npm, the package is distributed as a tarball without `.git`:
- `preinstall` script detects no `.git` directory
- Submodule initialization is skipped
- Build fails: "llama.cpp/include not found"

**Solution for npm registry:** Must vendor submodule sources in the package.

**Vendoring Submodules:**

Include submodule contents in npm package:

```json
{
  "files": [
    "lib/",
    "src/",
    "scripts/",
    "binding.gyp",
    "liblloyal/include/**/*.hpp",
    "liblloyal/tests/**/*",
    "llama.cpp/include/**/*.h",
    "llama.cpp/src/**/*.{cpp,h,hpp}",
    "llama.cpp/ggml/include/**/*.h",
    "llama.cpp/ggml/src/**/*.{c,h,cpp}"
  ]
}
```

**Trade-offs:**
- ✅ Works on npm registry (no git required)
- ✅ Users can install without git
- ❌ Large package size (~50MB tarball)
- ❌ Must sync submodules manually before publish

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

## Phase 2: Core Platform Prebuilts

### Overview

**Audience:** Production users on common platforms
**Timeline:** v0.5.0 - v1.0.0
**Distribution:** npm registry with 3 prebuilt packages

### Platform Selection

Target the **top 3 most common developer platforms**:

| Package | Platform | Arch | GPU | Coverage |
|---------|----------|------|-----|----------|
| `@lloyal/liblloyal-node-darwin-arm64` | macOS | arm64 | Metal | ~40% |
| `@lloyal/liblloyal-node-linux-x64` | Linux | x64 | CPU | ~20% |
| `@lloyal/liblloyal-node-win32-x64` | Windows | x64 | CPU | ~10% |

**Total coverage:** ~70% of developers with instant install

**Unsupported platforms** (Linux arm64, macOS x64, Windows arm64): Fallback to source build

### Architecture

**Main Package (`liblloyal-node`):**
```json
{
  "name": "liblloyal-node",
  "version": "0.5.0",
  "optionalDependencies": {
    "@lloyal/liblloyal-node-darwin-arm64": "0.5.0",
    "@lloyal/liblloyal-node-linux-x64": "0.5.0",
    "@lloyal/liblloyal-node-win32-x64": "0.5.0"
  },
  "scripts": {
    "install": "node scripts/install.js"
  }
}
```

**Platform Package (`@lloyal/liblloyal-node-darwin-arm64`):**
```json
{
  "name": "@lloyal/liblloyal-node-darwin-arm64",
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
const prebuiltPackage = `@lloyal/liblloyal-node-${platform}`;

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
        working-directory: packages/liblloyal-node-${{ matrix.platform }}
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
  name: `@lloyal/liblloyal-node-${platform}`,
  version: version.replace('v', ''),
  description: `Native module for liblloyal-node (${platform})`,
  main: 'index.node',
  os: [platform.split('-')[0]],
  cpu: [platform.split('-')[1]],
  repository: {
    type: 'git',
    url: 'https://github.com/lloyal-ai/liblloyal-node.git'
  },
  license: 'MIT',
  files: ['index.node', '*.dylib', '*.so', '*.dll']
};

const pkgDir = path.join('packages', `liblloyal-node-${platform}`);
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

## Phase 3: Full Platform Matrix

### Overview

**Audience:** All users, all platforms, all GPU variants
**Timeline:** v1.x.x+ (mature project with resources)
**Distribution:** 10+ platform/GPU packages

### Platform Packages

**CPU-only (6 packages):**
```
@lloyal/liblloyal-node-darwin-arm64   (macOS Apple Silicon, Metal built-in)
@lloyal/liblloyal-node-darwin-x64     (macOS Intel, CPU only)
@lloyal/liblloyal-node-linux-x64      (Linux x64, CPU only)
@lloyal/liblloyal-node-linux-arm64    (Linux ARM64, CPU only)
@lloyal/liblloyal-node-win32-x64      (Windows x64, CPU only)
@lloyal/liblloyal-node-win32-arm64    (Windows ARM64, CPU only)
```

**GPU variants (6+ packages):**
```
@lloyal/liblloyal-node-linux-x64-cuda     (Linux x64 + CUDA)
@lloyal/liblloyal-node-linux-x64-vulkan   (Linux x64 + Vulkan)
@lloyal/liblloyal-node-linux-arm64-cuda   (Linux ARM64 + CUDA)
@lloyal/liblloyal-node-linux-arm64-vulkan (Linux ARM64 + Vulkan)
@lloyal/liblloyal-node-win32-x64-cuda     (Windows x64 + CUDA)
@lloyal/liblloyal-node-win32-x64-vulkan   (Windows x64 + Vulkan)
```

### GPU Variant Installation

**Option 1: Manual Selection**

Users explicitly install GPU variant:

```bash
# Default (CPU or auto-GPU)
npm install liblloyal-node

# Force CUDA
npm install liblloyal-node
npm install @lloyal/liblloyal-node-linux-x64-cuda --save-optional

# Force Vulkan
npm install liblloyal-node
npm install @lloyal/liblloyal-node-linux-x64-vulkan --save-optional
```

**Option 2: Environment Variable**

```bash
# User sets preference
export LIBLLOYAL_GPU=cuda
npm install liblloyal-node

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

### CI/CD Expansion

Expand build matrix to 12+ jobs:

```yaml
strategy:
  matrix:
    include:
      # CPU variants
      - { os: macos-14, arch: arm64, variant: default }
      - { os: macos-13, arch: x64, variant: default }
      - { os: ubuntu-22.04, arch: x64, variant: default }
      - { os: ubuntu-22.04-arm, arch: arm64, variant: default }
      - { os: windows-latest, arch: x64, variant: default }
      - { os: windows-latest, arch: arm64, variant: default }

      # CUDA variants
      - { os: ubuntu-22.04, arch: x64, variant: cuda, container: nvidia/cuda:12.6 }
      - { os: ubuntu-22.04-arm, arch: arm64, variant: cuda, container: nvidia/cuda:12.6 }
      - { os: windows-latest, arch: x64, variant: cuda, cuda-version: 12.9 }

      # Vulkan variants
      - { os: ubuntu-22.04, arch: x64, variant: vulkan }
      - { os: ubuntu-22.04-arm, arch: arm64, variant: vulkan }
      - { os: windows-latest, arch: x64, variant: vulkan }
```

### Pros & Cons

**Pros:**
- Best user experience (instant install + optimal performance)
- Covers 100% of platforms
- GPU acceleration out of box
- Professional distribution

**Cons:**
- Complex CI/CD (12+ jobs, cross-compilation, GPU toolchains)
- High maintenance burden (12+ packages to version/publish)
- Storage/bandwidth costs ($$$)
- Platform-specific bugs to debug

### When to Use

- Established project with funding/resources
- Large user base demanding GPU support
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
  "name": "liblloyal-node",
  "version": "0.5.0",
  "optionalDependencies": {
    "@lloyal/liblloyal-node-darwin-arm64": "0.5.0",
    "@lloyal/liblloyal-node-linux-x64": "0.5.0",
    "@lloyal/liblloyal-node-win32-x64": "0.5.0"
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
npm install ../liblloyal-node-*.tgz

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
npm view liblloyal-node

# Test installation on different platforms
docker run -it node:20 sh -c "npm install liblloyal-node"

# Verify platform packages published
npm view @lloyal/liblloyal-node-darwin-arm64
npm view @lloyal/liblloyal-node-linux-x64
npm view @lloyal/liblloyal-node-win32-x64
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
  console.log('Troubleshooting: https://github.com/lloyal-ai/liblloyal-node#building');
  console.log('');
}
```

### 3. Provide Escape Hatches

Allow users to force source build:

```bash
# Skip prebuilt, always build from source
npm install liblloyal-node --build-from-source

# Or via environment variable
LIBLLOYAL_BUILD_FROM_SOURCE=1 npm install liblloyal-node
```

```javascript
// scripts/install.js
if (process.env.LIBLLOYAL_BUILD_FROM_SOURCE === '1' ||
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
npm rebuild liblloyal-node
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
npm install liblloyal-node --build-from-source
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

**Document Version:** 1.0
**Last Updated:** 2025-01-11
**Maintainer:** liblloyal-node team
