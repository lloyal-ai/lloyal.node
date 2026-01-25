# lloyal.node Build System Transformation Plan

## Philosophy

> "The person who has a genuinely simpler system - a system made out of genuinely simple parts, is going to be able to affect the greatest change with the least work." — Rich Hickey

We are removing complexity, not patching around it.

## What We're Removing

**Two-phase build architecture:**

```
Phase 1: CMake → build llama.cpp → libllama.so (CMake knows GPU paths)
Phase 2: node-gyp → build addon → link libllama.so (GPU paths lost)
```

This architecture cannot be fixed for GPU builds. CMake discovers CUDA/Vulkan paths, then node-gyp runs as a separate process with no access to that information. The failures are architectural, not bugs.

**Files to delete:**

- `binding.gyp`
- `scripts/build-llama.sh`
- `scripts/copy-dylib.sh`
- Any manual library path injection in CI

## What We're Keeping

1. **Vendoring strategy**: Submodules for development, `vendor/` for npm publish. npm packages must be self-contained.

2. **liblloyal as distinct target**: Separate CMakeLists.txt, separate concerns. Composed via `add_subdirectory`, not flattened.

3. **13 platform packages**: Same distribution architecture, just built correctly.

## Target Architecture

**Single CMake build graph:**

```
CMakeLists.txt
├── add_subdirectory(vendor/llama.cpp)   → llama, ggml targets (with GPU)
├── add_subdirectory(vendor/liblloyal)   → liblloyal target (links llama)
└── add_library(lloyal_node MODULE ...)  → N-API addon (links liblloyal)
```

CMake sees the entire dependency graph. GPU library paths propagate automatically via target properties. Build-time tools (like `vulkan-shaders-gen`) are included in the build closure.

## Technical Requirements

### 1. cmake-js Integration

Replace node-gyp with cmake-js in `package.json`:

```json
{
  "scripts": {
    "install": "cmake-js compile",
    "build": "cmake-js build",
    "rebuild": "cmake-js rebuild",
    "clean": "cmake-js clean"
  },
  "devDependencies": {
    "cmake-js": "^7.3.0"
  }
}
```

### 2. Top-Level CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(lloyal_node)

# cmake-js integration
include_directories(${CMAKE_JS_INC})

# Build llama.cpp with appropriate backend
# GPU flags set via cmake-js CLI or environment
add_subdirectory(vendor/llama.cpp)

# Build liblloyal (links llama internally)
add_subdirectory(vendor/liblloyal)

# Build Node addon
add_library(${PROJECT_NAME} MODULE
    src/addon.cpp
    src/context.cpp
    src/model.cpp
    # ... other addon sources
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    liblloyal      # Brings llama, ggml transitively
    ${CMAKE_JS_LIB}
)

# Node addon naming convention
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
)
```

### 3. CUDA Linking

Use CMake imported targets, not manual flags:

```cmake
# In llama.cpp's CMakeLists.txt (already done upstream)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(ggml-cuda PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
)
```

These propagate transitively. No manual `-L` paths needed.

### 4. Vulkan Build Closure

The Vulkan backend builds `vulkan-shaders-gen` as a build-time tool via `ExternalProject_Add`. When llama.cpp is built via `add_subdirectory`, this happens automatically within the same CMake invocation.

**Fix vendor script** to include all necessary sources:

- `ggml/src/ggml-vulkan/vulkan-shaders-gen.cpp`
- `ggml/src/ggml-vulkan/vulkan-shaders/*.comp`
- Any headers the generator needs

Better: audit `ggml/src/ggml-vulkan/CMakeLists.txt` for the complete file list rather than guessing.

### 5. CI Workflow Changes

**Current (broken):**

```yaml
- run: bash scripts/build-llama.sh
- run: node-gyp rebuild
```

**New:**

```yaml
- run: npx cmake-js compile --CDLLAMA_CUDA=ON
# or
- run: npx cmake-js compile --CDLLAMA_VULKAN=ON
# or (CPU only)
- run: npx cmake-js compile
```

cmake-js passes `-D` flags through to CMake. GPU backend selection happens at configure time, same CMake invocation.

### 6. Platform-Specific Builds

| Platform           | cmake-js flags        |
| ------------------ | --------------------- |
| darwin-arm64       | `--CDLLAMA_METAL=ON`  |
| darwin-x64         | (none, CPU only)      |
| linux-x64          | (none, CPU only)      |
| linux-x64-cuda     | `--CDLLAMA_CUDA=ON`   |
| linux-x64-vulkan   | `--CDLLAMA_VULKAN=ON` |
| linux-arm64        | (none, CPU only)      |
| linux-arm64-cuda   | `--CDLLAMA_CUDA=ON`   |
| linux-arm64-vulkan | `--CDLLAMA_VULKAN=ON` |
| win32-x64          | (none, CPU only)      |
| win32-x64-cuda     | `--CDLLAMA_CUDA=ON`   |
| win32-x64-vulkan   | `--CDLLAMA_VULKAN=ON` |
| win32-arm64        | (none, CPU only)      |
| win32-arm64-vulkan | `--CDLLAMA_VULKAN=ON` |

### 7. External Reuse (Future)

For C++ consumers who want to use liblloyal directly:

```cmake
# In vendor/liblloyal/CMakeLists.txt
install(TARGETS liblloyal EXPORT liblloyal-targets)
install(EXPORT liblloyal-targets
    FILE liblloyal-config.cmake
    NAMESPACE lloyal::
    DESTINATION lib/cmake/liblloyal
)
```

Then external projects can:

```cmake
find_package(liblloyal REQUIRED)
target_link_libraries(my_app PRIVATE lloyal::liblloyal)
```

This is separate from how lloyal.node builds internally. Both use the same CMakeLists.txt, different consumption patterns.

## Deliverables

### Day 1-2: Core Migration

- [ ] Write top-level `CMakeLists.txt` with `add_subdirectory` chain
- [ ] Update `vendor/liblloyal/CMakeLists.txt` if needed for target composition
- [ ] Replace node-gyp with cmake-js in `package.json`
- [ ] Delete `binding.gyp`, `build-llama.sh`, `copy-dylib.sh`
- [ ] Test CPU build locally on one platform

### Day 2-3: GPU Backends

- [ ] Audit `ggml/src/ggml-vulkan/CMakeLists.txt` for complete file list
- [ ] Update vendor script with missing Vulkan sources
- [ ] Test CUDA build locally (if you have NVIDIA hardware)
- [ ] Test Vulkan build locally
- [ ] Verify CMake imported targets propagate correctly

### Day 3-4: CI and All Platforms

- [ ] Update `.github/workflows/release.yml` for cmake-js
- [ ] Remove manual CUDA/Vulkan path injection
- [ ] Test all 13 platform builds in CI
- [ ] Fix any platform-specific issues (Windows generator output paths, etc.)
- [ ] Update `CONTRIBUTING.md` with new build commands

### Documentation

- [ ] Update README build instructions
- [ ] Update CONTRIBUTING.md development setup
- [ ] Document cmake-js flags for GPU variants

## Success Criteria

1. **All 13 platforms build in CI** — no failures
2. **Single build command per platform** — `npx cmake-js compile [flags]`
3. **No manual path threading** — CMake propagates everything
4. **Tests pass** — existing test suite works unchanged
5. **npm publish works** — vendored sources are complete and self-contained

## What This Unlocks

- GPU builds work reliably
- Adding new backends (e.g., ROCm) becomes straightforward
- Contributors have simpler onboarding (one build system)
- CI is faster (no two-phase builds)
- Future changes are lower risk (simpler system, fewer moving parts)
