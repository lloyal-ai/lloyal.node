#!/usr/bin/env node
/**
 * Build script for lloyal.node
 * 
 * Wraps cmake-js with GPU backend detection from LLOYAL_GPU environment variable.
 * 
 * Usage:
 *   npm run build                     # CPU/Metal (auto-detected)
 *   LLOYAL_GPU=cpu npm run build      # CPU only (disables Metal on macOS)
 *   LLOYAL_GPU=cuda npm run build     # CUDA
 *   LLOYAL_GPU=vulkan npm run build   # Vulkan
 *   LLOYAL_GPU=metal npm run build    # Metal (macOS only)
 */

const { execSync } = require('child_process');
const os = require('os');

const PLATFORM = process.platform;
const gpuBackend = process.env.LLOYAL_GPU?.toLowerCase();

// --- BACKEND_DL flavor (LLOYAL_BACKEND_DL=1) ---
// The additive R2-channel build: backends become dlopen-able MODULE libs
// selected at runtime by ggml (dlopen + score), with every CUDA arch and
// every CPU variant in one pack. Never published to npm (lloyal.node#38's
// ~384 MiB ceiling is the whole reason this flavor exists). linux-x64 +
// CUDA only for now — arm64 (GB200/GB10 hosts) is a named follow-on.
const backendDl = process.env.LLOYAL_BACKEND_DL === '1';
if (backendDl) {
  const targetArchDl = (process.env.ARCH || process.arch).toLowerCase();
  if (gpuBackend !== 'cuda' || PLATFORM !== 'linux' || (targetArchDl !== 'x64' && targetArchDl !== 'x86_64')) {
    console.error(
      `[lloyal.node] LLOYAL_BACKEND_DL=1 requires LLOYAL_GPU=cuda on linux-x64 ` +
        `(got gpu=${gpuBackend ?? 'unset'}, platform=${PLATFORM}, arch=${targetArchDl}).`,
    );
    process.exit(1);
  }
}

// Explicit CUDA arch list for the DL flavor — mirror-and-gate: pure
// inheritance of ggml's derivation is unreachable (both the root pin and
// ggml's derivation are NOT-DEFINED-gated), so scripts/dl-archs.js holds an
// explicit mirror (+ the documented 100a-real extension), asserted against
// the built fatbin per-arch in CI (create-dl-pack.js) and drift-checked on
// every llama.cpp sync (--check-archs).
const DL_CUDA_ARCHS = require('./dl-archs').DL_CUDA_ARCHS.join(';');

// Build cmake-js command with appropriate flags
const cmakeFlags = [];

if (backendDl) {
  cmakeFlags.push(
    '--CDLLOYAL_BACKEND_DL=ON',
    '--CDGGML_BACKEND_DL=ON',
    '--CDBUILD_SHARED_LIBS=ON',
    '--CDGGML_CPU_ALL_VARIANTS=ON',
    `"--CDCMAKE_CUDA_ARCHITECTURES=${DL_CUDA_ARCHS}"`,
  );
  console.log('[lloyal.node] Flavor: BACKEND_DL (runtime backend selection)');
  console.log(`[lloyal.node] CUDA archs: ${DL_CUDA_ARCHS}`);
}

if (gpuBackend === 'cuda') {
  cmakeFlags.push('--CDGGML_CUDA=ON');
  console.log('[lloyal.node] GPU backend: CUDA');
} else if (gpuBackend === 'vulkan') {
  cmakeFlags.push('--CDGGML_VULKAN=ON');
  console.log('[lloyal.node] GPU backend: Vulkan');
} else if (gpuBackend === 'metal') {
  cmakeFlags.push('--CDGGML_METAL=ON');
  console.log('[lloyal.node] GPU backend: Metal');
} else if (gpuBackend === 'cpu') {
  // Explicitly disable GPU backends (useful for CI with paravirtualized GPUs)
  if (PLATFORM === 'darwin') {
    cmakeFlags.push('--CDGGML_METAL=OFF');
  }
  console.log('[lloyal.node] GPU backend: CPU only (forced)');
} else if (PLATFORM === 'darwin') {
  // Metal is auto-enabled on macOS by llama.cpp
  console.log('[lloyal.node] GPU backend: Metal (auto-enabled on macOS)');
} else {
  console.log('[lloyal.node] GPU backend: CPU only');
}

// --- CPU ISA baseline (portability floor) ---
// llama.cpp defaults GGML_NATIVE=ON (-march=native), which bakes the *build
// host's* instruction set into the binary. On an AVX-512-capable CI host that
// yields prebuilts that abort with an illegal instruction (0xC000001D) inside
// createContext() on any consumer CPU without AVX-512. Pin an explicit AVX2
// baseline for x64 so prebuilts run on Intel Haswell (2013+) / AMD Zen (2017+)
// and newer. arm64 keeps GGML_NATIVE (its detection probes safely; no AVX).
// Target arch (not host): the Windows-ARM64 cross build sets ARCH=arm64 and a
// cross toolchain (cmake/arm64-cross.cmake), so CMAKE_CROSSCOMPILING is set and
// ggml defaults GGML_NATIVE OFF for cross-compiles (llama.cpp/ggml/CMakeLists.txt,
// GGML_NATIVE_DEFAULT) — no -march=native, nothing to pin.
// See https://github.com/lloyal-ai/hdk/issues/20.
const targetArch = (process.env.ARCH || process.arch).toLowerCase();
if (backendDl) {
  // ALL_VARIANTS owns per-variant ISA flags (each variant resets the full
  // x86 feature set — a global GGML_AVX2 would be ignored, but passing it
  // anyway invites confusion; the baseline x64 variant MUST stay
  // featureless so it always scores 1). GGML_NATIVE stays OFF.
  cmakeFlags.push('--CDGGML_NATIVE=OFF');
  console.log('[lloyal.node] CPU ISA baseline: ALL_VARIANTS (per-variant, baseline x64 featureless)');
} else if (targetArch === 'x64' || targetArch === 'x86_64') {
  cmakeFlags.push('--CDGGML_NATIVE=OFF', '--CDGGML_AVX2=ON');
  console.log('[lloyal.node] CPU ISA baseline: AVX2 (x64 portable floor)');
} else {
  console.log(`[lloyal.node] CPU ISA baseline: native (${targetArch})`);
}

// --- Self-contained prebuilt: drop llama.cpp's HTTPS download client ---
// llama.cpp's `common` links cpp-httplib, and LLAMA_OPENSSL (default ON) makes
// it link the *build host's* OpenSSL by ABSOLUTE path (e.g. Homebrew
// /opt/homebrew/opt/openssl@3/lib/libssl.3.dylib), which doesn't exist on a
// clean user machine → dlopen fails at load and the addon can't be loaded.
// lloyal.node never uses llama.cpp's HTTP layer (models load from local paths),
// so disable it: no external OpenSSL dependency, fully relocatable prebuilt.
// See https://github.com/lloyal-ai/lloyal.node/issues/35.
cmakeFlags.push('--CDLLAMA_OPENSSL=OFF');
console.log('[lloyal.node] LLAMA_OPENSSL=OFF (self-contained: no external OpenSSL)');

const buildCmd = `npx cmake-js compile ${cmakeFlags.join(' ')}`.trim();
console.log(`[lloyal.node] Running: ${buildCmd}`);

try {
  execSync(buildCmd, {
    cwd: __dirname + '/..',
    stdio: 'inherit'
  });
  console.log('[lloyal.node] ✅ Build successful!');
} catch (error) {
  console.error('[lloyal.node] ❌ Build failed');
  process.exit(1);
}
