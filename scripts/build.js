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

// Build cmake-js command with appropriate flags
const cmakeFlags = [];

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
if (targetArch === 'x64' || targetArch === 'x86_64') {
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
