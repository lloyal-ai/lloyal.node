#!/usr/bin/env node
/**
 * Build script for lloyal.node
 * 
 * Wraps cmake-js with GPU backend detection from LLOYAL_GPU environment variable.
 * 
 * Usage:
 *   npm run build                     # CPU/Metal (auto-detected)
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
} else if (PLATFORM === 'darwin') {
  // Metal is auto-enabled on macOS by llama.cpp
  console.log('[lloyal.node] GPU backend: Metal (auto-enabled on macOS)');
} else {
  console.log('[lloyal.node] GPU backend: CPU only');
}

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
