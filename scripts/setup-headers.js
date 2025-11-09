#!/usr/bin/env node
/**
 * Setup symlink structure for liblloyal include paths
 *
 * Creates directory structure that matches liblloyal's expectations:
 *   #include <llama/llama.h> -> llama.cpp/include/llama.h
 *   #include <llama/ggml.h>  -> llama.cpp/ggml/include/ggml.h
 *
 * This is necessary because:
 * - liblloyal (external package) expects <llama/...> paths
 * - llama.cpp provides headers at include/llama.h and ggml/include/ggml.h
 * - We can't modify liblloyal (package boundary)
 * - We can't restructure llama.cpp (submodule)
 *
 * Solution: Create symlinks that match expected layout (build-time, gitignored)
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const INCLUDE_DIR = path.join(ROOT, 'include');
const LLAMA_DIR = path.join(INCLUDE_DIR, 'llama');
const GGML_DIR = path.join(INCLUDE_DIR, 'ggml');

console.log('[setup-headers] Setting up include directory symlinks...');

// Clean existing structure
if (fs.existsSync(INCLUDE_DIR)) {
  console.log('[setup-headers] Cleaning existing include/ directory...');
  fs.rmSync(INCLUDE_DIR, { recursive: true, force: true });
}

// Create directory structure
fs.mkdirSync(LLAMA_DIR, { recursive: true });
fs.mkdirSync(GGML_DIR, { recursive: true });

// Helper to create symlink (cross-platform)
function symlinkHeader(target, link) {
  const targetAbs = path.resolve(ROOT, target);
  const linkAbs = path.resolve(ROOT, link);

  if (!fs.existsSync(targetAbs)) {
    console.warn(`[setup-headers] Warning: ${target} does not exist, skipping`);
    return;
  }

  // Use relative path for portability
  const relTarget = path.relative(path.dirname(linkAbs), targetAbs);

  try {
    fs.symlinkSync(relTarget, linkAbs, 'file');
  } catch (err) {
    if (err.code !== 'EEXIST') {
      console.error(`[setup-headers] Failed to link ${link} -> ${target}:`, err.message);
    }
  }
}

// Symlink llama.cpp headers to include/llama/
console.log('[setup-headers] Symlinking llama.cpp headers...');
const llamaHeaders = fs.readdirSync(path.join(ROOT, 'llama.cpp/include'))
  .filter(f => f.endsWith('.h'));

llamaHeaders.forEach(header => {
  symlinkHeader(
    `llama.cpp/include/${header}`,
    `include/llama/${header}`
  );
});

// Symlink ggml headers to include/ggml/
console.log('[setup-headers] Symlinking ggml headers...');
const ggmlHeadersPath = path.join(ROOT, 'llama.cpp/ggml/include');
if (fs.existsSync(ggmlHeadersPath)) {
  const ggmlHeaders = fs.readdirSync(ggmlHeadersPath)
    .filter(f => f.endsWith('.h'));

  ggmlHeaders.forEach(header => {
    symlinkHeader(
      `llama.cpp/ggml/include/${header}`,
      `include/ggml/${header}`
    );
  });

  // Cross-link ggml headers into include/llama/ (for internal includes like #include "ggml.h")
  console.log('[setup-headers] Cross-linking ggml headers into llama/ directory...');
  ggmlHeaders.forEach(header => {
    symlinkHeader(
      `llama.cpp/ggml/include/${header}`,
      `include/llama/${header}`
    );
  });
} else {
  console.warn('[setup-headers] Warning: llama.cpp/ggml/include not found');
}

console.log('[setup-headers] ✓ Include symlinks created successfully');
console.log('[setup-headers]   - include/llama/ -> llama.cpp/include/');
console.log('[setup-headers]   - include/ggml/ -> llama.cpp/ggml/include/');
