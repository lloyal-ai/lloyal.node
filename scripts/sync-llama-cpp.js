#!/usr/bin/env node
/**
 * Sync llama.cpp submodule to match liblloyal's .llama-cpp-version
 *
 * Single source of truth: liblloyal/.llama-cpp-version contains the tag
 * that the llama.cpp submodule should be checked out at.
 *
 * Usage:
 *   node scripts/sync-llama-cpp.js          # Sync submodule to target tag
 *   node scripts/sync-llama-cpp.js --check  # Validate match (CI mode)
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const VERSION_FILE = path.join(ROOT, 'liblloyal', '.llama-cpp-version');
const LLAMA_CPP_DIR = path.join(ROOT, 'llama.cpp');

const CHECK_ONLY = process.argv.includes('--check');

// --- Read target version ---

if (!fs.existsSync(VERSION_FILE)) {
  console.error('[sync-llama-cpp] Error: liblloyal/.llama-cpp-version not found.');
  console.error('[sync-llama-cpp] Make sure liblloyal submodule is initialized:');
  console.error('[sync-llama-cpp]   git submodule update --init --recursive');
  process.exit(1);
}

const versionFileContent = fs.readFileSync(VERSION_FILE, 'utf8');
const targetVersion = versionFileContent
  .split('\n')
  .filter(line => !line.startsWith('#') && line.trim().length > 0)
  [0]
  ?.trim();

if (!targetVersion) {
  console.error('[sync-llama-cpp] Error: Could not parse version from liblloyal/.llama-cpp-version');
  process.exit(1);
}

console.log(`[sync-llama-cpp] Target llama.cpp version: ${targetVersion}`);

// --- Check llama.cpp submodule exists ---

if (!fs.existsSync(path.join(LLAMA_CPP_DIR, '.git'))) {
  console.error('[sync-llama-cpp] Error: llama.cpp submodule not initialized.');
  console.error('[sync-llama-cpp] Run: git submodule update --init --recursive');
  process.exit(1);
}

// --- Helper ---

function exec(cmd, opts = {}) {
  return execSync(cmd, { cwd: LLAMA_CPP_DIR, encoding: 'utf8', stdio: 'pipe', ...opts }).trim();
}

// --- Get current llama.cpp state ---

const currentSha = exec('git rev-parse HEAD');

// Resolve target tag to SHA (may need to fetch in shallow clones)
let targetSha;
try {
  targetSha = exec(`git rev-parse ${targetVersion}`);
} catch {
  // Tag not available locally — fetch it
  console.log(`[sync-llama-cpp] Tag ${targetVersion} not found locally, fetching...`);
  try {
    exec(`git fetch origin tag ${targetVersion} --no-tags --depth 1`);
    targetSha = exec(`git rev-parse ${targetVersion}`);
  } catch (e) {
    console.error(`[sync-llama-cpp] Error: Tag ${targetVersion} not found in remote.`);
    console.error(`[sync-llama-cpp] Verify tag exists: https://github.com/ggml-org/llama.cpp/releases/tag/${targetVersion}`);
    process.exit(1);
  }
}

const currentShort = currentSha.slice(0, 7);
const targetShort = targetSha.slice(0, 7);

console.log(`[sync-llama-cpp] Current: ${currentShort} (${currentSha})`);
console.log(`[sync-llama-cpp] Target:  ${targetShort} (${targetVersion})`);

if (currentSha === targetSha) {
  console.log(`[sync-llama-cpp] llama.cpp submodule matches ${targetVersion}.`);
  process.exit(0);
}

// --- Mismatch ---

if (CHECK_ONLY) {
  console.error(`\n[sync-llama-cpp] MISMATCH: llama.cpp submodule is at ${currentShort}, expected ${targetVersion} (${targetShort})`);
  console.error(`[sync-llama-cpp] Fix: npm run sync:llama-cpp`);
  process.exit(1);
}

// --- Sync ---

console.log(`[sync-llama-cpp] Checking out ${targetVersion}...`);

try {
  exec(`git checkout ${targetVersion}`);
} catch {
  exec(`git fetch origin tag ${targetVersion} --no-tags --depth 1`);
  exec(`git checkout ${targetVersion}`);
}

const newShort = exec('git rev-parse --short HEAD');
console.log(`[sync-llama-cpp] llama.cpp now at: ${newShort} (${targetVersion})`);
console.log('');
console.log('[sync-llama-cpp] Next steps:');
console.log('  1. Build and test:  npm run build && npm test');
console.log('  2. Stage changes:   git add llama.cpp');
console.log('  3. Commit:          git commit -m "chore(deps): sync llama.cpp to ' + targetVersion + '"');
