#!/usr/bin/env node
/**
 * Initialize git submodules during npm install
 *
 * This is necessary because npm doesn't automatically initialize submodules
 * when installing from GitHub URLs. Without this, llama.cpp/ and liblloyal/
 * won't exist, causing build failures.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');

console.log('[init-submodules] Checking for git submodules...');

// Check if we're in a git repository
const isGitRepo = fs.existsSync(path.join(ROOT, '.git'));

if (!isGitRepo) {
  console.log('[init-submodules] Not a git repository, skipping submodule initialization');
  process.exit(0);
}

// Check if submodules are already initialized
const llamaCppExists = fs.existsSync(path.join(ROOT, 'llama.cpp/.git'));
const libloyalExists = fs.existsSync(path.join(ROOT, 'liblloyal/.git'));

if (llamaCppExists && libloyalExists) {
  console.log('[init-submodules] ✓ Submodules already initialized');
  process.exit(0);
}

// Initialize submodules
console.log('[init-submodules] Initializing git submodules...');
try {
  execSync('git submodule update --init --recursive', {
    cwd: ROOT,
    stdio: 'inherit'
  });
  console.log('[init-submodules] ✓ Submodules initialized successfully');
} catch (error) {
  console.error('[init-submodules] Failed to initialize submodules:', error.message);
  console.error('[init-submodules] Please run manually: git submodule update --init --recursive');
  process.exit(1);
}
