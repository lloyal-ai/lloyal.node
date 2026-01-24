#!/usr/bin/env node
/**
 * Vendor git submodules for npm distribution
 *
 * Why: npm doesn't support git submodules. To publish to npm, we must copy
 * submodule sources into the repo as regular files.
 *
 * Usage:
 *   npm run update-vendors          # Update all vendors
 *   npm run update-vendors liblloyal # Update specific vendor
 *
 * This script:
 * 1. Copies files from submodules to vendor/ directory
 * 2. Records commit hashes in vendor/VERSIONS.json
 * 3. Updates .gitignore to exclude vendor/ during development
 * 4. Updates package.json files[] to include vendor/ for publishing
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const ROOT = path.join(__dirname, '..');
const VENDOR_DIR = path.join(ROOT, 'vendor');

// Vendor configurations
const VENDORS = {
  liblloyal: {
    submodule: 'liblloyal',
    include: [
      'include/**/*.hpp',
      'include/**/*.h',
      // Note: tests/ directory NOT included - it contains:
      // - Unit test source files (not needed for npm consumers)
      // - tests/lib/ prebuilt xcframeworks (hundreds of MB)
      // - tests/fixtures/ test models (80MB+)
      // lloyal.node only needs headers for N-API compilation
      'CMakeLists.txt',
      'README.md',
      'LICENSE'
    ]
  },
  'llama.cpp': {
    submodule: 'llama.cpp',
    include: [
      // Core CMake files
      'CMakeLists.txt',
      'cmake/**/*.cmake',
      'cmake/**/*.in',

      // Public headers (what liblloyal includes)
      'include/*.h',

      // llama library source (direct files + models subdirectory)
      'src/*.{cpp,h,hpp}',
      'src/models/*.{cpp,h}',
      'src/CMakeLists.txt',

      // ggml library (core files in ggml/src/)
      'ggml/CMakeLists.txt',
      'ggml/cmake/**/*.cmake',
      'ggml/cmake/**/*.in',
      'ggml/include/*.h',
      'ggml/src/*.{c,cpp,h}',
      'ggml/src/CMakeLists.txt',

      // ggml-cpu backend (includes arch-specific subdirs)
      'ggml/src/ggml-cpu/*.{c,cpp,h}',
      'ggml/src/ggml-cpu/arch/**/*.{c,cpp,h,s,S}',
      'ggml/src/ggml-cpu/amx/**/*.{c,cpp,h}',
      'ggml/src/ggml-cpu/cmake/**/*.cmake',
      'ggml/src/ggml-cpu/kleidiai/**/*.{c,cpp,h}',
      'ggml/src/ggml-cpu/llamafile/**/*.{c,cpp,h}',
      'ggml/src/ggml-cpu/spacemit/**/*.{c,cpp,h}',
      'ggml/src/ggml-cpu/CMakeLists.txt',

      // ggml-metal backend (macOS GPU)
      'ggml/src/ggml-metal/*.{m,mm,cpp,h,metal}',
      'ggml/src/ggml-metal/CMakeLists.txt',

      // ggml-blas backend (Accelerate framework)
      'ggml/src/ggml-blas/*.cpp',
      'ggml/src/ggml-blas/CMakeLists.txt',

      // ggml-cuda backend (NVIDIA GPUs)
      'ggml/src/ggml-cuda/*.{cu,cuh,cpp,h}',
      'ggml/src/ggml-cuda/**/*.{cu,cuh,cpp,h}',
      'ggml/src/ggml-cuda/CMakeLists.txt',

      // ggml-vulkan backend (AMD/Intel GPUs, cross-platform)
      'ggml/src/ggml-vulkan/*.{cpp,h}',
      'ggml/src/ggml-vulkan/**/*.{comp,h,cpp}',
      'ggml/src/ggml-vulkan/CMakeLists.txt',

      'LICENSE'
    ]
  }
};

// Colors for output
const colors = {
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  reset: '\x1b[0m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

/**
 * Get current commit hash of a submodule
 */
function getSubmoduleCommit(submodulePath) {
  try {
    const commit = execSync('git rev-parse HEAD', {
      cwd: path.join(ROOT, submodulePath),
      encoding: 'utf8'
    }).trim();
    return commit;
  } catch (error) {
    log(`✗ Failed to get commit for ${submodulePath}`, 'red');
    log(`  Error: ${error.message}`, 'red');
    process.exit(1);
  }
}

/**
 * Copy files matching glob patterns
 */
function copyFiles(srcDir, destDir, patterns) {
  const glob = require('glob');

  let copiedCount = 0;

  patterns.forEach(pattern => {
    const files = glob.sync(pattern, { cwd: srcDir, nodir: true });

    files.forEach(file => {
      const srcFile = path.join(srcDir, file);
      const destFile = path.join(destDir, file);

      // Create destination directory if needed
      const destFileDir = path.dirname(destFile);
      if (!fs.existsSync(destFileDir)) {
        fs.mkdirSync(destFileDir, { recursive: true });
      }

      // Copy file
      fs.copyFileSync(srcFile, destFile);
      copiedCount++;
    });
  });

  return copiedCount;
}

/**
 * Remove directory recursively
 */
function rmrf(dir) {
  if (fs.existsSync(dir)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

/**
 * Load or create versions tracking file
 */
function loadVersions() {
  const versionsFile = path.join(VENDOR_DIR, 'VERSIONS.json');

  if (fs.existsSync(versionsFile)) {
    return JSON.parse(fs.readFileSync(versionsFile, 'utf8'));
  }

  return { vendoredAt: new Date().toISOString(), vendors: {} };
}

/**
 * Save versions tracking file
 */
function saveVersions(versions) {
  const versionsFile = path.join(VENDOR_DIR, 'VERSIONS.json');
  fs.writeFileSync(versionsFile, JSON.stringify(versions, null, 2));
}

/**
 * Vendor a single submodule
 */
function vendorSubmodule(name, config) {
  log(`\nVendoring ${name}...`, 'cyan');

  const submodulePath = path.join(ROOT, config.submodule);
  const vendorPath = path.join(VENDOR_DIR, name);

  // Check if submodule exists and is populated
  if (!fs.existsSync(submodulePath)) {
    log(`✗ Submodule directory not found: ${config.submodule}`, 'red');
    log(`  Run: git submodule update --init --recursive`, 'yellow');
    process.exit(1);
  }

  const testFile = path.join(submodulePath, 'include');
  if (!fs.existsSync(testFile) && !fs.existsSync(path.join(submodulePath, 'src'))) {
    log(`✗ Submodule appears empty: ${config.submodule}`, 'red');
    log(`  Run: git submodule update --init --recursive`, 'yellow');
    process.exit(1);
  }

  // Get current commit
  const commit = getSubmoduleCommit(config.submodule);
  const commitShort = commit.substring(0, 7);

  log(`  Commit: ${commitShort}`, 'yellow');

  // Remove old vendor directory
  log(`  Cleaning old vendor...`);
  rmrf(vendorPath);

  // Create vendor directory
  fs.mkdirSync(vendorPath, { recursive: true });

  // Copy files
  log(`  Copying files...`);
  const copiedCount = copyFiles(submodulePath, vendorPath, config.include);
  log(`  ✓ Copied ${copiedCount} files`, 'green');

  // Create README in vendor directory
  const readme = `# ${name} (Vendored)

This directory contains vendored sources from the ${name} project.

**Source:** ${config.submodule}/ git submodule
**Commit:** ${commit}
**Vendored:** ${new Date().toISOString()}

**DO NOT EDIT:** Files in this directory are copied from git submodules.
To update, run: npm run update-vendors

See: scripts/update-vendors.js
`;

  fs.writeFileSync(path.join(vendorPath, 'README.md'), readme);

  return { commit, commitShort, fileCount: copiedCount };
}

/**
 * Main execution
 */
function main() {
  const args = process.argv.slice(2);
  const targetVendor = args[0]; // Optional: specific vendor to update

  log('=== Vendoring Submodules for npm Distribution ===', 'cyan');

  // Create vendor directory if needed
  if (!fs.existsSync(VENDOR_DIR)) {
    fs.mkdirSync(VENDOR_DIR, { recursive: true });
  }

  // Load existing versions
  const versions = loadVersions();
  versions.vendoredAt = new Date().toISOString();

  // Determine which vendors to update
  const vendorsToUpdate = targetVendor
    ? { [targetVendor]: VENDORS[targetVendor] }
    : VENDORS;

  if (!vendorsToUpdate || Object.keys(vendorsToUpdate).length === 0) {
    log(`✗ Unknown vendor: ${targetVendor}`, 'red');
    log(`Available vendors: ${Object.keys(VENDORS).join(', ')}`, 'yellow');
    process.exit(1);
  }

  // Vendor each submodule
  Object.entries(vendorsToUpdate).forEach(([name, config]) => {
    const result = vendorSubmodule(name, config);

    versions.vendors[name] = {
      commit: result.commit,
      commitShort: result.commitShort,
      fileCount: result.fileCount,
      vendoredAt: new Date().toISOString()
    };
  });

  // Save versions file
  saveVersions(versions);
  log(`\n✓ Versions recorded in vendor/VERSIONS.json`, 'green');

  // Summary
  log('\n=== Summary ===', 'cyan');
  Object.entries(versions.vendors).forEach(([name, info]) => {
    log(`  ${name}: ${info.commitShort} (${info.fileCount} files)`, 'green');
  });

  log('\n=== Next Steps ===', 'yellow');
  log('1. Review changes: git status');
  log('2. Test build: npm run clean && npm install');
  log('3. Commit: git add vendor/ && git commit -m "chore: update vendored dependencies"');
  log('4. Publish: npm publish');
}

// Check if glob is available
try {
  require('glob');
} catch {
  log('✗ glob package not found', 'red');
  log('  Run: npm install --save-dev glob', 'yellow');
  process.exit(1);
}

main();
