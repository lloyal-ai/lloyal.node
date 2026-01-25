#!/usr/bin/env node
/**
 * Smart installer for lloyal.node
 *
 * Strategy:
 * 1. Check if prebuilt binary exists for this platform
 * 2. If yes, copy to build/Release/ and exit
 * 3. If no, show helpful error with build-from-source instructions
 *
 * Respects LLOYAL_GPU environment variable for GPU variant selection
 */

const fs = require('fs');
const path = require('path');

const PLATFORM = process.platform;
const ARCH = process.arch;
const ROOT = __dirname + '/..';
const BUILD_DIR = path.join(ROOT, 'build', 'Release');

// Logging helpers
const log = (msg) => console.log(`[lloyal.node] ${msg}`);
const error = (msg) => console.error(`[lloyal.node] ❌ ${msg}`);

/**
 * Check if a platform package is installed and has binaries
 */
function findPrebuilt(packageName) {
  try {
    const pkgPath = require.resolve(packageName);
    const binPath = require(packageName); // index.js exports path to binary

    if (fs.existsSync(binPath)) {
      const binDir = path.dirname(binPath);
      return binDir;
    }
  } catch (e) {
    // Package not installed or doesn't export binary path
  }
  return null;
}

/**
 * Copy prebuilt binaries to build/Release/
 */
function installPrebuilt(binDir, packageName) {
  log(`Found prebuilt binaries in ${packageName}`);

  try {
    // Create build/Release directory
    fs.mkdirSync(BUILD_DIR, { recursive: true });

    // Copy all files from bin directory
    const files = fs.readdirSync(binDir);
    files.forEach(file => {
      const src = path.join(binDir, file);
      const dest = path.join(BUILD_DIR, file);

      if (fs.statSync(src).isFile()) {
        fs.copyFileSync(src, dest);
        log(`  ✓ Copied ${file}`);
      }
    });

    log(`✅ Installed prebuilt binaries successfully`);
    process.exit(0);
  } catch (e) {
    error(`Failed to install prebuilt: ${e.message}`);
    // Don't exit - fall through to source build
  }
}

/**
 * Main installation logic
 */
function main() {
  log(`Platform: ${PLATFORM}-${ARCH}`);

  // 1. Check for user-specified GPU variant via environment variable
  if (process.env.LLOYAL_GPU) {
    const gpu = process.env.LLOYAL_GPU.toLowerCase();
    const packageName = `@lloyal-labs/lloyal.node-${PLATFORM}-${ARCH}-${gpu}`;

    log(`LLOYAL_GPU=${gpu}, looking for ${packageName}...`);
    const binDir = findPrebuilt(packageName);

    if (binDir) {
      installPrebuilt(binDir, packageName);
      return; // exit(0) called in installPrebuilt
    } else {
      log(`  ⚠️  Package ${packageName} not found`);
    }
  }

  // 2. Check for GPU variants in priority order
  const gpuVariants = ['cuda', 'vulkan'];
  for (const gpu of gpuVariants) {
    const packageName = `@lloyal-labs/lloyal.node-${PLATFORM}-${ARCH}-${gpu}`;
    const binDir = findPrebuilt(packageName);

    if (binDir) {
      log(`Auto-detected GPU variant: ${gpu}`);
      installPrebuilt(binDir, packageName);
      return; // exit(0) called in installPrebuilt
    }
  }

  // 3. Check for default platform package (CPU or Metal on macOS)
  const defaultPackage = `@lloyal-labs/lloyal.node-${PLATFORM}-${ARCH}`;
  const binDir = findPrebuilt(defaultPackage);

  if (binDir) {
    installPrebuilt(binDir, defaultPackage);
    return; // exit(0) called in installPrebuilt
  }

  // 4. No prebuilt found - error with helpful message
  log('');
  error('No prebuilt binary found for your platform');
  log('');
  log(`  Platform: ${PLATFORM}-${ARCH}`);
  log('');
  log('  Options:');
  log('  1. Install a platform-specific package:');
  log(`     npm install @lloyal-labs/lloyal.node-${PLATFORM}-${ARCH}`);
  log('');
  log('  2. Build from source (requires C++20, CMake 3.18+):');
  log('     git clone --recursive https://github.com/lloyal-ai/lloyal.node.git');
  log('     cd lloyal.node && npm run build');
  log('');
  log('  See: https://github.com/lloyal-ai/lloyal.node#building');
  log('');
  
  process.exit(1);
}

// Run installer
main();
