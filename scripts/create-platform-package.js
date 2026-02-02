#!/usr/bin/env node
/**
 * Create platform-specific package for prebuilt binaries
 *
 * Usage: node scripts/create-platform-package.js <package-name> <os> <arch>
 * Example: node scripts/create-platform-package.js darwin-arm64 macos-14 arm64
 */

const fs = require('fs');
const path = require('path');

const [packageName, osRunner, arch] = process.argv.slice(2);

if (!packageName || !osRunner || !arch) {
  console.error('Usage: node create-platform-package.js <package-name> <os-runner> <arch>');
  console.error('Example: node create-platform-package.js darwin-arm64 macos-14 arm64');
  process.exit(1);
}

const ROOT = path.join(__dirname, '..');
const BUILD_DIR = path.join(ROOT, 'build', 'Release');
const PACKAGES_DIR = path.join(ROOT, 'packages');
const PKG_DIR = path.join(PACKAGES_DIR, packageName);
const BIN_DIR = path.join(PKG_DIR, 'bin');

// Determine OS and CPU for package.json
const OS_MAP = {
  'macos-14': 'darwin',
  'macos-13': 'darwin',
  'ubuntu-22.04': 'linux',
  'windows-2022': 'win32'
};

const osName = OS_MAP[osRunner] || process.platform;

console.log(`\n=== Creating platform package: @lloyal-labs/lloyal.node-${packageName} ===\n`);

// Create directories
fs.mkdirSync(BIN_DIR, { recursive: true });

// Copy binaries
console.log('Copying binaries...');

// N-API binary
const nodeBinary = path.join(BUILD_DIR, 'lloyal.node');
if (!fs.existsSync(nodeBinary)) {
  console.error(`❌ Error: lloyal.node not found at ${nodeBinary}`);
  console.error('Available files in build/Release:');
  if (fs.existsSync(BUILD_DIR)) {
    fs.readdirSync(BUILD_DIR).forEach(f => console.error(`  - ${f}`));
  } else {
    console.error('  (build/Release directory does not exist)');
  }
  process.exit(1);
}

fs.copyFileSync(nodeBinary, path.join(BIN_DIR, 'lloyal.node'));
console.log(`  ✓ Copied lloyal.node`);

// Shared libraries (platform-specific)
// Shared libraries (platform-specific)
if (osName === 'darwin') {
  // Copy all .dylib files (libllama, libggml, libggml-metal, etc.)
  const dylibs = fs.readdirSync(BUILD_DIR).filter(f => f.endsWith('.dylib'));
  if (dylibs.length > 0) {
    dylibs.forEach(dylib => {
      fs.copyFileSync(path.join(BUILD_DIR, dylib), path.join(BIN_DIR, dylib));
      console.log(`  ✓ Copied ${dylib}`);
    });
  } else {
    console.warn(`  ⚠️  No .dylib files found in build/Release`);
  }

  // Copy Metal shaders if present
  const metalFiles = fs.readdirSync(BUILD_DIR).filter(f => f.endsWith('.metallib') || f.endsWith('.metal'));
  metalFiles.forEach(f => {
    fs.copyFileSync(path.join(BUILD_DIR, f), path.join(BIN_DIR, f));
    console.log(`  ✓ Copied ${f}`);
  });

} else if (osName === 'linux') {
  // Copy all .so files including versioned variants (e.g., libllama.so.0, libllama.so.0.0.X)
  // llama.cpp sets SOVERSION, producing versioned names that the binary references at runtime
  const sos = fs.readdirSync(BUILD_DIR).filter(f => /\.so(\.\d+)*$/.test(f));
  if (sos.length > 0) {
    sos.forEach(so => {
      fs.copyFileSync(path.join(BUILD_DIR, so), path.join(BIN_DIR, so));
      console.log(`  ✓ Copied ${so}`);
    });
  } else {
     console.warn(`  ⚠️  No .so files found in build/Release`);
  }
} else if (osName === 'win32') {
  // Copy all DLLs
  const dlls = fs.readdirSync(BUILD_DIR).filter(f => f.endsWith('.dll'));
  if (dlls.length > 0) {
    dlls.forEach(dll => {
      fs.copyFileSync(
        path.join(BUILD_DIR, dll),
        path.join(BIN_DIR, dll)
      );
      console.log(`  ✓ Copied ${dll}`);
    });
  } else {
    console.warn(`  ⚠️  No DLLs found in build/Release (optional)`);
  }
}

// Create package.json from template
console.log('\nGenerating package.json...');
const mainPackageJson = require(path.join(ROOT, 'package.json'));

// Platform package exports the binary directly (no index.js wrapper)
// This enables runtime dynamic require with automatic fallback:
//   require('@lloyal-labs/lloyal.node-linux-x64') → bin/lloyal.node
const pkgJson = {
  name: `@lloyal-labs/lloyal.node-${packageName}`,
  version: mainPackageJson.version,
  description: `Lloyal native binary for ${packageName}`,
  main: 'bin/lloyal.node',
  os: [osName],
  cpu: [arch],
  files: ['bin/'],
  repository: {
    type: 'git',
    url: 'git+https://github.com/lloyal-ai/lloyal.node.git'
  },
  author: 'lloyal.ai',
  license: 'Apache-2.0'
};

fs.writeFileSync(
  path.join(PKG_DIR, 'package.json'),
  JSON.stringify(pkgJson, null, 2) + '\n'
);
console.log(`  ✓ Created package.json (main: bin/lloyal.node)`);

// Summary
console.log(`\n✅ Platform package created successfully!`);
console.log(`\nPackage: @lloyal-labs/lloyal.node-${packageName}@${pkgJson.version}`);
console.log(`Location: ${PKG_DIR}`);
console.log(`\nContents:`);
fs.readdirSync(BIN_DIR).forEach(f => {
  const stats = fs.statSync(path.join(BIN_DIR, f));
  const sizeMB = (stats.size / 1024 / 1024).toFixed(2);
  console.log(`  - bin/${f} (${sizeMB} MB)`);
});
