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

console.log(`\n=== Creating platform package: @lloyal/lloyal.node-${packageName} ===\n`);

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
if (osName === 'darwin') {
  const dylib = path.join(BUILD_DIR, 'libllama.dylib');
  if (fs.existsSync(dylib)) {
    fs.copyFileSync(dylib, path.join(BIN_DIR, 'libllama.dylib'));
    console.log(`  ✓ Copied libllama.dylib`);
  } else {
    console.warn(`  ⚠️  libllama.dylib not found (optional)`);
  }
} else if (osName === 'linux') {
  const so = path.join(BUILD_DIR, 'libllama.so');
  if (fs.existsSync(so)) {
    fs.copyFileSync(so, path.join(BIN_DIR, 'libllama.so'));
    console.log(`  ✓ Copied libllama.so`);
  } else {
    console.warn(`  ⚠️  libllama.so not found (optional)`);
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
const templatePath = path.join(ROOT, 'packages', 'template', 'package.json');

let pkgJson;
if (fs.existsSync(templatePath)) {
  pkgJson = require(templatePath);
} else {
  // Fallback template if file doesn't exist yet
  pkgJson = {
    name: '@lloyal/lloyal.node-PLATFORM',
    version: '0.0.0',
    description: 'Lloyal native binary for PLATFORM',
    main: 'index.js',
    files: ['bin/', 'index.js'],
    repository: {
      type: 'git',
      url: 'git+https://github.com/lloyal-ai/lloyal.node.git'
    },
    license: 'MIT'
  };
}

// Update with actual values
pkgJson.name = `@lloyal/lloyal.node-${packageName}`;
pkgJson.version = mainPackageJson.version;
pkgJson.description = `Lloyal native binary for ${packageName}`;
pkgJson.os = [osName];
pkgJson.cpu = [arch];

fs.writeFileSync(
  path.join(PKG_DIR, 'package.json'),
  JSON.stringify(pkgJson, null, 2) + '\n'
);
console.log(`  ✓ Created package.json`);

// Create index.js
console.log('\nGenerating index.js...');
const indexJs = `// Platform-specific binary package for ${packageName}
// This file resolves to the native binary in bin/

const path = require('path');

module.exports = path.join(__dirname, 'bin', 'lloyal.node');
`;

fs.writeFileSync(path.join(PKG_DIR, 'index.js'), indexJs);
console.log(`  ✓ Created index.js`);

// Summary
console.log(`\n✅ Platform package created successfully!`);
console.log(`\nPackage: @lloyal/lloyal.node-${packageName}@${pkgJson.version}`);
console.log(`Location: ${PKG_DIR}`);
console.log(`\nContents:`);
fs.readdirSync(BIN_DIR).forEach(f => {
  const stats = fs.statSync(path.join(BIN_DIR, f));
  const sizeMB = (stats.size / 1024 / 1024).toFixed(2);
  console.log(`  - bin/${f} (${sizeMB} MB)`);
});
