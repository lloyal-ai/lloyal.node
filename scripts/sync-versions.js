#!/usr/bin/env node
/**
 * Synchronize all platform package versions with main package version
 *
 * Ensures optionalDependencies in package.json all reference the current version
 * Run before publishing or after `npm version`
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const PKG_JSON_PATH = path.join(ROOT, 'package.json');

console.log('[sync-versions] Synchronizing package versions...\n');

// Read main package.json
const pkg = JSON.parse(fs.readFileSync(PKG_JSON_PATH, 'utf8'));
const version = pkg.version;

console.log(`Main package version: ${version}`);

// Update optionalDependencies
if (pkg.optionalDependencies) {
  console.log('\nUpdating optionalDependencies:');

  Object.keys(pkg.optionalDependencies).forEach(dep => {
    const oldVersion = pkg.optionalDependencies[dep];
    pkg.optionalDependencies[dep] = version;

    if (oldVersion !== version) {
      console.log(`  ${dep}: ${oldVersion} → ${version}`);
    } else {
      console.log(`  ${dep}: ${version} (unchanged)`);
    }
  });

  // Write updated package.json
  fs.writeFileSync(PKG_JSON_PATH, JSON.stringify(pkg, null, 2) + '\n');
  console.log('\n✅ package.json updated');
} else {
  console.log('\n⚠️  No optionalDependencies found in package.json');
}

// Update any existing platform packages in packages/ directory
const PACKAGES_DIR = path.join(ROOT, 'packages');

if (fs.existsSync(PACKAGES_DIR)) {
  const dirs = fs.readdirSync(PACKAGES_DIR).filter(f => {
    const stat = fs.statSync(path.join(PACKAGES_DIR, f));
    return stat.isDirectory() && f !== 'template';
  });

  if (dirs.length > 0) {
    console.log('\nUpdating platform packages:');

    dirs.forEach(dir => {
      const pkgPath = path.join(PACKAGES_DIR, dir, 'package.json');

      if (fs.existsSync(pkgPath)) {
        const platformPkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
        const oldVersion = platformPkg.version;

        platformPkg.version = version;
        fs.writeFileSync(pkgPath, JSON.stringify(platformPkg, null, 2) + '\n');

        if (oldVersion !== version) {
          console.log(`  ${platformPkg.name}: ${oldVersion} → ${version}`);
        } else {
          console.log(`  ${platformPkg.name}: ${version} (unchanged)`);
        }
      }
    });

    console.log('\n✅ Platform packages updated');
  }
}

console.log('\n✅ Version synchronization complete!');
