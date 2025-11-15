# Vendoring Submodules for npm Distribution

## Problem

liblloyal-node uses git submodules for dependencies:
- `liblloyal/` - Header-only C++ wrapper library
- `llama.cpp/` - Inference engine

**npm does not support git submodules.** When users install from npm:
- Package is a tarball without `.git` directory
- Submodule directories are empty
- Build fails

## Solution: Vendoring

We copy submodule sources into a `vendor/` directory that gets committed to git and published to npm.

### Directory Structure

```
liblloyal-node/
├── liblloyal/          # Git submodule (development only, .gitignore'd after vendoring)
├── llama.cpp/          # Git submodule (development only, .gitignore'd after vendoring)
├── vendor/             # Vendored copies (committed to git, published to npm)
│   ├── liblloyal/      # Copy of liblloyal submodule
│   ├── llama.cpp/      # Copy of llama.cpp submodule
│   └── VERSIONS.json   # Records what was vendored
├── scripts/
│   └── update-vendors.js  # Script to update vendored copies
└── package.json        # files: ["vendor/"] for npm
```

### Workflow

**For Developers (using submodules):**
```bash
git clone --recursive https://github.com/lloyal-ai/lloyal.node.git
npm install  # Uses submodules directly
```

**For npm Users (using vendored sources):**
```bash
npm install liblloyal-node  # Downloads vendored sources from npm
```

**Scripts automatically detect which to use:**
- `vendor/` exists? Use it
- Otherwise, use submodules
- Neither exists? Error with instructions

## Updating Vendored Dependencies

### Step 1: Update Submodules

```bash
# Update to latest upstream
git submodule update --remote

# Or update to specific commits
cd liblloyal && git checkout <commit-hash> && cd ..
cd llama.cpp && git checkout <commit-hash> && cd ..

# Commit submodule updates
git add liblloyal llama.cpp
git commit -m "chore: update submodules to latest"
```

### Step 2: Run Vendor Script

```bash
# Update all vendored dependencies
npm run update-vendors

# Or update specific dependency
npm run update-vendors liblloyal
npm run update-vendors llama.cpp
```

This script:
1. Copies files from submodules to `vendor/`
2. Records commit hashes in `vendor/VERSIONS.json`
3. Creates README in each vendor directory

### Step 3: Test Build

```bash
# Clean and rebuild to test vendor sources
npm run clean
npm install
npm test
```

### Step 4: Commit Vendor Updates

```bash
git add vendor/
git commit -m "chore: vendor liblloyal and llama.cpp"
git push
```

### Step 5: Publish

```bash
# Bump version
npm version patch  # or minor/major

# Publish to npm
npm publish
```

## What Gets Vendored?

Configured in `scripts/update-vendors.js`:

**liblloyal:**
- `include/` - All header files
- `tests/` - Unit tests
- `CMakeLists.txt`
- `LICENSE`

**llama.cpp:**
- `include/` - Public headers
- `src/` - Source files
- `ggml/include/` - GGML headers
- `ggml/src/` - GGML source
- `common/` - Common utilities
- `build-xcframework.sh` - macOS build script
- `CMakeLists.txt`
- `LICENSE`

## Version Tracking

`vendor/VERSIONS.json` records:
```json
{
  "vendoredAt": "2025-01-11T12:00:00.000Z",
  "vendors": {
    "liblloyal": {
      "commit": "abc1234567890...",
      "commitShort": "abc1234",
      "fileCount": 42,
      "vendoredAt": "2025-01-11T12:00:00.000Z"
    },
    "llama.cpp": {
      "commit": "def4567890abc...",
      "commitShort": "def4567",
      "fileCount": 523,
      "vendoredAt": "2025-01-11T12:00:00.000Z"
    }
  }
}
```

This allows tracking:
- What version is currently vendored
- When it was vendored
- How many files were copied

## Why Not Include Submodules in npm Package?

**Attempted solution:** `"files": ["liblloyal/", "llama.cpp/"]`

**Problem:** Git submodules are just `.git` files pointing to parent repo. npm includes the pointer files but not the actual content.

**Result:** Empty directories in published package.

**Solution:** Vendor (copy) the actual files instead.

## Trade-offs

**Pros:**
- ✅ Works on npm registry (no git required)
- ✅ Users get exact tested versions
- ✅ No submodule initialization needed
- ✅ Faster installs (no git operations)

**Cons:**
- ❌ Larger repo size (~50MB vendored sources)
- ❌ Must manually sync when upstream updates
- ❌ Vendored sources committed to git (debatable if bad)

## Best Practices

### Update Vendors Before Releases

**Before publishing to npm:**
```bash
# 1. Update submodules to desired versions
git submodule update --remote

# 2. Test with submodules first
npm run clean && npm install && npm test

# 3. Vendor the tested versions
npm run update-vendors

# 4. Test with vendored sources
npm run clean && npm install && npm test

# 5. Commit and publish
git add vendor/
git commit -m "chore: vendor dependencies for v0.2.0"
npm version minor
npm publish
```

### Document Breaking Changes

When updating llama.cpp or liblloyal, check for breaking changes:

```bash
cd llama.cpp
git log --oneline <old-commit>..<new-commit>

cd ../liblloyal
git log --oneline <old-commit>..<new-commit>
```

Document any breaking changes in CHANGELOG.md.

### Keep Submodules and Vendors in Sync

Always vendor immediately after updating submodules:

```bash
# Update submodules
git submodule update --remote

# Vendor immediately
npm run update-vendors

# Commit together
git add liblloyal llama.cpp vendor/
git commit -m "chore: update dependencies to latest"
```

## Troubleshooting

### "llama.cpp not found" error

**Problem:** Neither submodules nor vendor/ exists

**Solution:**
```bash
# For development (use submodules)
git submodule update --init --recursive

# For publishing (create vendor/)
npm run update-vendors
```

### Vendor script fails with "glob not found"

**Problem:** Missing dev dependency

**Solution:**
```bash
npm install --save-dev glob
```

### Build uses wrong sources

**Problem:** Both submodules and vendor/ exist, unclear which is used

**Solution:** Build scripts prefer `vendor/` if it exists. Delete vendor/ to use submodules:
```bash
rm -rf vendor/
npm run clean && npm install
```

## Migration from Old Approach

**Old:** Submodules only, `preinstall` script tried to init them
**New:** Vendored sources, submodules for development only

**Migration steps:**
1. Delete `scripts/init-submodules.js` (no longer needed)
2. Remove `preinstall` script from package.json
3. Run `npm run update-vendors`
4. Update `.gitignore` if needed
5. Test build with vendored sources
6. Publish new version

---

**Last Updated:** 2025-01-11
