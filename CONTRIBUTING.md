# Contributing to lloyal.node

Thank you for your interest in contributing to lloyal.node! This guide covers development setup, testing, and the release process.

## Development Setup

### Prerequisites

- Node.js 22 or 24 (LTS)
- C++20 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.18+
- Git

### Clone with Submodules

lloyal.node uses git submodules for development:

```bash
git clone --recursive https://github.com/lloyal-ai/lloyal.node.git
cd lloyal.node
npm install
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### Build from Source

```bash
npm run clean
npm install
npm test
```

This builds:
1. llama.cpp (via `scripts/build-llama.sh`)
2. Native Node.js addon (via node-gyp)
3. Runs the test suite

## Testing

```bash
# Run all tests
npm test

# Clean build and test
npm run clean && npm install && npm test
```

### Test Models

The `models/` directory is not tracked in git (too large). Download test models automatically:

```bash
npm run download-models
```

This downloads:
- `SmolLM2-1.7B-Instruct-Q4_K_M.gguf` (1.0GB) - Text generation tests
- `nomic-embed-text-v1.5.Q4_K_M.gguf` (80MB) - Embedding tests

Models are cached in CI to avoid re-downloading on every run.

## Updating Dependencies

lloyal.node vendors its dependencies (`liblloyal` and `llama.cpp`) for npm distribution, since npm doesn't support git submodules.

### Workflow

**Step 1: Update Submodules**

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

**Step 2: Vendor the Updates**

```bash
# Copy submodule sources to vendor/
npm run update-vendors

# Or update specific dependency
npm run update-vendors liblloyal
npm run update-vendors llama.cpp
```

This script:
- Copies files from submodules to `vendor/`
- Records commit hashes in `vendor/VERSIONS.json`
- Creates README in each vendor directory

**Step 3: Test with Vendored Sources**

```bash
# Clean and rebuild to verify vendored sources work
npm run clean
npm install
npm test
```

**Step 4: Commit Vendor Updates**

```bash
git add vendor/
git commit -m "chore: vendor liblloyal and llama.cpp"
```

**Step 5: Check for Breaking Changes**

```bash
cd llama.cpp
git log --oneline <old-commit>..<new-commit>

cd ../liblloyal
git log --oneline <old-commit>..<new-commit>
```

Document any breaking changes in release notes.

### Keep Submodules and Vendors in Sync

Always vendor immediately after updating submodules:

```bash
# Update submodules
git submodule update --remote

# Vendor immediately
npm run update-vendors

# Test
npm run clean && npm install && npm test

# Commit together
git add liblloyal llama.cpp vendor/
git commit -m "chore: update dependencies to latest"
```

## Release Process

### Pre-release Checklist

Before publishing to npm:

```bash
# 1. Update submodules to desired versions
git submodule update --remote

# 2. Test with submodules first
npm run clean && npm install && npm test

# 3. Vendor the tested versions
npm run update-vendors

# 4. Test with vendored sources
npm run clean && npm install && npm test

# 5. Update CHANGELOG.md with changes
# 6. Update version in package.json
```

### Publishing

Releases are automated via GitHub Actions. To trigger a release:

```bash
# Create and push a version tag
git tag v0.2.0
git push origin v0.2.0
```

This triggers the `.github/workflows/release.yml` workflow which:
1. Builds prebuilt binaries for all 13 platforms
2. Publishes platform packages to npm (`@lloyal-labs/lloyal.node-*`)
3. Publishes the main package to npm (`lloyal.node`)

### Manual Publishing (if needed)

```bash
# Bump version
npm version patch  # or minor/major

# Publish to npm
npm publish --access public
```

## Project Structure

```
lloyal.node/
├── liblloyal/           # Git submodule (dev only)
├── llama.cpp/           # Git submodule (dev only)
├── vendor/              # Vendored copies (committed, published to npm)
│   ├── liblloyal/
│   ├── llama.cpp/
│   └── VERSIONS.json
├── src/                 # C++ binding code
├── scripts/             # Build and utility scripts
├── docs/                # User-facing documentation
└── .github/workflows/   # CI/CD pipelines
```

## Code Style

- **C++**: Follow liblloyal conventions (see `liblloyal/CONTRIBUTING.md`)
- **TypeScript**: Run `npm run lint` before committing
- **Commit messages**: Use conventional commits (`feat:`, `fix:`, `chore:`, etc.)

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Make your changes and test thoroughly
4. Commit your changes (`git commit -m 'feat: add amazing feature'`)
5. Push to your fork (`git push origin feat/amazing-feature`)
6. Open a Pull Request

## Questions?

- Open an issue on GitHub
- Check existing issues and PRs
- Review the [distribution documentation](./docs/distribution.md)

---

**License:** Apache 2.0
