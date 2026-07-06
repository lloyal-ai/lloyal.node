#!/usr/bin/env node
/**
 * Assemble the BACKEND_DL flavor pack for the binaries channel.
 *
 * Produces, under packs/:
 *   linux-x64-dl.tar.zst        — addon + ALL .so* from build/Release
 *                                 (hard-linked libllama/libggml-base +
 *                                 every dlopen-able backend module)
 *   runtime-cuda<M.m>.tar.zst   — companion CUDA runtime redistributables
 *                                 from $CUDA_PATH (cudart, cuBLAS,
 *                                 cuBLASLt, nvJitLink — cuBLAS 12.x links
 *                                 libnvJitLink, so it must ride along or
 *                                 the cuBLAS dlopen fails on a bare box)
 *   manifest.json               — unsigned manifest; the publish worker
 *                                 signs its canonical serialization AFTER
 *                                 the GPU rig passes (the publish act)
 *
 * Tarballs are deterministic (sorted names, zeroed owners, mtime = the
 * HEAD commit's epoch, zstd without content-derived metadata) so a rebuild
 * of the same commit is byte-identical. Requires system `tar` (GNU) and
 * `zstd` — CI installs zstd; fails loud when absent.
 *
 * Modes:
 *   node scripts/create-dl-pack.js               — build the pack
 *   node scripts/create-dl-pack.js --check-archs — drift gate only: assert
 *       scripts/dl-archs.js still mirrors ggml's derivation (run in plain
 *       CI on every llama.cpp sync; no build needed)
 */

const { execSync, execFileSync } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const { DL_CUDA_ARCHS, GGML_MIRRORED_ARCHS } = require('./dl-archs');

const ROOT = path.join(__dirname, '..');
const BUILD_DIR = path.join(ROOT, 'build', 'Release');
const PACKS_DIR = path.join(ROOT, 'packs');
const PLATFORM_TAG = 'linux-x64-dl';

// ── Drift gate: dl-archs.js vs ggml's derivation ─────────────────────

function checkArchsAgainstGgml() {
  const cmakePath = path.join(ROOT, 'llama.cpp', 'ggml', 'src', 'ggml-cuda', 'CMakeLists.txt');
  const text = fs.readFileSync(cmakePath, 'utf8');
  // Extract every arch token ggml appends in its non-native derivation.
  const appended = [...text.matchAll(/list\(APPEND CMAKE_CUDA_ARCHITECTURES ([^)]+)\)/g)]
    .flatMap((m) => m[1].trim().split(/\s+/));
  const mirror = new Set(GGML_MIRRORED_ARCHS);
  const upstream = new Set(appended);

  const missingFromMirror = appended.filter((a) => !mirror.has(a));
  const staleInMirror = GGML_MIRRORED_ARCHS.filter((a) => !upstream.has(a));
  if (missingFromMirror.length || staleInMirror.length) {
    console.error('[create-dl-pack] ARCH DRIFT between scripts/dl-archs.js and ggml derivation:');
    if (missingFromMirror.length) {
      console.error(`  upstream added (absent from mirror): ${missingFromMirror.join(', ')}`);
    }
    if (staleInMirror.length) {
      console.error(`  mirror carries (absent upstream):    ${staleInMirror.join(', ')}`);
    }
    console.error('  Reconcile dl-archs.js (and the pre-requisites design doc) before releasing.');
    process.exit(1);
  }
  console.log('[create-dl-pack] ✓ arch mirror matches ggml derivation (+ documented 100a-real extension)');
}

if (process.argv.includes('--check-archs')) {
  checkArchsAgainstGgml();
  process.exit(0);
}

// ── Helpers ──────────────────────────────────────────────────────────

function fail(msg) {
  console.error(`[create-dl-pack] ERROR: ${msg}`);
  process.exit(1);
}

function sha256File(file) {
  const hash = crypto.createHash('sha256');
  hash.update(fs.readFileSync(file));
  return hash.digest('hex');
}

function which(cmd) {
  try {
    execFileSync('which', [cmd], { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

/** Deterministic .tar.zst of `dir`'s contents (paths relative to dir). */
function tarZst(dir, outFile, mtimeEpoch) {
  execSync(
    `tar --sort=name --owner=0 --group=0 --numeric-owner --mtime=@${mtimeEpoch} ` +
      `-C ${JSON.stringify(dir)} -cf - . | zstd -19 -T0 -q -o ${JSON.stringify(outFile)}`,
    { stdio: 'inherit' },
  );
}

// ── Preconditions ────────────────────────────────────────────────────

if (process.platform !== 'linux') fail('pack assembly runs on linux only (the flavor target)');
if (!which('zstd')) fail('zstd not found — `apt-get install zstd` (CI installs it)');
if (!fs.existsSync(BUILD_DIR)) fail(`no build at ${BUILD_DIR} — run LLOYAL_BACKEND_DL=1 LLOYAL_GPU=cuda npm run build first`);

const cudaPath = process.env.CUDA_PATH;
if (!cudaPath || !fs.existsSync(cudaPath)) {
  fail('CUDA_PATH not set/found — provision-cuda exports it; the companion runtime archive needs it');
}

const pkg = require(path.join(ROOT, 'package.json'));
const llamaCppTag = fs
  .readFileSync(path.join(ROOT, 'liblloyal', '.llama-cpp-version'), 'utf8')
  .trim();
const cudaToolkit = path.basename(cudaPath).replace(/^cuda-/, ''); // e.g. "12.9"
const requiredCudaRuntime = cudaToolkit.split('.').slice(0, 2).join('.');
const mtimeEpoch = execSync('git log -1 --format=%ct', { cwd: ROOT, encoding: 'utf8' }).trim();

// ── Assert the fatbin actually carries the policy archs ─────────────
// The per-arch INDEPENDENT gate: cuobjdump --list-elf for every real,
// --list-ptx for every virtual. "It built" proves nothing about coverage.

const cudaModule = fs
  .readdirSync(BUILD_DIR)
  .find((f) => /^libggml-cuda.*\.so$/.test(f));
if (!cudaModule) {
  const sos = fs.readdirSync(BUILD_DIR).filter((f) => /\.so(\.\d+)*$/.test(f));
  fail(
    `no libggml-cuda*.so module in ${BUILD_DIR} — found ${sos.length} .so file(s)` +
      (sos.length ? `: ${sos.slice(0, 10).join(', ')}` : '') +
      ' (DL modules link into build/bin; scripts/build.js copies them to build/Release post-build)',
  );
}
const cudaModulePath = path.join(BUILD_DIR, cudaModule);

const listElf = execSync(`cuobjdump --list-elf ${JSON.stringify(cudaModulePath)}`, {
  encoding: 'utf8',
});
const listPtx = execSync(`cuobjdump --list-ptx ${JSON.stringify(cudaModulePath)}`, {
  encoding: 'utf8',
});
for (const arch of DL_CUDA_ARCHS) {
  const [code, kind] = arch.split('-');
  const haystack = kind === 'real' ? listElf : listPtx;
  const needle = `sm_${code}`;
  if (!haystack.includes(needle)) {
    fail(`fatbin gate: expected ${needle} (${kind}) in libggml-cuda — arch ${arch} MISSING`);
  }
  console.log(`[create-dl-pack] ✓ ${arch}`);
}
// No unexpected arch-locked entries beyond policy.
const unexpectedA = [...listElf.matchAll(/sm_(\d+a)/g)]
  .map((m) => m[1])
  .filter((code) => !DL_CUDA_ARCHS.includes(`${code}-real`));
if (unexpectedA.length) fail(`unexpected arch-locked SASS beyond policy: ${[...new Set(unexpectedA)].join(', ')}`);

// Expected CPU-variant module presence (exact set varies by toolchain; the
// count floor catches ALL_VARIANTS silently not applying).
const cpuModules = fs.readdirSync(BUILD_DIR).filter((f) => /^libggml-cpu-.*\.so$/.test(f));
if (cpuModules.length < 8) {
  fail(`expected ≥8 libggml-cpu-<variant>.so modules (GGML_CPU_ALL_VARIANTS), found ${cpuModules.length}`);
}
console.log(`[create-dl-pack] ✓ ${cpuModules.length} CPU variant modules`);

// ── Assemble the pack ────────────────────────────────────────────────

fs.rmSync(PACKS_DIR, { recursive: true, force: true });
const stageDir = path.join(PACKS_DIR, 'stage-pack');
fs.mkdirSync(stageDir, { recursive: true });

const nodeBinary = path.join(BUILD_DIR, 'lloyal.node');
if (!fs.existsSync(nodeBinary)) fail('lloyal.node not found in build/Release');
fs.copyFileSync(nodeBinary, path.join(stageDir, 'lloyal.node'));

// Same selection create-platform-package.js uses: every .so incl. versioned
// SOVERSION names — hard-linked deps AND dlopen-able modules together, one
// self-contained delivery unit resolved by $ORIGIN RPATH.
const sos = fs.readdirSync(BUILD_DIR).filter((f) => /\.so(\.\d+)*$/.test(f));
for (const so of sos) {
  fs.copyFileSync(path.join(BUILD_DIR, so), path.join(stageDir, so));
}
console.log(`[create-dl-pack] pack payload: lloyal.node + ${sos.length} shared objects`);

const files = {};
for (const f of fs.readdirSync(stageDir).sort()) {
  files[f] = sha256File(path.join(stageDir, f));
}

const archiveFile = `${PLATFORM_TAG}.tar.zst`;
tarZst(stageDir, path.join(PACKS_DIR, archiveFile), mtimeEpoch);

// ── Companion CUDA runtime archive ───────────────────────────────────
// Redistributable runtime libs from the build toolkit, co-extracted into
// the pack's module dir on boxes whose system runtime fails the gate
// ($ORIGIN RPATH resolves them there with zero plumbing).

const runtimeStage = path.join(PACKS_DIR, 'stage-runtime');
fs.mkdirSync(runtimeStage, { recursive: true });
const cudaLib = path.join(cudaPath, 'lib64');
const RUNTIME_LIB_PATTERNS = [/^libcudart\.so(\.\d+)*$/, /^libcublas\.so(\.\d+)*$/, /^libcublasLt\.so(\.\d+)*$/, /^libnvJitLink\.so(\.\d+)*$/];
let runtimeCount = 0;
for (const f of fs.readdirSync(cudaLib)) {
  if (RUNTIME_LIB_PATTERNS.some((re) => re.test(f))) {
    const src = path.join(cudaLib, f);
    const stat = fs.lstatSync(src);
    // Preserve the SONAME symlink chain as real files (the extractor
    // rejects symlinks by design; duplicate bytes are fine at this size).
    fs.copyFileSync(fs.realpathSync(src), path.join(runtimeStage, f));
    runtimeCount++;
    void stat;
  }
}
if (runtimeCount < 4) fail(`companion runtime: expected cudart/cublas/cublasLt/nvJitLink under ${cudaLib}, found ${runtimeCount} files`);
console.log(`[create-dl-pack] companion runtime payload: ${runtimeCount} libs (CUDA ${requiredCudaRuntime})`);

const runtimeFile = `runtime-cuda${requiredCudaRuntime}.tar.zst`;
tarZst(runtimeStage, path.join(PACKS_DIR, runtimeFile), mtimeEpoch);

// ── Manifest ─────────────────────────────────────────────────────────

const manifest = {
  name: 'lloyal.node-backend-pack',
  version: pkg.version,
  platform: PLATFORM_TAG,
  llamaCppTag,
  cudaToolkit,
  requiredCudaRuntime,
  archs: {
    real: DL_CUDA_ARCHS.filter((a) => a.endsWith('-real')).map((a) => a.replace('-real', '')),
    virtual: DL_CUDA_ARCHS.filter((a) => a.endsWith('-virtual')).map((a) => a.replace('-virtual', '')),
  },
  archive: {
    file: archiveFile,
    sizeBytes: fs.statSync(path.join(PACKS_DIR, archiveFile)).size,
    sha256: sha256File(path.join(PACKS_DIR, archiveFile)),
  },
  runtimeArchive: {
    file: runtimeFile,
    sizeBytes: fs.statSync(path.join(PACKS_DIR, runtimeFile)).size,
    sha256: sha256File(path.join(PACKS_DIR, runtimeFile)),
  },
  files,
};

fs.writeFileSync(path.join(PACKS_DIR, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
fs.rmSync(stageDir, { recursive: true, force: true });
fs.rmSync(runtimeStage, { recursive: true, force: true });

// Also run the drift gate as part of every pack build.
checkArchsAgainstGgml();

console.log('\n[create-dl-pack] ✅ pack assembled:');
for (const f of fs.readdirSync(PACKS_DIR).sort()) {
  const size = (fs.statSync(path.join(PACKS_DIR, f)).size / 1024 / 1024).toFixed(1);
  console.log(`  packs/${f} (${size} MB)`);
}
