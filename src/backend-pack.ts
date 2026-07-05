/**
 * Backend pack acquisition — the consumer side of the binaries channel.
 *
 * The BACKEND_DL flavor ships as a signed archive on lloyal.ai R2 (never
 * npm): the addon, ALL backend modules (every CUDA arch + every CPU
 * variant), and a platform-signed manifest. This module implements the
 * three-gate probe and the download → verify → extract → cache pipeline.
 * Selection stays inside ggml (dlopen + score over the module dir); this
 * file is delivery only.
 *
 * Cache: `$XDG_CACHE_HOME|~/.cache/lloyal/backends/<version>-<platform>-<arch>/`
 * — keyed by lloyal.node version so N harnesses on one box share ONE pack.
 * The cache can only exist through explicit consent (a harness prompt) or
 * explicit provisioning (deploy), so `loadBinary` auto-preferring it is
 * honoring a prior decision. A present-but-invalid cache (no completion
 * marker) throws — it never falls through to npm.
 *
 * Fail-loud doctrine: upstream ggml skips broken modules silently in
 * release builds (NDEBUG), so every step here that can lie must throw
 * instead — signature, hash, extraction, and (in loadBinary) the
 * post-load device assertion.
 */

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { PassThrough, Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import type { ReadableStream as WebReadableStream } from 'node:stream/web';
import * as zlib from 'node:zlib';

import { sha256Hex, verifyPlatformSignature } from './verify';

// ── Channel + cache conventions ──────────────────────────────────────

const CHANNEL_BASE = 'https://apps.lloyal.ai/v1/binaries/lloyal.node';
/** Completion marker written after a verified extract — its presence IS cache validity. */
const MARKER_FILE = '.lloyal-pack.json';

/** The one platform the channel currently publishes; arm64 is a named follow-on. */
export type BackendPackPlatform = 'linux-x64-dl';

function packageVersion(): string {
  // dist/backend-pack.js → ../package.json (CJS build).
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  return (require('../package.json') as { version: string }).version;
}

function platformTag(): BackendPackPlatform | null {
  return process.platform === 'linux' && process.arch === 'x64' ? 'linux-x64-dl' : null;
}

/** `~/.cache/lloyal/backends/<version>-<platform>-<arch>/` (sibling of the models cache). */
export function backendPackCacheDir(version = packageVersion()): string {
  const cacheRoot =
    process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache');
  return path.join(cacheRoot, 'lloyal', 'backends', `${version}-${process.platform}-${process.arch}`);
}

/**
 * Synchronous cache resolution for `loadBinary`: the module dir when a
 * COMPLETED pack for this lloyal.node version exists, else null. Cheap by
 * design (one stat + one small JSON read) — runs on every process boot.
 */
export function resolveBackendPackDirSync(version = packageVersion()): string | null {
  const dir = backendPackCacheDir(version);
  const marker = path.join(dir, MARKER_FILE);
  if (!fs.existsSync(marker)) return null;
  try {
    const parsed = JSON.parse(fs.readFileSync(marker, 'utf8')) as { version?: string };
    return parsed.version === version ? dir : null;
  } catch {
    // A present-but-unreadable marker is a corrupt cache: loud, no fallthrough.
    throw new Error(
      `[lloyal.node] Backend pack cache at ${dir} is corrupt (unreadable ${MARKER_FILE}). ` +
        `Delete the directory to re-acquire, or set LLOYAL_BACKEND_DIR to override.`,
    );
  }
}

// ── Manifest ─────────────────────────────────────────────────────────

export interface ArchiveRef {
  file: string;
  sizeBytes: number;
  sha256: string;
}

/** The platform-signed pack manifest (worker: publish-worker src/binaries.ts). */
export interface BackendPackManifest {
  name: 'lloyal.node-backend-pack';
  version: string;
  platform: BackendPackPlatform;
  llamaCppTag: string;
  cudaToolkit: string;
  /** Minimum CUDA runtime (cudart/cuBLAS minor, e.g. "12.9") the modules link against. */
  requiredCudaRuntime: string;
  archs: { real: string[]; virtual: string[] };
  archive: ArchiveRef;
  /** Companion CUDA runtime redistributables — fetched only when gate (c) fails. */
  runtimeArchive?: ArchiveRef;
  files?: Record<string, string>;
}

// ── Probe ────────────────────────────────────────────────────────────

/** Injectable command runner so gate logic is unit-testable without hardware. */
export type CommandRunner = (cmd: string, args: string[]) => { status: number | null; stdout: string };

const defaultRunner: CommandRunner = (cmd, args) => {
  const res = spawnSync(cmd, args, { encoding: 'utf8', timeout: 10_000 });
  return { status: res.status, stdout: res.stdout ?? '' };
};

export interface GpuInfo {
  name: string;
  /** e.g. "9.0" (H100), "8.0" (A100). */
  computeCap: string;
  driverVersion: string;
  /** Max CUDA version the driver can JIT/serve, from nvidia-smi's header (e.g. "12.2"). */
  driverCudaVersion: string | null;
}

export interface BackendPackProbe {
  gpu: GpuInfo | null;
  manifest: BackendPackManifest | null;
  gates: {
    /** (a) device covered by the pack's arch list (real SASS or a JIT-able PTX floor). */
    device: boolean;
    /** (b) real SASS present, OR the driver can JIT the pack's PTX toolkit. */
    driver: boolean;
    /** (c) installed CUDA runtime satisfies the manifest's minimum. */
    runtime: boolean;
  };
  /**
   * Offer-worthy: all gates pass(able) AND the device isn't already served
   * natively by the npm pin (sm_86/89 — the consumer-Ampere/Ada class the
   * standard package fully serves; an 800 MB pack buys them nothing).
   */
  recommended: boolean;
  /** Gate (c) failed ⇒ the companion runtime archive must be co-installed. */
  needsRuntimeArchive: boolean;
  sizeBytes: number;
  runtimeSizeBytes: number;
  /** Legible, one-per-line reasons for the verdict (shown by harnesses). */
  reasons: string[];
}

/** sm caps the npm channel's `86-real` serves natively (same-major minor-forward). */
const NPM_NATIVE_CAPS = new Set(['8.6', '8.9']);

/** "9.0" → "90"; "10.0" → "100"; "12.1" → "121". */
function capCode(computeCap: string): string {
  const [major, minor] = computeCap.split('.');
  return `${major}${minor ?? '0'}`;
}

/** Strip arch-variant suffixes: "100a" → "100". */
function archCode(arch: string): string {
  return arch.replace(/[a-z]+$/, '');
}

export function detectGpu(run: CommandRunner = defaultRunner): GpuInfo | null {
  const q = run('nvidia-smi', [
    '--query-gpu=name,compute_cap,driver_version',
    '--format=csv,noheader',
  ]);
  if (q.status !== 0 || !q.stdout.trim()) return null;
  const [name, computeCap, driverVersion] = q.stdout
    .trim()
    .split('\n')[0]
    .split(',')
    .map((s) => s.trim());
  if (!name || !computeCap) return null;

  // The driver's CUDA ceiling only appears in the plain-invocation banner.
  const banner = run('nvidia-smi', []);
  const cudaMatch = banner.stdout.match(/CUDA Version:\s*(\d+\.\d+)/);
  return {
    name,
    computeCap,
    driverVersion: driverVersion ?? '',
    driverCudaVersion: cudaMatch ? cudaMatch[1] : null,
  };
}

/**
 * Installed CUDA runtime minor version (e.g. "12.2"), from the realpath of
 * libcudart.so.12 in the linker cache. The SONAME is major-versioned, so
 * only the fully-versioned file name reveals the minor — exactly the skew
 * the runtime gate exists to catch. Null = not found / undeterminable.
 */
export function detectCudaRuntime(run: CommandRunner = defaultRunner): string | null {
  const res = run('ldconfig', ['-p']);
  if (res.status !== 0) return null;
  const line = res.stdout.split('\n').find((l) => l.includes('libcudart.so.12'));
  if (!line) return null;
  const libPath = line.split('=>').pop()?.trim();
  if (!libPath) return null;
  try {
    const real = fs.realpathSync(libPath);
    const m = path.basename(real).match(/libcudart\.so\.(\d+)\.(\d+)/);
    return m ? `${m[1]}.${m[2]}` : null;
  } catch {
    return null;
  }
}

function versionGte(a: string, b: string): boolean {
  const pa = a.split('.').map(Number);
  const pb = b.split('.').map(Number);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const x = pa[i] ?? 0;
    const y = pb[i] ?? 0;
    if (x !== y) return x > y;
  }
  return true;
}

async function fetchManifest(
  version: string,
  platform: BackendPackPlatform,
): Promise<{ manifest: BackendPackManifest; bytes: Uint8Array }> {
  const base = `${CHANNEL_BASE}/${version}/${platform}.manifest.json`;
  const [manifestRes, sigRes] = await Promise.all([fetch(base), fetch(`${base}.sig`)]);
  if (!manifestRes.ok || !sigRes.ok) {
    throw new Error(
      `[lloyal.node] Backend pack manifest not available for ${version}/${platform} ` +
        `(${manifestRes.status}/${sigRes.status} from ${base}). ` +
        `The channel publishes only released versions that passed the GPU gate.`,
    );
  }
  const bytes = new Uint8Array(await manifestRes.arrayBuffer());
  const signature = (await sigRes.text()).trim();
  // Signature is over the exact stored bytes — verify BEFORE parsing.
  if (!verifyPlatformSignature(bytes, signature)) {
    throw new Error(
      `[lloyal.node] Backend pack manifest signature verification FAILED for ${base}. Refusing to proceed.`,
    );
  }
  const manifest = JSON.parse(new TextDecoder().decode(bytes)) as BackendPackManifest;
  if (manifest.version !== version || manifest.platform !== platform) {
    throw new Error(
      `[lloyal.node] Backend pack manifest is signed but names ${manifest.version}/${manifest.platform}; ` +
        `expected ${version}/${platform}. Refusing (same-path substitution).`,
    );
  }
  return { manifest, bytes };
}

/**
 * Run all three gates against the pack's manifest BEFORE offering anything
 * (the prompted-download path has no provisioner, so the checks deploy
 * performs at provision time run here instead).
 */
export async function probeBackendPack(opts?: {
  run?: CommandRunner;
  version?: string;
  /** Injectable manifest for tests / pinned CI runs. */
  manifest?: BackendPackManifest;
}): Promise<BackendPackProbe> {
  const run = opts?.run ?? defaultRunner;
  const version = opts?.version ?? packageVersion();
  const reasons: string[] = [];

  const platform = platformTag();
  const gpu = platform ? detectGpu(run) : null;
  if (!platform) reasons.push(`no backend pack is published for ${process.platform}-${process.arch}`);
  else if (!gpu) reasons.push('no NVIDIA GPU detected (nvidia-smi absent or reported none)');

  let manifest = opts?.manifest ?? null;
  if (!manifest && platform && gpu) {
    manifest = (await fetchManifest(version, platform)).manifest;
  }

  if (!gpu || !manifest) {
    return {
      gpu,
      manifest,
      gates: { device: false, driver: false, runtime: false },
      recommended: false,
      needsRuntimeArchive: false,
      sizeBytes: manifest?.archive.sizeBytes ?? 0,
      runtimeSizeBytes: manifest?.runtimeArchive?.sizeBytes ?? 0,
      reasons,
    };
  }

  const code = capCode(gpu.computeCap);
  const hasRealSass = manifest.archs.real.some((a) => archCode(a) === code);
  const jitFloor = manifest.archs.virtual
    .map((a) => Number(archCode(a)))
    .filter((v) => v <= Number(code))
    .sort((a, b) => b - a)[0];

  // (a) device: covered by real SASS or a JIT-able PTX floor.
  const device = hasRealSass || jitFloor !== undefined;
  if (!device) reasons.push(`GPU ${gpu.name} (sm_${code}) is below the pack's arch floor`);

  // (b) driver: real SASS needs no JIT; otherwise the driver must be able
  // to JIT PTX emitted by the pack's toolkit.
  const driver =
    hasRealSass ||
    (gpu.driverCudaVersion !== null &&
      versionGte(gpu.driverCudaVersion, manifest.cudaToolkit.split('.').slice(0, 2).join('.')));
  if (!driver) {
    reasons.push(
      `driver ${gpu.driverVersion} (CUDA ${gpu.driverCudaVersion ?? 'unknown'}) cannot JIT ` +
        `the pack's ${manifest.cudaToolkit} PTX and no native SASS covers sm_${code}`,
    );
  }

  // (c) runtime: installed cudart minor vs the manifest's minimum. Failure
  // is not a decline — the companion runtime archive is the remedy.
  const installedRuntime = detectCudaRuntime(run);
  const runtime = installedRuntime !== null && versionGte(installedRuntime, manifest.requiredCudaRuntime);
  const needsRuntimeArchive = device && driver && !runtime && manifest.runtimeArchive !== undefined;
  if (!runtime) {
    reasons.push(
      `installed CUDA runtime ${installedRuntime ?? 'not found'} < required ${manifest.requiredCudaRuntime}` +
        (needsRuntimeArchive ? ' — the companion runtime archive covers this' : ''),
    );
  }

  const npmNative = NPM_NATIVE_CAPS.has(gpu.computeCap);
  if (npmNative) {
    reasons.push(`GPU ${gpu.name} (sm_${code}) is served natively by the standard npm package`);
  }
  const recommended = device && driver && (runtime || needsRuntimeArchive) && !npmNative;
  if (recommended) {
    reasons.push(
      hasRealSass
        ? `${gpu.name}: pack provides native sm_${code} kernels (npm runs JIT-degraded here)`
        : `${gpu.name}: pack raises the JIT floor to compute_${jitFloor} for this device`,
    );
  }

  return {
    gpu,
    manifest,
    gates: { device, driver, runtime },
    recommended,
    needsRuntimeArchive,
    sizeBytes: manifest.archive.sizeBytes,
    runtimeSizeBytes: needsRuntimeArchive ? manifest.runtimeArchive?.sizeBytes ?? 0 : 0,
    reasons,
  };
}

// ── Download + extract ───────────────────────────────────────────────

export interface EnsureBackendPackOpts {
  onProgress?: (got: number, total: number, file: string) => void;
  run?: CommandRunner;
  version?: string;
  /**
   * Pinned mode — bypass manifest signature discovery and fetch a named
   * archive with a caller-pinned sha256. Used by the CI rig (which tests
   * pre-publish archives) and by deploy manifests that pin the hash
   * client-side. Everything downstream (hash check, extract, marker) is
   * identical to the signed path.
   */
  pinned?: {
    archiveUrl: string;
    sha256: string;
    runtimeArchive?: { url: string; sha256: string };
  };
  /**
   * Override the runtime-archive decision when the caller has ALREADY
   * probed (harness flow: probe once for the offer UI, pass
   * `probe.needsRuntimeArchive` here — avoids a second manifest fetch +
   * nvidia-smi pass). `true` = include, `false` = skip, `undefined` =
   * probe internally. A wrong `false` on a runtime-skewed box yields a
   * pack whose CUDA module cannot dlopen — caught LOUDLY by loadBinary's
   * device assertion, never a silent CPU fallback.
   */
  includeRuntime?: boolean;
}

/**
 * Acquire the backend pack into the shared cache; returns the module dir.
 * Verify → extract to a temp sibling → atomic rename → completion marker.
 * Any failure throws; a partial cache (no marker) is invisible to
 * `loadBinary` and re-acquirable.
 */
export async function ensureBackendPack(opts: EnsureBackendPackOpts = {}): Promise<string> {
  const version = opts.version ?? packageVersion();
  const platform = platformTag();
  if (!platform && !opts.pinned) {
    throw new Error(`[lloyal.node] No backend pack is published for ${process.platform}-${process.arch}.`);
  }

  const existing = resolveBackendPackDirSync(version);
  if (existing) return existing;

  const dir = backendPackCacheDir(version);
  // A destination dir WITHOUT a marker at this point is a crashed install
  // (the only live-racer window is the instant between a winner's rename
  // and its marker write, and a full download cannot fit inside it) —
  // remove the remnant so a crash stays re-acquirable, per the design's
  // crash-anywhere contract. The rename-failure path below stays strict:
  // mid-download races resolve to the winner or a hard error, never a guess.
  if (fs.existsSync(dir)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
  const parent = path.dirname(dir);
  fs.mkdirSync(parent, { recursive: true });
  const staging = fs.mkdtempSync(path.join(parent, `.staging-${version}-`));

  try {
    let archives: { url: string; sha256: string; label: string }[];
    let markerBody: Record<string, unknown>;

    if (opts.pinned) {
      archives = [{ url: opts.pinned.archiveUrl, sha256: opts.pinned.sha256, label: 'backend-pack' }];
      if (opts.pinned.runtimeArchive) {
        archives.push({ ...opts.pinned.runtimeArchive, sha256: opts.pinned.runtimeArchive.sha256, label: 'cuda-runtime' });
      }
      markerBody = { version, source: 'pinned', archiveSha256: opts.pinned.sha256 };
    } else {
      const { manifest } = await fetchManifest(version, platform!);
      const runtimeNeeded =
        opts.includeRuntime ??
        (await probeBackendPack({ run: opts.run, version, manifest })).needsRuntimeArchive;
      const base = `${CHANNEL_BASE}/${version}`;
      archives = [
        { url: `${base}/${manifest.archive.file}`, sha256: manifest.archive.sha256, label: 'backend-pack' },
      ];
      if (runtimeNeeded && manifest.runtimeArchive) {
        archives.push({
          url: `${base}/${manifest.runtimeArchive.file}`,
          sha256: manifest.runtimeArchive.sha256,
          label: 'cuda-runtime',
        });
      }
      markerBody = { version, source: 'channel', manifest };
    }

    for (const { url, sha256, label } of archives) {
      const archivePath = path.join(staging, `${label}.tar.zst`);
      await downloadWithHash(url, archivePath, sha256, label, opts.onProgress);
      // The companion runtime co-extracts into the SAME dir as the pack —
      // the modules' $ORIGIN RPATH resolves siblings, nothing else.
      await extractTarZst(archivePath, staging);
      fs.unlinkSync(archivePath);
    }

    // Publish the cache: rename, then marker. A crash before the rename
    // leaves only invisible staging; a crash between rename and marker
    // leaves a marker-less dir that resolveBackendPackDirSync ignores and
    // the remnant sweep above reclaims on the next ensure — re-acquirable
    // either way, visible to loadBinary only once the marker lands.
    try {
      fs.renameSync(staging, dir);
    } catch (renameErr) {
      // N harnesses cold-starting on one box can race this install (the
      // shared cache invites exactly that). If another process won and
      // completed, use its pack; a dir without a marker (winner mid-flight
      // or crashed) stays a hard error — never guess at a partial install.
      const winner = resolveBackendPackDirSync(version);
      if (winner) {
        fs.rmSync(staging, { recursive: true, force: true });
        return winner;
      }
      throw renameErr;
    }
    fs.writeFileSync(path.join(dir, MARKER_FILE), JSON.stringify(markerBody, null, 2));
    return dir;
  } catch (err) {
    fs.rmSync(staging, { recursive: true, force: true });
    throw err;
  }
}

async function downloadWithHash(
  url: string,
  dest: string,
  expectedSha256: string,
  label: string,
  onProgress?: (got: number, total: number, file: string) => void,
): Promise<void> {
  const res = await fetch(url);
  if (!res.ok || !res.body) {
    throw new Error(`[lloyal.node] Download failed for ${label}: ${res.status} ${url}`);
  }
  const total = Number(res.headers.get('content-length') ?? 0);
  const hash = createHash('sha256');
  let got = 0;
  const meter = new PassThrough();
  meter.on('data', (chunk: Buffer) => {
    hash.update(chunk);
    got += chunk.byteLength;
    onProgress?.(got, total, label);
  });
  // pipeline owns backpressure, error propagation (incl. write-stream I/O
  // failures like disk-full), and teardown of every stage on failure —
  // no hand-rolled drain-wait that could hang after an 'error' event.
  await pipeline(
    Readable.fromWeb(res.body as WebReadableStream<Uint8Array>),
    meter,
    fs.createWriteStream(dest),
  );
  const actual = hash.digest('hex');
  if (actual !== expectedSha256) {
    throw new Error(
      `[lloyal.node] sha256 mismatch for ${label}: expected ${expectedSha256}, got ${actual}. ` +
        `Refusing to install (possible tamper or same-path rollback).`,
    );
  }
}

// ── Minimal streaming ustar extractor ────────────────────────────────
//
// Packs are produced by our own CI (`create-dl-pack.js`: GNU tar, sorted,
// regular files only), so the extractor supports exactly: regular files,
// directories, GNU long names (typeflag 'L'), and skips pax headers. It
// REJECTS symlinks/hardlinks/absolute paths/`..` — fail loud, never a
// surprising filesystem write. Precedent for in-repo tar handling: the
// publish worker's zero-dep ustar walker (tarball-inspect.ts).

export async function extractTarZst(archivePath: string, destDir: string): Promise<void> {
  const source = fs.createReadStream(archivePath);
  // node:zlib zstd (v23.8.0+, backported v22.15.0; engines pins >=24).
  const unzstd = zlib.createZstdDecompress();
  source.on('error', (e) => unzstd.destroy(e));
  source.pipe(unzstd);
  const extract = new TarExtractor(destDir);
  try {
    for await (const chunk of unzstd) {
      extract.push(chunk as Buffer);
    }
    extract.finish();
  } finally {
    extract.close();
  }
}

class TarExtractor {
  private buffer: Buffer = Buffer.alloc(0);
  private pendingLongName: string | null = null;
  private current: { fd: number; remaining: number; padding: number } | null = null;
  /** Padding (and pax) bytes still to discard before the next header. */
  private skip = 0;

  constructor(private readonly destDir: string) {}

  push(chunk: Buffer): void {
    this.buffer = this.buffer.length === 0 ? chunk : Buffer.concat([this.buffer, chunk]);
    this.drain();
  }

  finish(): void {
    this.drain();
    // EOF is only clean when nothing is mid-flight: no open file body, no
    // padding/pax bytes still owed, no dangling GNU long-name, and any
    // leftover buffer is all-zero terminator blocks. Anything else is a
    // truncated archive that would otherwise silently drop trailing entries.
    if (this.current) {
      throw new Error('[lloyal.node] Truncated archive: file body ended mid-entry.');
    }
    if (this.skip > 0) {
      throw new Error('[lloyal.node] Truncated archive: ended inside entry padding.');
    }
    if (this.pendingLongName !== null) {
      throw new Error('[lloyal.node] Truncated archive: GNU long-name without its entry.');
    }
    if (!this.buffer.every((b) => b === 0)) {
      throw new Error('[lloyal.node] Truncated archive: trailing partial header.');
    }
  }

  /** Best-effort fd cleanup on abort paths; safe to call twice. */
  close(): void {
    if (this.current) {
      try {
        fs.closeSync(this.current.fd);
      } catch {
        /* already closed */
      }
      this.current = null;
    }
  }

  private drain(): void {
    for (;;) {
      if (this.skip > 0) {
        const take = Math.min(this.skip, this.buffer.length);
        this.buffer = this.buffer.subarray(take);
        this.skip -= take;
        if (this.skip > 0) return; // need more input
      }

      if (this.current) {
        const take = Math.min(this.current.remaining, this.buffer.length);
        if (take > 0) {
          // write(2) may legally return short counts — loop to completion
          // so `remaining` never over-decrements against bytes on disk.
          let written = 0;
          while (written < take) {
            const n = fs.writeSync(this.current.fd, this.buffer.subarray(written, take));
            if (n <= 0) {
              throw new Error('[lloyal.node] Extraction write made no progress (disk full?).');
            }
            written += n;
          }
          this.buffer = this.buffer.subarray(take);
          this.current.remaining -= take;
        }
        if (this.current.remaining > 0) return; // need more input
        fs.closeSync(this.current.fd);
        this.skip = this.current.padding; // discard body padding NEXT, not before
        this.current = null;
        continue;
      }

      if (this.buffer.length < 512) return;
      const header = this.buffer.subarray(0, 512);
      if (header.every((b) => b === 0)) {
        // End-of-archive terminator block.
        this.buffer = this.buffer.subarray(512);
        continue;
      }

      const size = parseInt(header.subarray(124, 136).toString('ascii').trim() || '0', 8);
      const typeflag = String.fromCharCode(header[156]);
      const padded = Math.ceil(size / 512) * 512;

      if (typeflag === 'L') {
        // GNU long name: the data block carries the next entry's name.
        if (this.buffer.length < 512 + padded) return;
        this.pendingLongName = this.buffer
          .subarray(512, 512 + size)
          .toString('utf8')
          .replace(/\0+$/, '');
        this.buffer = this.buffer.subarray(512 + padded);
        continue;
      }

      if (typeflag === 'x' || typeflag === 'g') {
        // pax headers — consume header now, discard the data via skip state.
        this.buffer = this.buffer.subarray(512);
        this.skip = padded;
        continue;
      }

      let name =
        this.pendingLongName ??
        (() => {
          const prefix = header.subarray(345, 500).toString('utf8').replace(/\0+$/, '');
          const base = header.subarray(0, 100).toString('utf8').replace(/\0+$/, '');
          return prefix ? `${prefix}/${base}` : base;
        })();
      this.pendingLongName = null;
      name = name.replace(/^\.\//, '');

      if (path.isAbsolute(name) || name.split('/').includes('..')) {
        throw new Error(`[lloyal.node] Refusing archive entry with unsafe path: ${name}`);
      }

      if (typeflag === '5') {
        fs.mkdirSync(path.join(this.destDir, name), { recursive: true });
        this.buffer = this.buffer.subarray(512);
        continue;
      }

      if (typeflag !== '0' && typeflag !== '\0') {
        throw new Error(
          `[lloyal.node] Refusing archive entry type '${typeflag}' (${name}) — packs contain regular files only.`,
        );
      }

      const target = path.join(this.destDir, name);
      fs.mkdirSync(path.dirname(target), { recursive: true });
      const fd = fs.openSync(target, 'w', 0o755);
      this.buffer = this.buffer.subarray(512);
      this.current = { fd, remaining: size, padding: padded - size };
    }
  }
}
