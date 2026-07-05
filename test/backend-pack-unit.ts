/**
 * Unit tests for the backend-pack acquisition layer — no GPU, no network.
 * tsx + node:assert (the repo's test convention; run via `npm run test:unit`).
 *
 * Coverage: signature verify round-trip + pinned-key fingerprint, probe
 * gate matrix against faked nvidia-smi/ldconfig outputs (A100/H100/L4/
 * B200/no-GPU), cache resolution precedence incl. corrupt-cache-throws,
 * and the ustar extractor (round-trip, GNU longname, traversal + symlink
 * rejection, truncation). Everything network/GPU-shaped is proven on the
 * L4 rig (gpu-tests-dl) and the end-to-end run.
 */

import assert from 'node:assert/strict';
import { generateKeyPairSync, sign as cryptoSign, createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import * as zlib from 'node:zlib';

import { sha256Hex, verifyPlatformSignature } from '../src/verify';
import {
  backendPackCacheDir,
  detectCudaRuntime,
  detectGpu,
  extractTarZst,
  probeBackendPack,
  resolveBackendPackDirSync,
  type BackendPackManifest,
  type CommandRunner,
} from '../src/backend-pack';

let passed = 0;
async function test(name: string, fn: () => void | Promise<void>): Promise<void> {
  try {
    await fn();
    passed++;
    console.log(`  ✓ ${name}`);
  } catch (err) {
    console.error(`  ✗ ${name}`);
    console.error(err);
    process.exit(1);
  }
}

// ── Fixtures ─────────────────────────────────────────────────────────

const MANIFEST: BackendPackManifest = {
  name: 'lloyal.node-backend-pack',
  version: '3.1.0',
  platform: 'linux-x64-dl',
  llamaCppTag: 'b9581',
  cudaToolkit: '12.9',
  requiredCudaRuntime: '12.9',
  archs: {
    real: ['86', '89', '100a', '120a', '121a'],
    virtual: ['50', '61', '70', '75', '80', '90'],
  },
  archive: { file: 'linux-x64-dl.tar.zst', sizeBytes: 800_000_000, sha256: 'a'.repeat(64) },
  runtimeArchive: { file: 'runtime-cuda12.9.tar.zst', sizeBytes: 250_000_000, sha256: 'b'.repeat(64) },
};

/** Fake CommandRunner: gpu query CSV, plain nvidia-smi banner, ldconfig line. */
function fakeRunner(opts: {
  gpuCsv?: string | null;
  bannerCuda?: string | null;
  cudartRealpath?: string | null;
}): CommandRunner {
  return (cmd, args) => {
    if (cmd === 'nvidia-smi' && args.length > 0) {
      return opts.gpuCsv == null
        ? { status: 1, stdout: '' }
        : { status: 0, stdout: `${opts.gpuCsv}\n` };
    }
    if (cmd === 'nvidia-smi') {
      return opts.bannerCuda == null
        ? { status: 1, stdout: '' }
        : { status: 0, stdout: `| NVIDIA-SMI 580.65  Driver Version: 580.65  CUDA Version: ${opts.bannerCuda} |\n` };
    }
    if (cmd === 'ldconfig') {
      return opts.cudartRealpath == null
        ? { status: 0, stdout: 'libfoo.so.1 => /lib/libfoo.so.1\n' }
        : { status: 0, stdout: `\tlibcudart.so.12 (libc6,x86-64) => ${opts.cudartRealpath}\n` };
    }
    return { status: 1, stdout: '' };
  };
}

/** ldconfig realpath fixture — a real temp file so realpathSync works. */
function cudartFixture(version: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cudart-'));
  const file = path.join(dir, `libcudart.so.${version}`);
  fs.writeFileSync(file, '');
  const link = path.join(dir, 'libcudart.so.12');
  fs.symlinkSync(file, link);
  return link;
}

// Fake linux-x64 platform for gate tests when running on darwin dev boxes.
const IS_LINUX_X64 = process.platform === 'linux' && process.arch === 'x64';

// ── Minimal ustar writer (test-side only) ────────────────────────────

function tarHeader(name: string, size: number, typeflag: string): Buffer {
  const header = Buffer.alloc(512);
  header.write(name, 0, 100, 'utf8');
  header.write('0000755\0', 100, 8, 'ascii'); // mode
  header.write('0000000\0', 108, 8, 'ascii'); // uid
  header.write('0000000\0', 116, 8, 'ascii'); // gid
  header.write(size.toString(8).padStart(11, '0') + '\0', 124, 12, 'ascii');
  header.write('00000000000\0', 136, 12, 'ascii'); // mtime
  header.write('        ', 148, 8, 'ascii'); // chksum placeholder = spaces
  header.write(typeflag, 156, 1, 'ascii');
  header.write('ustar\0', 257, 6, 'ascii');
  header.write('00', 263, 2, 'ascii');
  let sum = 0;
  for (const b of header) sum += b;
  header.write(sum.toString(8).padStart(6, '0') + '\0 ', 148, 8, 'ascii');
  return header;
}

function tarEntry(name: string, content: Buffer, typeflag = '0'): Buffer[] {
  const blocks: Buffer[] = [];
  if (name.length > 99) {
    const nameBuf = Buffer.from(name + '\0', 'utf8');
    blocks.push(tarHeader('././@LongLink', nameBuf.length, 'L'));
    blocks.push(pad512(nameBuf));
    name = name.slice(0, 99);
  }
  blocks.push(tarHeader(name, content.length, typeflag));
  if (typeflag === '0' && content.length > 0) blocks.push(pad512(content));
  return blocks;
}

function pad512(buf: Buffer): Buffer {
  const padded = Math.ceil(buf.length / 512) * 512;
  return Buffer.concat([buf, Buffer.alloc(padded - buf.length)]);
}

function makeTarZst(entries: Buffer[][], truncate = 0): string {
  let tar = Buffer.concat([...entries.flat(), Buffer.alloc(1024)]);
  if (truncate > 0) tar = tar.subarray(0, tar.length - truncate);
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'pack-')), 'fixture.tar.zst');
  fs.writeFileSync(file, zlib.zstdCompressSync(tar));
  return file;
}

// ── Tests ────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log('verify.ts');

  await test('Ed25519 verify round-trip over exact bytes', () => {
    const { publicKey, privateKey } = generateKeyPairSync('ed25519');
    const raw = new Uint8Array(publicKey.export({ format: 'der', type: 'spki' })).slice(-32);
    const bytes = new TextEncoder().encode('{"a":1}');
    const sig = cryptoSign(null, bytes, privateKey).toString('base64');
    assert.equal(verifyPlatformSignature(bytes, sig, raw), true);
    assert.equal(verifyPlatformSignature(new TextEncoder().encode('{"a":2}'), sig, raw), false);
    assert.equal(verifyPlatformSignature(bytes, 'not-base64!', raw), false);
  });

  await test('pinned platform key matches its documented fingerprint', () => {
    // The docstring's SHA-256 fingerprint — byte-parity with harness.dev/rig.
    const src = fs.readFileSync(path.join(__dirname, '..', 'src', 'verify.ts'), 'utf8');
    const nums = src
      .match(/LLOYAL_PLATFORM_KEY_2026_Q2 = new Uint8Array\(\[([\s\S]*?)\]\)/)![1]
      .match(/\d+/g)!
      .map(Number);
    const fingerprint = createHash('sha256').update(Buffer.from(nums)).digest('hex');
    assert.equal(fingerprint, '9e0df3d25b8968a8b2ae9b86cb17a6922368c7cff9674a84b4a2527dd6457ec1');
  });

  await test('sha256Hex', () => {
    assert.equal(
      sha256Hex(new TextEncoder().encode('abc')),
      'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
    );
  });

  console.log('probe gates');

  await test('detectGpu parses csv + banner; null without nvidia-smi', () => {
    const gpu = detectGpu(fakeRunner({ gpuCsv: 'NVIDIA H100 80GB HBM3, 9.0, 580.65', bannerCuda: '13.0' }));
    assert.deepEqual(gpu, {
      name: 'NVIDIA H100 80GB HBM3',
      computeCap: '9.0',
      driverVersion: '580.65',
      driverCudaVersion: '13.0',
    });
    assert.equal(detectGpu(fakeRunner({ gpuCsv: null })), null);
  });

  await test('detectCudaRuntime reads the minor from the realpath', () => {
    assert.equal(detectCudaRuntime(fakeRunner({ cudartRealpath: cudartFixture('12.2.140') })), '12.2');
    assert.equal(detectCudaRuntime(fakeRunner({ cudartRealpath: null })), null);
  });

  if (IS_LINUX_X64) {
    await test('B200 (sm_100): native SASS → recommended; old runtime → companion', async () => {
      const p = await probeBackendPack({
        manifest: MANIFEST,
        run: fakeRunner({ gpuCsv: 'NVIDIA B200, 10.0, 580.65', bannerCuda: '13.0', cudartRealpath: cudartFixture('12.2.140') }),
      });
      assert.deepEqual(p.gates, { device: true, driver: true, runtime: false });
      assert.equal(p.needsRuntimeArchive, true);
      assert.equal(p.recommended, true);
    });

    await test('H100 (sm_90): PTX-only — driver must JIT the pack toolkit', async () => {
      const newDriver = await probeBackendPack({
        manifest: MANIFEST,
        run: fakeRunner({ gpuCsv: 'NVIDIA H100, 9.0, 580.65', bannerCuda: '12.9', cudartRealpath: cudartFixture('12.9.1') }),
      });
      assert.equal(newDriver.recommended, true);
      const oldDriver = await probeBackendPack({
        manifest: MANIFEST,
        run: fakeRunner({ gpuCsv: 'NVIDIA H100, 9.0, 535.129', bannerCuda: '12.2', cudartRealpath: cudartFixture('12.9.1') }),
      });
      assert.equal(oldDriver.gates.driver, false);
      assert.equal(oldDriver.recommended, false);
    });

    await test('L4 (sm_89): npm-native class → never offered', async () => {
      const p = await probeBackendPack({
        manifest: MANIFEST,
        run: fakeRunner({ gpuCsv: 'NVIDIA L4, 8.9, 535.129', bannerCuda: '12.2', cudartRealpath: cudartFixture('12.9.1') }),
      });
      assert.equal(p.recommended, false);
      assert.match(p.reasons.join('\n'), /served natively by the standard npm package/);
    });

    await test('no GPU → no offer, legible reason', async () => {
      const p = await probeBackendPack({ manifest: MANIFEST, run: fakeRunner({ gpuCsv: null }) });
      assert.equal(p.gpu, null);
      assert.equal(p.recommended, false);
    });
  } else {
    console.log('  (gate matrix skipped — platform gate short-circuits off linux-x64; runs in CI)');
  }

  console.log('cache resolution');

  await test('no marker → null; valid marker → dir; corrupt marker → throws', () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'cache-'));
    const prev = process.env.XDG_CACHE_HOME;
    process.env.XDG_CACHE_HOME = tmp;
    try {
      assert.equal(resolveBackendPackDirSync('9.9.9'), null);
      const dir = backendPackCacheDir('9.9.9');
      fs.mkdirSync(dir, { recursive: true });
      assert.equal(resolveBackendPackDirSync('9.9.9'), null); // dir but no marker → invisible
      fs.writeFileSync(path.join(dir, '.lloyal-pack.json'), JSON.stringify({ version: '9.9.9' }));
      assert.equal(resolveBackendPackDirSync('9.9.9'), dir);
      fs.writeFileSync(path.join(dir, '.lloyal-pack.json'), '{corrupt');
      assert.throws(() => resolveBackendPackDirSync('9.9.9'), /corrupt/);
    } finally {
      if (prev === undefined) delete process.env.XDG_CACHE_HOME;
      else process.env.XDG_CACHE_HOME = prev;
    }
  });

  console.log('ustar extractor');

  await test('round-trips files, dirs, and GNU long names', async () => {
    const longName = `deep/${'x'.repeat(120)}/libggml-cuda.so`;
    const archive = makeTarZst([
      tarEntry('lloyal.node', Buffer.from('addon-bytes')),
      tarEntry('sub/', Buffer.alloc(0), '5'),
      tarEntry('sub/libllama.so.0', Buffer.from('l'.repeat(1000))),
      tarEntry(longName, Buffer.from('module')),
    ]);
    const dest = fs.mkdtempSync(path.join(os.tmpdir(), 'extract-'));
    await extractTarZst(archive, dest);
    assert.equal(fs.readFileSync(path.join(dest, 'lloyal.node'), 'utf8'), 'addon-bytes');
    assert.equal(fs.readFileSync(path.join(dest, 'sub/libllama.so.0'), 'utf8'), 'l'.repeat(1000));
    assert.equal(fs.readFileSync(path.join(dest, longName), 'utf8'), 'module');
  });

  await test('rejects path traversal and symlink entries', async () => {
    const dest = fs.mkdtempSync(path.join(os.tmpdir(), 'extract-'));
    await assert.rejects(
      extractTarZst(makeTarZst([tarEntry('../escape', Buffer.from('x'))]), dest),
      /unsafe path/,
    );
    await assert.rejects(
      extractTarZst(makeTarZst([tarEntry('link', Buffer.alloc(0), '2')]), dest),
      /regular files only/,
    );
  });

  await test('throws on a truncated archive', async () => {
    const dest = fs.mkdtempSync(path.join(os.tmpdir(), 'extract-'));
    await assert.rejects(
      extractTarZst(makeTarZst([tarEntry('big', Buffer.alloc(4096, 7))], 2048), dest),
      /Truncated/,
    );
  });

  console.log(`\nPASSED: ${passed}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
