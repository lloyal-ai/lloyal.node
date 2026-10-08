/** Exercise the release scripts with fixture binaries; no model, GPU or build. */
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { test } = require('node:test');

const ROOT = path.resolve(__dirname, '..');
const LEGAL_FILES = ['LICENSE', 'NOTICE', 'GRANT.md', 'LICENSE-FAQ.md', 'liblloyal/LICENSE', 'liblloyal/NOTICE', 'llama.cpp/LICENSE', 'llama.cpp/NOTICE'];
const ROOT_LEGAL_FILES = LEGAL_FILES.slice(0, 4);
const { DL_CUDA_ARCHS, GGML_MIRRORED_ARCHS } = require('../scripts/dl-archs');

function write(root, file, content) {
  const destination = path.join(root, file);
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.writeFileSync(destination, content);
}

function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'lloyal-packaging-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  for (const script of ['create-platform-package.js', 'create-dl-pack.js', 'package-legal.js', 'dl-archs.js']) {
    write(root, `scripts/${script}`, fs.readFileSync(path.join(ROOT, 'scripts', script)));
  }
  write(root, 'package.json', fs.readFileSync(path.join(ROOT, 'package.json')));
  for (const file of ROOT_LEGAL_FILES) write(root, file, fs.readFileSync(path.join(ROOT, file)));
  // Deliberately distinct: a root-license substitution must fail the assertions.
  write(root, 'liblloyal/LICENSE', fs.readFileSync(path.join(ROOT, 'LICENSE'), 'utf8')
    .replace('Copyright 2026 Lloyal Labs', 'Copyright 2026 liblloyal fixture'));
  write(root, 'liblloyal/NOTICE', 'liblloyal fixture notice\n');
  write(root, 'llama.cpp/LICENSE', 'llama.cpp fixture MIT license\n');
  write(root, 'llama.cpp/NOTICE', 'llama.cpp fixture notice\n');
  write(root, 'build/Release/lloyal.node', 'fixture addon bytes\n');
  write(root, 'build/Release/libllama.so.0', 'fixture versioned library bytes\n');
  write(root, 'build/Release/libllama.dylib', 'fixture macOS library bytes\n');
  write(root, 'build/Release/llama.dll', 'fixture Windows library bytes\n');
  return root;
}

function run(root, script, args = [], env = {}) {
  return execFileSync(process.execPath, [path.join(root, 'scripts', script), ...args], {
    cwd: root, env: { ...process.env, ...env }, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'],
  });
}

function packedFiles(root) {
  const result = execFileSync('npm', ['pack', '--dry-run', '--json', '--ignore-scripts'], {
    cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'],
  });
  return JSON.parse(result)[0].files.map((file) => file.path);
}

function sameFile(root, destination, file) {
  assert.deepEqual(fs.readFileSync(path.join(destination, file)), fs.readFileSync(path.join(root, file)), file);
}

function mitFutureLicenses(root) {
  for (const file of ['LICENSE', 'liblloyal/LICENSE']) {
    const license = fs.readFileSync(path.join(root, file), 'utf8');
    assert.match(license, /^# Functional Source License, Version 1\.1, MIT Future License\n/, file);
    assert.match(license, /\nFSL-1\.1-MIT\n/, file);
    assert.match(license, /MIT license that is effective on the second anniversary/, file);
    assert.match(license, /Permission is hereby granted, free of charge/, file);
  }
}

test('main npm payload includes the grant, FAQ, license and notices', () => {
  const files = packedFiles(ROOT);
  for (const file of ROOT_LEGAL_FILES) assert.ok(files.includes(file), `npm payload missing ${file}`);
});

test('platform packages preserve legal documents and ship only selected binaries', (t) => {
  const root = fixture(t);
  for (const [name, runner, expectedOS, library] of [
    ['linux-x64', 'ubuntu-22.04', 'linux', 'libllama.so.0'],
    ['darwin-arm64', 'macos-14', 'darwin', 'libllama.dylib'],
    ['win32-x64', 'windows-2022', 'win32', 'llama.dll'],
  ]) {
    const arch = name.endsWith('arm64') ? 'arm64' : 'x64';
    run(root, 'create-platform-package.js', [name, runner, arch]);
    const destination = path.join(root, 'packages', name);
    const pkg = JSON.parse(fs.readFileSync(path.join(destination, 'package.json'), 'utf8'));
    assert.equal(pkg.license, 'SEE LICENSE IN LICENSE');
    assert.deepEqual(pkg.os, [expectedOS]);
    const files = packedFiles(destination);
    for (const file of LEGAL_FILES) {
      assert.ok(files.includes(file), `${name}: npm payload missing ${file}`);
      sameFile(root, destination, file);
    }
    mitFutureLicenses(destination);
    assert.ok(files.includes('bin/lloyal.node'));
    assert.ok(files.includes(`bin/${library}`));
    assert.equal(files.filter((file) => file.startsWith('bin/')).length, 2);
    assert.deepEqual(fs.readFileSync(path.join(destination, 'bin', library)), fs.readFileSync(path.join(root, 'build/Release', library)));
  }
});

test('packaging fails when a required legal document is missing', (t) => {
  const root = fixture(t);
  fs.unlinkSync(path.join(root, 'GRANT.md'));
  assert.throws(() => run(root, 'create-platform-package.js', ['linux-x64', 'ubuntu-22.04', 'x64']), /Missing legal document:.*GRANT\.md/);
  assert.equal(fs.existsSync(path.join(root, 'packages')), false);
});

test('stale root or kernel licensing fails before any payload is written', (t) => {
  for (const file of ['LICENSE', 'liblloyal/LICENSE']) {
    for (const [current, stale] of [
      ['MIT Future License', 'Apache 2.0 Future License'],
      ['FSL-1.1-MIT', 'FSL-1.1-Apache-2.0'],
    ]) {
      const root = fixture(t);
      const source = fs.readFileSync(path.join(root, file), 'utf8');
      write(root, file, source.replace(current, stale));
      const thirdPartyLicense = fs.readFileSync(path.join(root, 'llama.cpp/LICENSE'));
      const { copyNativeLegalFiles } = require(path.join(root, 'scripts/package-legal'));
      const destination = path.join(root, 'legal-copy');
      const expected = /Expected FSL-1\.1-MIT in (?:liblloyal\/)?LICENSE/;
      assert.throws(() => copyNativeLegalFiles(root, destination), expected);
      assert.equal(fs.existsSync(destination), false);
      assert.throws(() => run(root, 'create-platform-package.js', ['linux-x64', 'ubuntu-22.04', 'x64']), expected);
      assert.equal(fs.existsSync(path.join(root, 'packages')), false);
      assert.throws(() => run(root, 'create-dl-pack.js'), expected);
      assert.equal(fs.existsSync(path.join(root, 'packs')), false);
      assert.deepEqual(fs.readFileSync(path.join(root, 'llama.cpp/LICENSE')), thirdPartyLicense);
    }
  }
});

test('R2 backend and CUDA archives retain legal documents, binaries and deterministic digests', { skip: process.platform !== 'linux' }, (t) => {
  const root = fixture(t);
  write(root, 'liblloyal/.llama-cpp-version', 'fixture-llama-tag\n');
  write(root, 'llama.cpp/ggml/src/ggml-cuda/CMakeLists.txt', `list(APPEND CMAKE_CUDA_ARCHITECTURES ${GGML_MIRRORED_ARCHS.join(' ')})\n`);
  write(root, 'build/Release/libggml-cuda.so', 'fixture CUDA backend bytes\n');
  for (let i = 0; i < 8; i++) write(root, `build/Release/libggml-cpu-fixture${i}.so`, `fixture CPU backend ${i}\n`);
  const cudaPath = path.join(root, 'cuda-12.9');
  for (const lib of ['cudart', 'cublas', 'cublasLt', 'nvJitLink']) {
    write(cudaPath, `lib64/lib${lib}.so.12`, `fixture NVIDIA ${lib} bytes\n`);
    fs.symlinkSync(`lib${lib}.so.12`, path.join(cudaPath, 'lib64', `lib${lib}.so`));
  }
  write(cudaPath, 'EULA.txt', 'fixture NVIDIA EULA, distinct from Lloyal FSL\n');
  const probe = path.join(root, 'fixture-bin', 'cuobjdump');
  write(root, 'fixture-bin/cuobjdump', `#!${process.execPath}\nconsole.log(${JSON.stringify(DL_CUDA_ARCHS.map((arch) => `sm_${arch.split('-')[0]}`).join('\n'))});\n`);
  fs.chmodSync(probe, 0o755);
  const env = { CUDA_PATH: cudaPath, PATH: `${path.dirname(probe)}${path.delimiter}${process.env.PATH}` };
  execFileSync('git', ['init', '-q'], { cwd: root });
  execFileSync('git', ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null', 'commit', '--allow-empty', '-qm', 'Fixture epoch'], { cwd: root });

  run(root, 'create-dl-pack.js', [], env);
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'packs/manifest.json'), 'utf8'));
  const extracted = path.join(root, 'extracted');
  fs.mkdirSync(extracted);
  for (const archive of [manifest.archive, manifest.runtimeArchive]) {
    const source = path.join(root, 'packs', archive.file);
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(source)).digest('hex'), archive.sha256);
    assert.equal(fs.statSync(source).size, archive.sizeBytes);
    execFileSync('tar', ['--zstd', '-xf', source, '-C', extracted]);
  }
  for (const file of LEGAL_FILES) sameFile(root, extracted, file);
  mitFutureLicenses(extracted);
  for (const file of ['lloyal.node', 'libllama.so.0', 'libggml-cuda.so']) {
    assert.deepEqual(fs.readFileSync(path.join(extracted, file)), fs.readFileSync(path.join(root, 'build/Release', file)));
  }
  for (const [file, digest] of Object.entries(manifest.files)) {
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(extracted, file))).digest('hex'), digest, file);
  }
  for (const file of LEGAL_FILES) assert.ok(manifest.files[file], `manifest must cover ${file}`);
  assert.deepEqual(fs.readFileSync(path.join(extracted, 'third-party/cuda/EULA.txt')), fs.readFileSync(path.join(cudaPath, 'EULA.txt')));
  assert.deepEqual(fs.readFileSync(path.join(extracted, 'libcudart.so')), fs.readFileSync(path.join(cudaPath, 'lib64/libcudart.so.12')));
  run(root, 'create-dl-pack.js', [], env);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(root, 'packs/manifest.json'), 'utf8')), manifest, 'same inputs must yield identical archives and manifest');
  fs.unlinkSync(path.join(cudaPath, 'EULA.txt'));
  assert.throws(() => run(root, 'create-dl-pack.js', [], env), /Missing legal document:.*EULA\.txt/);
});
