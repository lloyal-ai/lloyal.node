#!/usr/bin/env node
/**
 * check-isa-floor.js — CI guard against shipping over-baselined x64 prebuilts.
 *
 * Background (github.com/lloyal-ai/hdk#20): with GGML_NATIVE=ON (-march=native),
 * the binary's CPU-ISA floor is whatever the *build host* had. On an AVX-512
 * CI host that bakes AVX-512 into the x64 prebuilts, which then abort with an
 * illegal instruction (0xC000001D) inside createContext() on any consumer CPU
 * without AVX-512. Our matrix never runs the artifact on a weaker CPU (build
 * host == test host), so a runtime test can't catch this — but a *static*
 * disassembly check can, independent of the CI machine's own CPU.
 *
 * This script disassembles the addon + bundled ggml/llama libraries and fails
 * if AVX-512 instructions/registers appear in an x64 artifact.
 *
 * Usage:  node scripts/check-isa-floor.js <dir> [arch]
 *   <dir>  directory containing lloyal.node + shared libs
 *          (e.g. build/Release, or packages/<name>/bin)
 *   [arch] target arch (default: process.arch). arm64 targets are exempt.
 *
 * Exit codes: 0 = clean / exempt, 1 = AVX-512 found, 2 = no disassembler / no input.
 */

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// AVX-512 signal: 512-bit zmm registers, k-mask predication ({k1}), k-register
// ops, and EVEX-only mnemonics. Any of these in an x64 binary => AVX-512 baked
// in. Heuristic but low-false-positive (none exist below AVX-512).
const AVX512 = new RegExp(
  [
    '\\bzmm[0-9]',                       // 512-bit registers
    '\\{%?k[0-7]\\}',                    // masked operand {k1} / {%k1}
    '\\bk(?:mov[bwdq]|and[bwdq]?|or[bwdq]?|xor|not|add|shift|test|unpck)\\b', // k-register ops
    // EVEX-only mnemonics:
    '\\bv(?:pternlog|pconflict|plzcnt|pmadd52|popcnt[bwdq]|permt2|permi2|' +
      'fixupimm|rndscale|reduce[ps][sd]|scalef|rcp14|rsqrt14|getexp|getmant)',
  ].join('|'),
  'i',
);

/** True if a single disassembly line uses AVX-512. */
const lineHasAvx512 = (line) => AVX512.test(line);

const isX64 = (arch) => arch === 'x64' || arch === 'x86_64';

// Pick a disassembler: prefer llvm-objdump (portable), then platform default.
function pickDisassembler() {
  const candidates =
    process.platform === 'win32'
      ? [['llvm-objdump', ['-d']], ['dumpbin', ['/disasm:nobytes']]]
      : [['llvm-objdump', ['-d']], ['objdump', ['-d']]];
  for (const [bin, baseArgs] of candidates) {
    const probe = spawnSync(bin, ['--version'], { encoding: 'utf8' });
    if (!probe.error && probe.status === 0) return { bin, baseArgs };
  }
  return null;
}

const BIN_EXTS = ['.node', '.so', '.dll', '.dylib'];
const isBinary = (f) =>
  f === 'lloyal.node' || BIN_EXTS.some((e) => f.endsWith(e)) || /\.so(\.\d+)+$/.test(f);

/** Returns true if any line of the file's disassembly uses AVX-512. */
function fileHasAvx512(tool, file) {
  const out = spawnSync(tool.bin, [...tool.baseArgs, file], {
    encoding: 'utf8',
    maxBuffer: 1024 * 1024 * 1024,
  });
  if (out.error) throw new Error(`disassemble ${path.basename(file)}: ${out.error.message}`);
  const text = (out.stdout || '') + (out.stderr || '');
  return text.split('\n').find(lineHasAvx512) || null;
}

function main(argv) {
  const dir = argv[0] || path.join(__dirname, '..', 'build', 'Release');
  const arch = (argv[1] || process.arch).toLowerCase();

  if (!isX64(arch)) {
    console.log(`[check-isa-floor] arch "${arch}" is not x64 — AVX-512 check not applicable. PASS.`);
    return 0;
  }
  if (!fs.existsSync(dir)) {
    console.error(`[check-isa-floor] directory not found: ${dir}`);
    return 2;
  }

  const tool = pickDisassembler();
  if (!tool) {
    console.error(
      '[check-isa-floor] no disassembler found (tried llvm-objdump / objdump / dumpbin). ' +
        'Install one so the ISA-floor gate cannot silently no-op.',
    );
    return 2;
  }

  const targets = fs.readdirSync(dir).filter(isBinary).map((f) => path.join(dir, f));
  if (targets.length === 0) {
    console.error(`[check-isa-floor] no binaries (${BIN_EXTS.join(', ')}) found in ${dir}`);
    return 2;
  }

  console.log(`[check-isa-floor] disassembler: ${tool.bin} · ${targets.length} binaries · arch=${arch}`);
  let failed = false;
  for (const file of targets) {
    let hit;
    try {
      hit = fileHasAvx512(tool, file);
    } catch (e) {
      console.error(`[check-isa-floor] ${e.message}`);
      return 2;
    }
    if (hit) {
      failed = true;
      console.error(`[check-isa-floor] ❌ AVX-512 in ${path.basename(file)} → ${hit.trim()}`);
    } else {
      console.log(`[check-isa-floor] ✓ ${path.basename(file)} (no AVX-512)`);
    }
  }

  if (failed) {
    console.error(
      '\n[check-isa-floor] AVX-512 detected in an x64 artifact. This will crash ' +
        '(0xC000001D) on non-AVX-512 CPUs. Ensure scripts/build.js pins ' +
        'GGML_NATIVE=OFF + GGML_AVX2=ON for x64. See hdk#20.',
    );
    return 1;
  }
  console.log('[check-isa-floor] ✅ ISA floor OK — no AVX-512 in x64 artifacts.');
  return 0;
}

if (require.main === module) {
  process.exit(main(process.argv.slice(2)));
}

module.exports = { AVX512, lineHasAvx512, isX64, fileHasAvx512, pickDisassembler, main };
