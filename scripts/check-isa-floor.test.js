#!/usr/bin/env node
/**
 * Unit tests for the AVX-512 detector in check-isa-floor.js.
 * Pure (no disassembler / binaries needed) so it runs anywhere, fast.
 * Run: node scripts/check-isa-floor.test.js   (exit 0 = pass, 1 = fail)
 */
const assert = require('assert');
const { lineHasAvx512, isX64 } = require('./check-isa-floor.js');

// Lines that MUST be flagged as AVX-512 (zmm regs, k-mask, k-ops, EVEX mnemonics);
// both AT&T and Intel disassembly forms.
const positives = [
  'vpaddd %zmm1, %zmm2, %zmm3',
  'vmovdqu64 (%rax), %zmm0',
  // 2-digit zmm registers (zmm16..zmm31) must also be caught:
  'vpaddd %zmm16, %zmm17, %zmm18',
  'vmovdqa64 zmm31, zmm0',
  'vaddps %zmm2, %zmm3, %zmm1 {%k1}',
  '  4a1: 62 f1 ... vaddps zmm1{k1}, zmm2, zmm3', // Intel/EVEX form
  'kmovw %k1, %eax',
  // k-mask ops with mandatory size suffixes (the cases the first pattern missed):
  'kaddw %k1, %k2, %k3',
  'knotw %k1, %k2',
  'kshiftlw $1, %k1, %k2',
  'kandnq %k1, %k2, %k3',
  'kunpckbw %k1, %k2, %k3',
  'kxnorw %k1, %k2, %k3',
  'kortestw %k1, %k2',
  'vpternlogd $0xff, %zmm0, %zmm0, %zmm0',
  'vpconflictd %zmm0, %zmm1',
];

// Lines that MUST NOT be flagged (AVX2/SSE, plain instrs, symbol names with "kor").
const negatives = [
  'vpaddd %ymm1, %ymm2, %ymm3',
  'vmovdqu (%rax), %ymm0',
  'mov %rax, %rbx',
  'vfmadd231ps %ymm1, %ymm2, %ymm3',
  'vbroadcastss %xmm0, %ymm1',
  '0000000000001234 <_some_kor_function>:',
  'callq  0x1234 <ggml_vec_dot>',
];

let failures = 0;
const check = (cond, msg) => {
  try { assert.ok(cond, msg); } catch (e) { failures++; console.error('FAIL:', e.message); }
};

for (const l of positives) check(lineHasAvx512(l), `expected AVX-512 match: ${l}`);
for (const l of negatives) check(!lineHasAvx512(l), `unexpected AVX-512 match: ${l}`);
check(isX64('x64') && isX64('x86_64') && !isX64('arm64'), 'isX64 arch classification');

if (failures) {
  console.error(`check-isa-floor.test: ${failures} assertion(s) failed`);
  process.exit(1);
}
console.log(
  `check-isa-floor.test: OK (${positives.length} positives, ${negatives.length} negatives)`,
);
