/**
 * The BACKEND_DL flavor's explicit CUDA architecture list — single source
 * of truth shared by scripts/build.js (passes it to CMake) and
 * scripts/create-dl-pack.js (asserts the built fatbin against it, and
 * drift-checks it against ggml's own derivation on every llama.cpp sync).
 *
 * MIRROR of ggml's derivation (llama.cpp/ggml/src/ggml-cuda/CMakeLists.txt,
 * toolkit ≥12.9 <13 branch) PLUS the one documented extension:
 *   100a-real — B200. Upstream defaults leave sm_100 JIT-ing Hopper PTX;
 *   the rental fleet this channel exists for goes native instead. Same
 *   arch-locked shape as upstream's own 120a/121a choice, backstopped by
 *   the 90-virtual PTX forward floor.
 *
 * Don't invent entries here — that's how the npm pin's A100 trap happened.
 * Changes must update the mirror check in create-dl-pack.js --check-archs.
 */

const DL_CUDA_ARCHS = [
  '50-virtual',
  '61-virtual',
  '70-virtual',
  '75-virtual',
  '80-virtual',
  '86-real',
  '89-real',
  '90-virtual',
  '100a-real', // lloyal extension (B200) — NOT in ggml's derivation
  '120a-real',
  '121a-real',
];

/** Entries mirroring ggml's derivation exactly (everything but our extensions). */
const GGML_MIRRORED_ARCHS = DL_CUDA_ARCHS.filter((a) => a !== '100a-real');

module.exports = { DL_CUDA_ARCHS, GGML_MIRRORED_ARCHS };
