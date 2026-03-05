/**
 * liblloyal-node — native-only type definitions
 *
 * Types specific to the Node.js native addon (binary loading, GPU variant
 * selection). All inference primitives and shared types are in
 * {@link @lloyal-labs/sdk | @lloyal-labs/sdk}.
 */

import type { ContextOptions, SessionContext } from '@lloyal-labs/sdk';

/**
 * GPU variant for binary loading
 *
 * Specifies which GPU-accelerated binary to load:
 * - 'default': CPU-only (works everywhere)
 * - 'cuda': NVIDIA CUDA (requires libcudart.so/cudart64.dll)
 * - 'vulkan': Vulkan (AMD/Intel/NVIDIA, requires Vulkan runtime)
 *
 * If the requested variant is unavailable (package not installed or
 * runtime libraries missing), loading automatically falls back to CPU.
 *
 * @category Core
 */
export type GpuVariant = 'default' | 'cuda' | 'vulkan';

/**
 * Options for binary loading
 *
 * Controls which native binary variant is loaded when creating a context.
 * Use this for explicit GPU variant selection with automatic fallback.
 *
 * @category Core
 */
export interface LoadOptions {
  /**
   * GPU variant to use
   *
   * - 'cuda': NVIDIA CUDA (requires libcudart.so)
   * - 'vulkan': Vulkan (AMD/Intel/NVIDIA)
   * - 'default' or undefined: CPU only
   *
   * If the requested variant is unavailable (missing runtime libraries),
   * automatically falls back to CPU with a console warning.
   */
  gpuVariant?: GpuVariant;
}

/**
 * Native binding interface — what loadBinary() returns
 *
 * @category Core
 */
export interface NativeBinding {
  createContext(options: ContextOptions): Promise<SessionContext>;
}
