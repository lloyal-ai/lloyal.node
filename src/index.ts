/**
 * liblloyal-node - Thin N-API wrapper over liblloyal
 *
 * Exposes raw llama.cpp inference primitives for Node.js.
 *
 * @example
 * ```js
 * const { createContext } = require('@lloyal-labs/lloyal.node');
 *
 * const ctx = await createContext({
 *   modelPath: './model.gguf',
 *   nCtx: 2048,
 *   nThreads: 4
 * });
 *
 * // Tokenize
 * const tokens = await ctx.tokenize("Hello world");
 *
 * // Generate via Branch API
 * const branch = Branch.create(ctx, 0, { temperature: 0.7 });
 * await branch.prefill(tokens);
 * for await (const { text } of branch) {
 *   process.stdout.write(text);
 * }
 * await branch.prune();
 *
 * // Cleanup
 * ctx.dispose();
 * ```
 */

import type {
  ContextOptions,
  GpuVariant,
  LoadOptions,
  NativeBinding,
  SessionContext,
} from './types';

import { Branch } from './Branch';
import { BranchStore } from './BranchStore';
import { Session } from './Session';
import { forkAgent, runAgents } from './Agent';

/**
 * Platform package naming: @lloyal-labs/lloyal.node-{platform}-{arch}[-{gpu}]
 */
const getPlatformPackageName = (variant?: string): string => {
  const platform = process.platform;
  const arch = process.arch;
  const noSuffix = !variant || variant === 'default' || variant === 'cpu' || variant === 'metal';
  const suffix = noSuffix ? '' : `-${variant}`;
  return `@lloyal-labs/lloyal.node-${platform}-${arch}${suffix}`;
};

/**
 * Try to load a platform package, return null on failure.
 */
const tryLoadPackage = (packageName: string, verbose = false): NativeBinding | null => {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require(packageName) as NativeBinding;
    if (mod && typeof mod.createContext === 'function') {
      return mod;
    }
    if (verbose) {
      console.warn(`[lloyal.node] ${packageName} loaded but missing createContext export`);
    }
    return null;
  } catch (e) {
    if (verbose) {
      console.warn(`[lloyal.node] Failed to load ${packageName}: ${(e as Error).message}`);
    }
    return null;
  }
};

/**
 * Load native binary for a specific GPU variant
 *
 * lloyal.node ships as a family of platform-specific npm packages, each
 * containing a prebuilt native addon:
 * `@lloyal-labs/lloyal.node-{platform}-{arch}[-{gpu}]`
 * (e.g., `darwin-arm64`, `linux-x64-cuda`, `win32-x64-vulkan`).
 *
 * `loadBinary()` resolves the correct package at runtime with a prioritized
 * fallback chain:
 *
 * 1. Requested GPU variant package (if `variant` or `LLOYAL_GPU` env var set)
 * 2. Local development build (`build/Release/lloyal.node`)
 * 3. Default CPU platform package
 *
 * Most callers should use {@link createContext} directly — it calls
 * `loadBinary()` internally. Use this function when you need to:
 * - Pre-check whether a GPU variant is available before creating contexts
 * - Share one loaded binary across multiple context creations
 * - Inspect or test the binary loading logic in isolation
 *
 * **Environment variables:**
 * - `LLOYAL_LOCAL=1` — Force local build only; throws if not found
 *   (use during development to test local C++ changes)
 * - `LLOYAL_GPU=cuda|vulkan` — Request GPU variant (equivalent to `variant` param)
 * - `LLOYAL_NO_FALLBACK=1` — Disable silent CPU fallback; throws if GPU
 *   variant fails (use in CI to catch missing runtime libraries)
 *
 * @param variant GPU variant: 'cuda', 'vulkan', or undefined for CPU
 * @returns Native binary module with createContext method
 * @throws Error if no binary available for the current platform
 *
 * @example
 * ```typescript
 * // Load default (CPU) binary
 * const binary = loadBinary();
 *
 * // Load CUDA binary (falls back to CPU if unavailable)
 * const binary = loadBinary('cuda');
 *
 * // Create context from loaded binary
 * const ctx = await binary.createContext({ modelPath: './model.gguf' });
 * ```
 *
 * @category Core
 */
export const loadBinary = (variant?: GpuVariant): NativeBinding => {
  const resolvedVariant = variant ?? process.env.LLOYAL_GPU;
  const noFallback = process.env.LLOYAL_NO_FALLBACK === '1';
  const useLocal = process.env.LLOYAL_LOCAL === '1';

  // 0. Use local build if explicitly requested (no fallback)
  if (useLocal) {
    try {
      return require('../build/Release/lloyal.node') as NativeBinding;
    } catch {
      throw new Error(
        '[lloyal.node] LLOYAL_LOCAL=1 but local build not found. ' +
        'Run `npm run build` first.'
      );
    }
  }

  // 1. Try requested variant (if specified)
  if (resolvedVariant && resolvedVariant !== 'default') {
    const pkgName = getPlatformPackageName(resolvedVariant);
    const binary = tryLoadPackage(pkgName, true);
    if (binary) return binary;

    if (noFallback) {
      throw new Error(
        `[lloyal.node] GPU variant "${resolvedVariant}" failed to load. ` +
        `Package: ${pkgName}. Check that runtime libraries are available.`
      );
    }
    console.warn(`[lloyal.node] GPU variant "${resolvedVariant}" unavailable, falling back to CPU`);
  }

  // 2. Try local build (always fresher than installed packages during development)
  try {
    return require('../build/Release/lloyal.node') as NativeBinding;
  } catch {
    // ignore — no local build
  }

  // 3. Try default platform package (CPU)
  const defaultPkg = getPlatformPackageName();
  const binary = tryLoadPackage(defaultPkg, true);
  if (binary) return binary;

  throw new Error(
    `No lloyal.node binary found for ${process.platform}-${process.arch}. ` +
    `Tried: ${resolvedVariant ? getPlatformPackageName(resolvedVariant) + ', ' : ''}${defaultPkg}`
  );
};

// Default binary (loaded lazily on first use)
let _binary: NativeBinding | null = null;
const getBinary = (): NativeBinding => {
  if (!_binary) {
    _binary = loadBinary(process.env.LLOYAL_GPU as GpuVariant | undefined);
  }
  return _binary;
};

/**
 * Create a new inference context
 *
 * Entry point for all inference. Resolves the correct native binary (see
 * {@link loadBinary} for the platform/GPU fallback chain), loads the model
 * via a reference-counted registry (multiple contexts can share one model's
 * weight tensors in memory), and allocates a `llama_context` with its own
 * KV cache and compute scratch buffers.
 *
 * **What gets allocated:**
 * - KV cache: `nCtx * 2 * nLayers * dHead` bytes per KV type (fp16 default).
 *   For a 7B model with `nCtx: 4096`, expect ~1-2 GB of KV memory.
 * - Compute scratch: temporary buffers for the forward pass, sized to `nBatch`.
 *
 * **Model sharing:** If two contexts use the same `modelPath`, the model
 * weights are loaded once and shared. Only the KV cache and compute buffers
 * are per-context. This makes multi-context setups (e.g., one context per
 * conversation) memory-efficient.
 *
 * @param options Context creation options
 * @param loadOptions Optional binary loading options (GPU variant selection)
 * @returns Promise resolving to SessionContext instance
 *
 * @example Basic usage
 * ```typescript
 * const ctx = await createContext({
 *   modelPath: './model.gguf',
 *   nCtx: 2048,
 *   nThreads: 4
 * });
 *
 * try {
 *   const tokens = await ctx.tokenize("Hello");
 *   const branch = Branch.create(ctx, 0, { temperature: 0.7 });
 *   await branch.prefill(tokens);
 *   for await (const { text } of branch) process.stdout.write(text);
 * } finally {
 *   ctx.dispose();
 * }
 * ```
 *
 * @example Multi-branch context (tree search, best-of-N)
 * ```typescript
 * const ctx = await createContext({
 *   modelPath: './model.gguf',
 *   nCtx: 8192,
 *   nBatch: 512,     // Bin-packing capacity for BranchStore.prefill
 *   nSeqMax: 33,     // 32 branches + 1 root sequence
 * });
 * ```
 *
 * @example With GPU variant selection
 * ```typescript
 * const ctx = await createContext(
 *   { modelPath: './model.gguf', nCtx: 4096 },
 *   { gpuVariant: 'cuda' }
 * );
 * ```
 *
 * @category Core
 */
export const createContext = async (
  options: ContextOptions,
  loadOptions?: LoadOptions
): Promise<SessionContext> => {
  const variant = loadOptions?.gpuVariant || process.env.LLOYAL_GPU;
  const binary = variant ? loadBinary(variant as GpuVariant) : getBinary();
  return binary.createContext(options);
};

export { Branch, BranchStore, Session, forkAgent, runAgents };
export { PoolingType, ChatFormat, ReasoningFormat, GrammarTriggerType } from './types';
export type {
  GpuVariant,
  KvCacheType,
  LoadOptions,
  ContextOptions,
  FormatChatOptions,
  GrammarTrigger,
  FormattedChatResult,
  ParseChatOutputOptions,
  ParsedToolCall,
  ParseChatOutputResult,
  PenaltyParams,
  MirostatParams,
  DryParams,
  XtcParams,
  AdvancedSamplingParams,
  SamplingParams,
  SessionContext,
  Produced,
  AgentTask,
  AgentState,
  RunAgentsOptions,
  RunAgentsResult,
  NativeBinding,
} from './types';
