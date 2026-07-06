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

import * as path from "node:path";

import { resolveBackendPackDirSync } from "./backend-pack";
import type { BackendDevice, GpuVariant, LoadOptions, NativeBinding } from "./types";

import type { ContextOptions, SessionContext } from "@lloyal-labs/sdk";

/**
 * Platform package naming: @lloyal-labs/lloyal.node-{platform}-{arch}[-{gpu}]
 */
const getPlatformPackageName = (variant?: string): string => {
  const platform = process.platform;
  const arch = process.arch;
  const noSuffix =
    !variant ||
    variant === "default" ||
    variant === "cpu" ||
    variant === "metal";
  const suffix = noSuffix ? "" : `-${variant}`;
  return `@lloyal-labs/lloyal.node-${platform}-${arch}${suffix}`;
};

/**
 * Try to load a platform package, return null on failure.
 */
const tryLoadPackage = (
  packageName: string,
  verbose = false,
): NativeBinding | null => {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require(packageName) as NativeBinding;
    if (mod && typeof mod.createContext === "function") {
      return mod;
    }
    if (verbose) {
      console.warn(
        `[lloyal.node] ${packageName} loaded but missing createContext export`,
      );
    }
    return null;
  } catch (e) {
    if (verbose) {
      console.warn(
        `[lloyal.node] Failed to load ${packageName}: ${(e as Error).message}`,
      );
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
 * - `LLOYAL_BACKEND_DIR=<dir>` — Load the BACKEND_DL flavor addon from this
 *   directory ONLY; throws on any failure (provisioner/dev override —
 *   mirrors `LLOYAL_LOCAL`'s exclusivity). ggml dlopens backend modules
 *   from the same directory at init.
 * - `LLOYAL_GPU=cuda|vulkan` — Request GPU variant (equivalent to `variant` param)
 * - `LLOYAL_NO_FALLBACK=1` — Disable silent CPU fallback; throws if GPU
 *   variant fails (use in CI to catch missing runtime libraries)
 *
 * **Backend pack resolution (BACKEND_DL flavor):** before the npm chain,
 * `loadBinary` checks (a) `LLOYAL_BACKEND_DIR`, then (b) the shared cache
 * (`~/.cache/lloyal/backends/<version>-<platform>-<arch>/`, populated only
 * by explicit consent via {@link ensureBackendPack} or by a provisioner —
 * its existence IS the consent record). Both are exclusive: a resolution
 * failure throws and never falls through — a corrupt pack must not
 * silently degrade to the npm CPU package.
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
  const noFallback = process.env.LLOYAL_NO_FALLBACK === "1";
  const useLocal = process.env.LLOYAL_LOCAL === "1";

  // 0. Use local build if explicitly requested (no fallback)
  if (useLocal) {
    try {
      return require("../build/Release/lloyal.node") as NativeBinding;
    } catch {
      throw new Error(
        "[lloyal.node] LLOYAL_LOCAL=1 but local build not found. " +
          "Run `npm run build` first.",
      );
    }
  }

  // 0.5. Explicit backend-pack pointer — exclusive, fail loud (the
  // LLOYAL_LOCAL pattern): a provisioner or dev that names a dir gets
  // exactly that dir or an error, never a silent substitute.
  const backendDir = process.env.LLOYAL_BACKEND_DIR;
  if (backendDir) {
    return assertRequestedDevices(
      loadPackAddon(backendDir, `LLOYAL_BACKEND_DIR=${backendDir}`),
      resolvedVariant,
    );
  }

  // 0.6. Verified backend-pack cache for THIS lloyal.node version. The
  // cache only exists through explicit consent (ensureBackendPack) or
  // explicit provisioning, so preferring it honors a prior decision. A
  // present-but-invalid cache throws inside the resolver; a load failure
  // of a valid cache throws here — no fallthrough to npm either way.
  const packDir = resolveBackendPackDirSync();
  if (packDir) {
    return assertRequestedDevices(
      loadPackAddon(packDir, `backend pack cache ${packDir}`),
      resolvedVariant,
    );
  }

  // 1. Try requested variant (if specified)
  if (resolvedVariant && resolvedVariant !== "default") {
    const pkgName = getPlatformPackageName(resolvedVariant);
    const binary = tryLoadPackage(pkgName, true);
    if (binary) return binary;

    if (noFallback) {
      throw new Error(
        `[lloyal.node] GPU variant "${resolvedVariant}" failed to load. ` +
          `Package: ${pkgName}. Check that runtime libraries are available.`,
      );
    }
    console.warn(
      `[lloyal.node] GPU variant "${resolvedVariant}" unavailable, falling back to CPU`,
    );
  }

  // 2. Try local build (always fresher than installed packages during development)
  try {
    return require("../build/Release/lloyal.node") as NativeBinding;
  } catch {
    // ignore — no local build
  }

  // 3. Try default platform package (CPU)
  const defaultPkg = getPlatformPackageName();
  const binary = tryLoadPackage(defaultPkg, true);
  if (binary) return binary;

  throw new Error(
    `No lloyal.node binary found for ${process.platform}-${process.arch}. ` +
      `Tried: ${resolvedVariant ? getPlatformPackageName(resolvedVariant) + ", " : ""}${defaultPkg}`,
  );
};

/** Require the DL-flavor addon out of a resolved pack dir; loud on failure. */
const loadPackAddon = (dir: string, sourceLabel: string): NativeBinding => {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require(path.join(dir, "lloyal.node")) as NativeBinding;
    if (!mod || typeof mod.createContext !== "function") {
      throw new Error("loaded module is missing createContext");
    }
    return mod;
  } catch (e) {
    throw new Error(
      `[lloyal.node] Failed to load backend pack addon from ${sourceLabel}: ` +
        `${(e as Error).message}. Refusing to fall back — delete the pack ` +
        `directory (or unset LLOYAL_BACKEND_DIR) to use the npm packages.`,
    );
  }
};

/**
 * Post-load fail-loud assertion: a requested GPU backend must actually
 * have registered a device. In DL builds ggml SKIPS broken modules
 * silently in release builds (missing CUDA runtime, ABI mismatch) and
 * would proceed on CPU — the exact silent-fallback class LLOYAL_NO_FALLBACK
 * exists to kill, one layer down.
 */
const assertRequestedDevices = (
  binding: NativeBinding,
  requestedVariant: string | undefined,
): NativeBinding => {
  const wantsGpu = requestedVariant === "cuda" || requestedVariant === "vulkan";
  if (!wantsGpu || typeof binding.listDevices !== "function") return binding;
  const devices = binding.listDevices();
  if (!devices.some((d: BackendDevice) => d.type === "gpu")) {
    throw new Error(
      `[lloyal.node] GPU variant "${requestedVariant}" was requested but no GPU ` +
        `device registered after backend load (devices: ` +
        `${devices.map((d) => `${d.name}:${d.type}`).join(", ") || "none"}). ` +
        `Likely a missing/skewed CUDA runtime — the backend module was ` +
        `silently skipped at dlopen. Refusing to run on CPU.`,
    );
  }
  return binding;
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
  loadOptions?: LoadOptions,
): Promise<SessionContext> => {
  const variant = loadOptions?.gpuVariant || process.env.LLOYAL_GPU;
  const binary = variant ? loadBinary(variant as GpuVariant) : getBinary();
  return binary.createContext(options);
};

// ── Re-export from @lloyal-labs/sdk ──────────────────────────────
export {
  Branch,
  BranchStore,
  Session,
  Rerank,
  buildUserDelta,
  buildToolResultDelta,
} from "@lloyal-labs/sdk";

export {
  PoolingType,
  CHAT_FORMAT_CONTENT_ONLY,
  CHAT_FORMAT_GENERIC,
  ReasoningFormat,
  GrammarTriggerType,
} from "@lloyal-labs/sdk";
export type { ChatFormat } from "@lloyal-labs/sdk";
export type {
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
  RerankOptions,
  RerankResult,
  RerankProgress,
  KvCacheType,
} from "@lloyal-labs/sdk";

// ── Re-export from @lloyal-labs/lloyal-agents ────────────────────
export {
  Ctx,
  Store,
  Events,
  Tool,
  Agent,
  agent,
  agentPool,
  useAgent,
  useAgentPool,
  reduce,
  diverge,
  createToolkit,
  initAgents,
  withSpine,
  DefaultAgentPolicy,
  renderTemplate,
} from "@lloyal-labs/lloyal-agents";

export type {
  Toolkit,
  AgentHandle,
  SpineOptions,
  JsonSchema,
  ToolSchema,
  ToolContext,
  AgentTaskSpec,
  AgentPoolOptions,
  AgentResult,
  AgentPoolResult,
  DivergeOptions,
  DivergeAttempt,
  DivergeResult,
  AgentEvent,
  UseAgentOpts,
  CreateAgentPoolOpts,
  SpawnSpec,
} from "@lloyal-labs/lloyal-agents";

// ── Backend pack (BACKEND_DL flavor acquisition) ─────────────────
export {
  ensureBackendPack,
  probeBackendPack,
  resolveBackendPackDirSync,
  backendPackCacheDir,
} from "./backend-pack";
export type {
  BackendPackManifest,
  BackendPackProbe,
  BackendPackPlatform,
  EnsureBackendPackOpts,
  ArchiveRef,
  GpuInfo,
} from "./backend-pack";

// ── Native-only types (stay in lloyal.node) ──────────────────────
export type { BackendDevice, GpuVariant, LoadOptions, NativeBinding } from "./types";
