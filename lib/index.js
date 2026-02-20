/**
 * liblloyal-node - Thin N-API wrapper over liblloyal
 *
 * Exposes raw llama.cpp inference primitives for Node.js.
 * Primary use case: Integration testing for tsampler.
 *
 * @example
 * ```js
 * const { createContext, withLogits } = require('@lloyal-labs/lloyal.node');
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
 *
 * @example GPU variant selection
 * ```js
 * // Option 1: Environment variable (affects all contexts)
 * // Set LLOYAL_GPU=cuda before running
 *
 * // Option 2: Per-context selection (recommended)
 * const ctx = await createContext(
 *   { modelPath: './model.gguf', nCtx: 4096 },
 *   { gpuVariant: 'cuda' }  // Falls back to CPU if CUDA unavailable
 * );
 * ```
 */

/**
 * Platform package naming: @lloyal-labs/lloyal.node-{platform}-{arch}[-{gpu}]
 * @param {string} [variant] - GPU variant: 'cuda', 'vulkan', or undefined for CPU
 * @returns {string} Platform package name
 */
const getPlatformPackageName = (variant) => {
  const platform = process.platform;
  const arch = process.arch;
  // cpu/metal/default = no suffix, cuda/vulkan = suffix
  const noSuffix = !variant || variant === 'default' || variant === 'cpu' || variant === 'metal';
  const suffix = noSuffix ? '' : `-${variant}`;
  return `@lloyal-labs/lloyal.node-${platform}-${arch}${suffix}`;
};

/**
 * Try to load a platform package, return null on failure.
 * Failures include: package not installed, missing GPU runtime libs (dlopen fails),
 * or module doesn't export expected interface.
 * @param {string} packageName - Package name to load
 * @param {boolean} [verbose=false] - Log failure reasons
 * @returns {object|null} The native binary module or null
 */
const tryLoadPackage = (packageName, verbose = false) => {
  try {
    const mod = require(packageName);
    // Validate it's actually a native module with expected exports
    if (mod && typeof mod.createContext === 'function') {
      return mod;
    }
    if (verbose) {
      console.warn(`[lloyal.node] ${packageName} loaded but missing createContext export`);
    }
    return null;
  } catch (e) {
    if (verbose) {
      console.warn(`[lloyal.node] Failed to load ${packageName}: ${e.message}`);
    }
    return null;
  }
};

/**
 * Load the native binary with automatic fallback.
 *
 * **Loading Priority:**
 *
 * When `LLOYAL_LOCAL=1`:
 * - Uses local build exclusively (`build/Release/lloyal.node`)
 * - Throws error if not found (no fallback)
 *
 * Otherwise:
 * 1. Requested GPU variant package (if `variant` param or `LLOYAL_GPU` env var specified)
 * 2. Local build (`build/Release/lloyal.node`) — always fresher during development
 * 3. Default platform package (`@lloyal-labs/lloyal.node-{platform}-{arch}`)
 *
 * **Environment Variables:**
 * - `LLOYAL_LOCAL=1` — Use local build exclusively (`build/Release/lloyal.node`).
 *   Throws an error if local build not found. Use during development to test
 *   local changes without uninstalling npm packages.
 * - `LLOYAL_GPU` — GPU variant to load: `'cuda'` or `'vulkan'`. Equivalent to
 *   passing the `variant` parameter.
 * - `LLOYAL_NO_FALLBACK=1` — Disable fallback when GPU variant fails. Throws an
 *   error instead of silently falling back to CPU. Use in CI to ensure the
 *   specific GPU package loads correctly and catch missing runtime libraries.
 *
 * @param {string} [variant] - GPU variant: `'cuda'`, `'vulkan'`, or `undefined` for CPU.
 *   Overrides `LLOYAL_GPU` env var if specified.
 * @returns {object} The native binary module with `createContext` and `SessionContext`
 * @throws {Error} If no binary can be loaded for the current platform
 *
 * @example Development testing with local build
 * ```bash
 * # Build locally, then test without uninstalling npm packages
 * npm run build
 * LLOYAL_LOCAL=1 node my-script.js
 * ```
 *
 * @example GPU variant selection
 * ```bash
 * # Via environment variable
 * LLOYAL_GPU=cuda node my-script.js
 *
 * # Or programmatically
 * const binary = loadBinary('cuda');
 * ```
 *
 * @example CI: Ensure GPU package loads (no silent fallback)
 * ```bash
 * LLOYAL_GPU=cuda LLOYAL_NO_FALLBACK=1 npm test
 * ```
 */
const loadBinary = (variant) => {
  // Use env var if no variant specified
  variant = variant ?? process.env.LLOYAL_GPU;
  // LLOYAL_NO_FALLBACK=1 disables fallback (for CI testing specific packages)
  const noFallback = process.env.LLOYAL_NO_FALLBACK === '1';
  // LLOYAL_LOCAL=1 forces local build first (development)
  const useLocal = process.env.LLOYAL_LOCAL === '1';

  // 0. Use local build if explicitly requested (no fallback)
  if (useLocal) {
    try {
      return require('../build/Release/lloyal.node');
    } catch (e) {
      throw new Error(
        '[lloyal.node] LLOYAL_LOCAL=1 but local build not found. ' +
        'Run `npm run build` first.'
      );
    }
  }

  // 1. Try requested variant (if specified)
  if (variant && variant !== 'default') {
    const pkgName = getPlatformPackageName(variant);
    const binary = tryLoadPackage(pkgName, true); // verbose=true to see errors
    if (binary) return binary;

    if (noFallback) {
      throw new Error(
        `[lloyal.node] GPU variant "${variant}" failed to load. ` +
        `Package: ${pkgName}. Check that runtime libraries are available.`
      );
    }
    console.warn(`[lloyal.node] GPU variant "${variant}" unavailable, falling back to CPU`);
  }

  // 2. Try local build (always fresher than installed packages during development)
  try {
    return require('../build/Release/lloyal.node');
  } catch (e) {
    // ignore — no local build
  }

  // 3. Try default platform package (CPU)
  const defaultPkg = getPlatformPackageName();
  const binary = tryLoadPackage(defaultPkg, true); // verbose=true
  if (binary) return binary;

  throw new Error(
    `No lloyal.node binary found for ${process.platform}-${process.arch}. ` +
    `Tried: ${variant ? getPlatformPackageName(variant) + ', ' : ''}${defaultPkg}`
  );
};

// Default binary (loaded lazily on first use)
let _binary = null;
const getBinary = () => {
  if (!_binary) {
    _binary = loadBinary(process.env.LLOYAL_GPU);
  }
  return _binary;
};

/**
 * Safe logits access with Runtime Borrow Checker pattern
 *
 * Ensures logits are only accessed synchronously within the callback.
 * The callback MUST NOT:
 * - Store the logits reference
 * - Return a Promise (will throw)
 * - Call decode() (would invalidate logits)
 *
 * This is a "runtime borrow checker" - it prevents async mutations
 * while you're working with borrowed logits.
 *
 * @template T
 * @param {SessionContext} ctx - The session context
 * @param {(logits: Float32Array) => T} fn - Synchronous callback that uses logits
 * @returns {T} The result from the callback
 * @throws {Error} If callback returns a Promise (async usage not allowed)
 *
 * @example
 * ```js
 * // Safe: synchronous computation
 * const entropy = withLogits(ctx, (logits) => {
 *   let sum = 0;
 *   for (let i = 0; i < logits.length; i++) {
 *     sum += Math.exp(logits[i]);
 *   }
 *   return Math.log(sum);
 * });
 *
 * // ERROR: callback returns Promise (will throw)
 * withLogits(ctx, async (logits) => {
 *   await something();  // NOT ALLOWED
 *   return logits[0];
 * });
 * ```
 */
function withLogits(ctx, fn) {
  // Get logits (memoized - same buffer if called twice in same step)
  const logits = ctx.getLogits();

  // Execute user callback with logits
  const result = fn(logits);

  // Detect async usage (not allowed - logits would be invalidated)
  if (result && typeof result.then === 'function') {
    throw new Error(
      'withLogits callback must be synchronous. ' +
      'Returning a Promise is not allowed because logits become invalid after decode(). ' +
      'Complete all logits processing synchronously within the callback.'
    );
  }

  return result;
}

const { Branch } = require('./Branch');
const { BranchStore } = require('./BranchStore');

module.exports = {
  /**
   * Branch class for parallel generation
   * @see Branch.create()
   */
  Branch,
  /**
   * BranchStore class for batched multi-branch decode
   * @see BranchStore
   */
  BranchStore,
  /**
   * Create a new inference context
   *
   * @param {ContextOptions} options - Context configuration
   * @param {LoadOptions} [loadOptions] - Binary loading options
   * @returns {Promise<SessionContext>} The inference context
   *
   * @example
   * ```js
   * // Basic usage
   * const ctx = await createContext({
   *   modelPath: './model.gguf',
   *   nCtx: 2048,
   *   nThreads: 4
   * });
   *
   * // With GPU variant
   * const ctx = await createContext(
   *   { modelPath: './model.gguf' },
   *   { gpuVariant: 'cuda' }
   * );
   * ```
   */
  createContext: async (options, loadOptions) => {
    const variant = loadOptions?.gpuVariant || process.env.LLOYAL_GPU;
    const binary = variant ? loadBinary(variant) : getBinary();
    return binary.createContext(options);
  },

  /**
   * Load binary for a specific GPU variant.
   * Useful for checking variant availability before creating context.
   *
   * @param {string} [variant] - 'cuda', 'vulkan', or undefined for CPU
   * @returns {object} Native binary module
   * @throws {Error} If no binary available for platform
   *
   * @example
   * ```js
   * // Load default (CPU) binary
   * const binary = loadBinary();
   *
   * // Load CUDA binary (falls back to CPU if unavailable)
   * const binary = loadBinary('cuda');
   * ```
   */
  loadBinary,

  /**
   * Safe logits access with Runtime Borrow Checker pattern.
   * See function JSDoc for full documentation.
   */
  withLogits,
};
