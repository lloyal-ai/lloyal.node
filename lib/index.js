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
 * // Decode
 * await ctx.decode(tokens, 0);
 *
 * // Safe logits access (Runtime Borrow Checker pattern)
 * const entropy = withLogits(ctx, (logits) => {
 *   // logits is valid here - use synchronously only!
 *   return computeEntropy(logits);
 * });
 *
 * // Or with native reference implementations (for testing)
 * const nativeEntropy = ctx.computeEntropy();
 * const token = ctx.greedySample();
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
  const suffix = variant && variant !== 'default' ? `-${variant}` : '';
  return `@lloyal-labs/lloyal.node-${platform}-${arch}${suffix}`;
};

/**
 * Try to load a platform package, return null on failure.
 * Failures include: package not installed, missing GPU runtime libs (dlopen fails)
 * @param {string} packageName - Package name to load
 * @returns {object|null} The native binary module or null
 */
const tryLoadPackage = (packageName) => {
  try {
    return require(packageName);
  } catch (e) {
    return null;
  }
};

/**
 * Load the native binary with automatic fallback.
 *
 * Loading priority:
 * 1. Requested GPU variant (if specified)
 * 2. Default platform package (CPU)
 * 3. Local build (development: build/Release/lloyal.node)
 *
 * @param {string} [variant] - GPU variant: 'cuda', 'vulkan', or undefined for CPU
 * @returns {object} The native binary module
 * @throws {Error} If no binary can be loaded
 */
const loadBinary = (variant) => {
  // 1. Try requested variant (if specified)
  if (variant && variant !== 'default') {
    const pkgName = getPlatformPackageName(variant);
    const binary = tryLoadPackage(pkgName);
    if (binary) return binary;

    console.warn(`[lloyal.node] GPU variant "${variant}" unavailable, falling back to CPU`);
  }

  // 2. Try default platform package (CPU)
  const defaultPkg = getPlatformPackageName();
  const binary = tryLoadPackage(defaultPkg);
  if (binary) return binary;

  // 3. Try local build (development)
  try {
    return require('../build/Release/lloyal.node');
  } catch (e) {
    // ignore
  }

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

module.exports = {
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
