const path = require('path');
const binary = require('node-gyp-build')(path.join(__dirname, '..'));

/**
 * liblloyal-node - Thin N-API wrapper over liblloyal
 *
 * Exposes raw llama.cpp inference primitives for Node.js.
 * Primary use case: Integration testing for tsampler.
 *
 * @example
 * ```js
 * const { createContext, withLogits } = require('lloyal.node');
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
 * const entropy = await withLogits(ctx, (logits) => {
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
 */

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
   * @param {Object} options
   * @param {string} options.modelPath - Path to .gguf model file
   * @param {number} [options.nCtx=2048] - Context size
   * @param {number} [options.nThreads=4] - Number of threads
   * @returns {Promise<SessionContext>}
   */
  createContext: async (options) => {
    // For now, createContext is synchronous in C++
    // Wrap in Promise for future async model loading
    return binary.createContext(options);
  },

  /**
   * Safe logits access with Runtime Borrow Checker pattern
   *
   * Ensures logits are only accessed synchronously within the callback.
   * See function JSDoc for full documentation.
   */
  withLogits,

  SessionContext: binary.SessionContext
};
