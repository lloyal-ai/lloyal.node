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
 * const { createContext } = require('liblloyal-node');
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
 * // Get raw logits (zero-copy Float32Array)
 * const logits = ctx.getLogits();
 *
 * // Native reference implementations (for testing)
 * const entropy = ctx.computeEntropy();
 * const token = ctx.greedySample();
 *
 * // Cleanup
 * ctx.dispose();
 * ```
 */

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

  SessionContext: binary.SessionContext
};
