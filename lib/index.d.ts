/**
 * liblloyal-node - Thin N-API wrapper over liblloyal
 *
 * TypeScript definitions for raw llama.cpp inference primitives.
 */

/**
 * Options for creating an inference context
 */
export interface ContextOptions {
  /** Path to .gguf model file */
  modelPath: string;

  /** Context size (default: 2048) */
  nCtx?: number;

  /** Number of threads (default: 4) */
  nThreads?: number;
}

/**
 * Inference context - raw llama.cpp primitives
 *
 * Lifetime:
 * - Call dispose() when done to free native resources
 * - getLogits() returns zero-copy buffer valid until next decode()
 */
export interface SessionContext {
  // ===== CORE PRIMITIVES =====

  /**
   * Get raw logits (zero-copy Float32Array)
   *
   * WARNING: Buffer is only valid until next decode() call!
   * Copy the data if you need to retain it across async boundaries.
   *
   * @returns Float32Array pointing to llama.cpp memory
   */
  getLogits(): Float32Array;

  /**
   * Decode tokens through the model (forward pass)
   *
   * @param tokens Array of token IDs
   * @param position Position in sequence where tokens start
   */
  decode(tokens: number[], position: number): Promise<void>;

  /**
   * Tokenize text to token IDs
   *
   * @param text Text to tokenize
   * @returns Array of token IDs
   */
  tokenize(text: string): Promise<number[]>;

  /**
   * Detokenize tokens to text
   *
   * @param tokens Array of token IDs
   * @returns Reconstructed text
   */
  detokenize(tokens: number[]): Promise<string>;

  // ===== NATIVE REFERENCE IMPLEMENTATIONS =====

  /**
   * Compute entropy of current logits distribution (native reference)
   *
   * Uses numerically stable log-sum-exp implementation.
   * Useful for validating TS sampler implementations.
   *
   * @returns Entropy in nats
   */
  computeEntropy(): number;

  /**
   * Sample greedily from current logits (native reference)
   *
   * Selects token with highest logit value.
   * Useful for validating TS sampler implementations.
   *
   * @returns Token ID
   */
  greedySample(): number;

  // ===== LIFECYCLE =====

  /**
   * Free native resources
   *
   * Call this when done with the context to avoid memory leaks.
   * Context is unusable after calling dispose().
   */
  dispose(): void;

  // ===== PROPERTIES =====

  /** Model vocabulary size */
  readonly vocabSize: number;
}

/**
 * Create a new inference context
 *
 * @param options Context configuration
 * @returns SessionContext instance
 *
 * @example
 * ```typescript
 * const ctx = await createContext({
 *   modelPath: './model.gguf',
 *   nCtx: 2048,
 *   nThreads: 4
 * });
 *
 * try {
 *   const tokens = await ctx.tokenize("Hello");
 *   await ctx.decode(tokens, 0);
 *   const logits = ctx.getLogits();
 *   const entropy = ctx.computeEntropy();
 * } finally {
 *   ctx.dispose();
 * }
 * ```
 */
export function createContext(options: ContextOptions): Promise<SessionContext>;
