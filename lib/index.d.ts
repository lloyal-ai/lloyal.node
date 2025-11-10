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
 * Result from formatChat operation
 */
export interface FormattedChatResult {
  /** Formatted prompt string ready for tokenization */
  prompt: string;

  /** Additional stop strings from chat template (e.g., "<|im_end|>") */
  stopTokens: string[];
}

/**
 * Sampling parameters for native sampling
 */
export interface SamplingParams {
  /** Randomness (0.0 = deterministic, 2.0 = very random) */
  temperature?: number;

  /** Only consider top K tokens (0 = disabled) */
  topK?: number;

  /** Nucleus sampling threshold (1.0 = disabled) */
  topP?: number;

  /** Minimum probability threshold */
  minP?: number;

  /** Random seed for reproducible generation */
  seed?: number;

  /** Penalty parameters for repetition control */
  penalties?: {
    /** Repetition penalty (1.0 = disabled, >1.0 = penalize repeats) */
    repeat?: number;

    /** Frequency penalty (0.0 = disabled) */
    frequency?: number;

    /** Presence penalty (0.0 = disabled) */
    presence?: number;

    /** Tokens to consider for penalties (-1 = context size) */
    lastN?: number;
  };
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

  /**
   * Convert single token to text (sync, fast)
   *
   * Optimized for per-token conversion during generation.
   * For batch conversion, use detokenize().
   *
   * @param token Token ID
   * @returns Text piece for this token
   */
  tokenToText(token: number): string;

  /**
   * Check if token is a stop token (EOS, EOT, etc.)
   *
   * Checks vocabulary stop tokens. For custom stop sequences,
   * compare generated text in application code.
   *
   * @param token Token ID to check
   * @returns True if token marks end of generation
   */
  isStopToken(token: number): boolean;

  /**
   * Format messages using model's chat template
   *
   * Converts [{role, content}] → formatted prompt string.
   * Uses model's built-in template (ChatML, Llama, Mistral, etc.).
   *
   * @param messagesJson JSON string containing array of messages
   * @param templateOverride Optional custom template string
   * @returns Formatted prompt and stop tokens from template
   *
   * @example
   * ```typescript
   * const messages = JSON.stringify([
   *   { role: "system", content: "You are a helpful assistant." },
   *   { role: "user", content: "What is the capital of France?" }
   * ]);
   *
   * const result = await ctx.formatChat(messages);
   * // result.prompt: "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n..."
   * // result.stopTokens: ["<|im_end|>"]
   *
   * const tokens = await ctx.tokenize(result.prompt);
   * ```
   */
  formatChat(messagesJson: string, templateOverride?: string): Promise<FormattedChatResult>;

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

  /**
   * Sample with full parameters (native implementation for benchmarking)
   *
   * Full native sampling with temperature, top-k, top-p, penalties, etc.
   * Useful for benchmarking TypeScript sampler implementations.
   *
   * @param params Sampling parameters (all optional, uses defaults if not provided)
   * @returns Sampled token ID
   *
   * @example
   * ```typescript
   * // Greedy (equivalent to greedySample)
   * const token = ctx.sample({ temperature: 0 });
   *
   * // Creative
   * const token = ctx.sample({
   *   temperature: 0.9,
   *   topK: 40,
   *   topP: 0.95,
   *   penalties: { repeat: 1.1 }
   * });
   * ```
   */
  sample(params?: SamplingParams): number;

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
