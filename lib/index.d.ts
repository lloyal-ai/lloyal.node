/**
 * liblloyal-node TypeScript Definitions
 *
 * N-API bindings for liblloyal - Node.js native addon for llama.cpp inference
 * API adapted from @lloyal/nitro-llama SessionContext for full feature parity
 */

/**
 * Pooling type for embedding extraction
 */
export enum PoolingType {
  /** No pooling - raw per-token embeddings */
  NONE = 0,
  /** Mean pooling - average of all token embeddings */
  MEAN = 1,
  /** CLS pooling - use first token embedding */
  CLS = 2,
  /** Last token pooling - use last token embedding */
  LAST = 3,
}

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

  /**
   * Enable embedding extraction mode
   *
   * When true, context is optimized for embedding extraction.
   * Use with encode() and getEmbeddings() methods.
   * Default: false (text generation mode)
   */
  embeddings?: boolean;

  /**
   * Pooling type for embedding extraction
   *
   * Only relevant when embeddings=true.
   * Default: MEAN for embedding contexts, NONE otherwise
   */
  poolingType?: PoolingType;

  /**
   * Maximum number of sequences for multi-sequence support
   *
   * Set > 1 to enable multiple independent KV cache sequences.
   * When sequences share a common prefix, kv_unified=true (default) is optimal.
   * Default: 1 (single sequence)
   */
  nSeqMax?: number;
}

/**
 * Result from chat template formatting
 */
export interface FormattedChatResult {
  prompt: string;
  stopTokens: string[];
}

/**
 * Penalty parameters for repetition control
 */
export interface PenaltyParams {
  /** Repetition penalty (1.0 = disabled, >1.0 = penalize repeats) */
  repeat?: number;

  /** Frequency penalty (0.0 = disabled) */
  frequency?: number;

  /** Presence penalty (0.0 = disabled) */
  presence?: number;

  /** Tokens to consider for penalties (-1 = context size) */
  lastN?: number;
}

/**
 * Mirostat sampling configuration
 */
export interface MirostatParams {
  /** Mirostat mode (0 = disabled, 1 = v1, 2 = v2) */
  mode?: number;

  /** Target entropy */
  tau?: number;

  /** Learning rate */
  eta?: number;
}

/**
 * DRY (Don't Repeat Yourself) sampling parameters
 */
export interface DryParams {
  multiplier?: number;
  base?: number;
  allowedLength?: number;
  penaltyLastN?: number;
}

/**
 * XTC sampler parameters
 */
export interface XtcParams {
  probability?: number;
  threshold?: number;
}

/**
 * Advanced sampling parameters
 */
export interface AdvancedSamplingParams {
  /** Locally typical sampling (1.0 = disabled) */
  typicalP?: number;

  /** Mirostat sampling configuration */
  mirostat?: MirostatParams;

  /** DRY (Don't Repeat Yourself) sampling */
  dry?: DryParams;

  /** XTC sampler */
  xtc?: XtcParams;
}

/**
 * Sampling parameters for token generation
 *
 * Common presets:
 * - Factual/Precise: { temperature: 0.1 }
 * - Balanced: { temperature: 0.7 }
 * - Creative: { temperature: 1.0 }
 */
export interface SamplingParams {
  // ===== COMMON CONTROLS =====

  /** Randomness (0.0 = always most likely, 2.0 = very random) */
  temperature?: number;

  /** Only consider top K most likely tokens (0 = disabled) */
  topK?: number;

  /** Nucleus sampling threshold (1.0 = disabled) */
  topP?: number;

  /** Minimum probability threshold */
  minP?: number;

  /** Random seed for reproducible generation (-1 = random) */
  seed?: number;

  /** GBNF grammar string for constrained generation */
  grammar?: string;

  // ===== GROUPED CONTROLS =====

  /** Penalty parameters for repetition control */
  penalties?: PenaltyParams;

  /** Advanced sampling parameters */
  advanced?: AdvancedSamplingParams;
}

/**
 * A llama.cpp context for text generation
 *
 * Thread safety: All synchronous methods are safe on JS thread.
 * Async methods automatically run on Node.js worker threads.
 */
export interface SessionContext {
  // ===== THE GENERATION LOOP =====

  /**
   * STEP 1: Process tokens through the model (forward pass)
   *
   * This feeds tokens through the transformer and updates the KV cache.
   * After decoding, the model has "read" this text and is ready to predict.
   *
   * Think of this as: "the model reads your prompt"
   *
   * Why async? Model inference takes time (~45ms per token)
   * Why position? Model needs to know where in conversation this text appears
   *
   * Cost: ~45ms per token (generation), ~120ms for 50 tokens (prompt)
   *
   * @param tokens Token IDs from tokenize()
   * @param position Where these tokens start in the sequence
   * @param seqId Sequence ID (default: 0)
   * @example
   * ```typescript
   * const tokens = await ctx.tokenize("Hello world");
   * await ctx.decode(tokens, 0);
   * let position = tokens.length;
   *
   * // Generate next token
   * await ctx.decode([nextToken], position++);
   *
   * // Multi-sequence: decode to different sequences
   * await ctx.decode(tokens, 0, 0);  // Sequence 0
   * await ctx.decode(tokens, 0, 1);  // Sequence 1
   * ```
   */
  decode(tokens: number[], position: number, seqId?: number): Promise<void>;

  /**
   * STEP 2a: Get token scores for custom sampling (zero-copy, mutable)
   *
   * Returns unnormalized scores for every possible next token.
   * Higher score = model thinks this token is more likely.
   *
   * Use this for custom sampling logic or grammar-constrained generation.
   * For reading scores (entropy computation), use getLogits() instead.
   *
   * ⚠️ CRITICAL LIFETIME CONSTRAINTS:
   * - This is a zero-copy buffer (points directly to model memory)
   * - Valid ONLY until next decode() call
   * - NOT thread-safe - use only on JS thread
   * - DO NOT retain reference across async boundaries
   * - Buffer is invalidated by: decode(), sample() with grammar
   *
   * Cost: ~0.5ms (zero-copy pointer)
   *
   * @returns Buffer containing vocabSize floats (Float32Array compatible)
   * @example Safe usage
   * ```typescript
   * const buffer = ctx.getTokenScores();
   * const scores = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
   *
   * // Modify immediately (safe - still on JS thread)
   * scores[BANNED_TOKEN] = -Infinity;
   *
   * // Use immediately
   * const token = customSample(scores);
   *
   * // Now decode invalidates the buffer
   * await ctx.decode([token], position++);
   * // Buffer is now INVALID - do not access!
   * ```
   */
  getTokenScores(): Buffer;

  /**
   * STEP 2b: Get logits for reading (zero-copy, readonly usage pattern)
   *
   * Returns Float32Array for computational tasks like entropy calculation.
   * For custom sampling or grammar, use getTokenScores() instead.
   *
   * WARNING: Buffer is only valid until next decode() call!
   *
   * @returns Float32Array pointing to llama.cpp memory
   */
  getLogits(): Float32Array;

  /**
   * STEP 3: Sample a token from scores
   *
   * Converts raw scores into a token decision using:
   * - Temperature: controls randomness
   * - Top-K/Top-P: filters unlikely tokens
   * - Grammar: enforces format constraints (if grammar initialized)
   *
   * This is where generation strategy happens.
   *
   * Cost: ~0.1ms (native sampling)
   *
   * @param params Sampling strategy (greedy if omitted)
   * @returns Selected token ID
   * @example
   * ```typescript
   * // Greedy (always pick most likely)
   * const token = ctx.sample();
   *
   * // Creative generation
   * const token = ctx.sample({ temperature: 0.9 });
   *
   * // Constrained to valid JSON
   * ctx.initGrammar(grammar);
   * const token = ctx.sample({ temperature: 0.7 });
   * ```
   */
  sample(params?: SamplingParams): number;

  /**
   * Convert token ID to text piece
   *
   * Fast synchronous lookup in vocabulary table.
   * Call this on each generated token for streaming display.
   *
   * Uses llama_token_to_piece() internally - optimized for per-token
   * conversion during generation. For batch conversion of many tokens,
   * use detokenize() instead.
   *
   * Cost: ~0.05ms
   *
   * @param token Token ID from sample()
   * @returns Text string for this token
   * @example
   * ```typescript
   * while (true) {
   *   const token = ctx.sample({ temperature: 0.8 });
   *   if (ctx.isStopToken(token)) break;
   *
   *   const text = ctx.tokenToText(token);
   *   process.stdout.write(text); // Stream to output
   *
   *   await ctx.decode([token], position++);
   * }
   * ```
   */
  tokenToText(token: number): string;

  /**
   * Check if token is a model stop token
   *
   * Returns true for built-in end-of-generation tokens:
   * - </s> (Llama 2)
   * - <|endoftext|> (GPT)
   * - <|eot_id|> (Llama 3)
   * - Model-specific EOS tokens
   *
   * Note: This checks vocabulary stop tokens, not custom stop sequences.
   * For custom stops (e.g., "\n\n", "###"), compare generated text
   * against your stop strings in application code.
   *
   * Cost: <0.01ms (fast vocabulary lookup)
   *
   * @param token Token ID to check
   * @example
   * ```typescript
   * const token = ctx.sample();
   * if (ctx.isStopToken(token)) {
   *   console.log('Generation complete');
   *   break;
   * }
   * ```
   */
  isStopToken(token: number): boolean;

  // ===== PROMPT PREPARATION =====

  /**
   * Tokenize text into model's vocabulary
   *
   * Converts human text → token IDs for decode().
   * Same text always produces same tokens for a given model.
   *
   * Cost: ~1ms per 100 characters
   *
   * @param text Text to tokenize
   * @returns Array of token IDs
   * @example
   * ```typescript
   * const tokens = await ctx.tokenize("Hello world");
   * console.log(tokens); // [15496, 1917] for Llama models
   *
   * await ctx.decode(tokens, 0);
   * ```
   */
  tokenize(text: string): Promise<number[]>;

  /**
   * Detokenize array of tokens back to text
   *
   * Inverse of tokenize(). Use for reconstructing complete text
   * from token sequences (e.g., after KV cache operations).
   *
   * Uses llama_detokenize() for efficient batch conversion.
   * For single-token conversion during generation, use tokenToText().
   *
   * Cost: ~1ms per 100 tokens
   *
   * @param tokens Array of token IDs
   * @returns Complete text representation
   * @example
   * ```typescript
   * const tokens = [15496, 1917]; // "Hello world"
   * const text = await ctx.detokenize(tokens);
   * console.log(text); // "Hello world"
   * ```
   */
  detokenize(tokens: number[]): Promise<string>;

  // ===== KV CACHE MANAGEMENT =====

  /**
   * Get current sequence length (number of decoded tokens)
   *
   * The KV cache stores model state for all decoded tokens.
   * This tells you how many tokens are currently in memory.
   *
   * Think of this as: "How much has the model read so far?"
   *
   * Cost: <0.01ms (fast sync operation - safe to call frequently)
   *
   * @param sequenceId Sequence ID (defaults to 0 for single conversation)
   * @returns Number of tokens in cache, or -1 if empty
   * @example
   * ```typescript
   * const tokens = await ctx.tokenize("Hello world");
   * await ctx.decode(tokens, 0);
   *
   * const length = ctx.kvCacheSize(0);
   * console.log(length); // 2 (number of tokens)
   * ```
   */
  kvCacheSize(sequenceId?: number): number;

  /**
   * Remove token range from KV cache
   *
   * Deletes tokens from model's memory. Use cases:
   * - Removing old context when hitting limit (sliding window)
   * - Implementing conversation pruning
   * - Forgetting specific messages
   * - Preparing for injection of new context
   *
   * ⚠️ CRITICAL: Call BEFORE next decode(), not after!
   * The model needs to know about the removal before processing new tokens.
   *
   * Cost: ~1-5ms depending on range
   *
   * @param sequenceId Sequence ID (use 0 for single sequence)
   * @param start Start position (inclusive)
   * @param end End position (exclusive), -1 = to end
   * @example
   * ```typescript
   * // Remove old tokens to stay under context limit
   * const currentLength = ctx.kvCacheSize(0);
   * if (currentLength > 2000) {
   *   // Remove oldest 500 tokens
   *   await ctx.kvCacheRemove(0, 0, 500);
   *
   *   // THEN decode new tokens
   *   await ctx.decode(newTokens, currentLength - 500);
   * }
   * ```
   */
  kvCacheRemove(sequenceId: number, start: number, end: number): Promise<void>;

  /**
   * Snapshot KV cache state for branching/undo
   *
   * Serializes entire model state to Buffer.
   * Restore later with kvCacheLoad() for:
   * - Conversation branching ("what if I said X instead?")
   * - Undo/redo functionality
   * - Checkpointing long conversations
   *
   * Size: ~500MB-2GB depending on context length and model
   *
   * Cost: ~100-500ms depending on cache size
   *
   * @param sequenceId Sequence ID (use 0 for single sequence)
   * @returns Serialized state buffer
   * @example
   * ```typescript
   * // Save state before risky operation
   * const snapshot = await ctx.kvCacheSave(0);
   *
   * // Try something
   * await ctx.decode(riskyTokens, position);
   *
   * // Didn't work - restore previous state
   * await ctx.kvCacheLoad(0, snapshot);
   * ```
   */
  kvCacheSave(sequenceId?: number): Promise<Buffer>;

  /**
   * Restore KV cache from previous snapshot
   *
   * Loads saved model state. Context returns to exact state
   * when snapshot was taken.
   *
   * Cost: ~100-500ms depending on snapshot size
   *
   * @param sequenceId Sequence ID (use 0 for single sequence)
   * @param state Buffer from kvCacheSave()
   * @example
   * ```typescript
   * const snapshot = await ctx.kvCacheSave(0);
   *
   * // ... many operations later ...
   *
   * // Restore to saved state
   * await ctx.kvCacheLoad(0, snapshot);
   * ```
   */
  kvCacheLoad(sequenceId: number, state: Buffer): Promise<void>;

  /**
   * Clear all KV cache (fresh start)
   *
   * Removes all cached tokens. Model returns to initial state
   * as if no text has been processed.
   *
   * Use when starting a completely new conversation.
   *
   * Cost: ~1ms
   *
   * @example
   * ```typescript
   * // Start fresh conversation
   * await ctx.kvCacheClear();
   *
   * const tokens = await ctx.tokenize("New conversation");
   * await ctx.decode(tokens, 0);
   * ```
   */
  kvCacheClear(): Promise<void>;

  // ===== GRAMMAR-CONSTRAINED GENERATION =====

  /**
   * Initialize grammar parser (once per generation session)
   *
   * Grammars constrain generation to valid formats (JSON, XML, etc.).
   * Parser tracks state across tokens to enforce rules.
   *
   * Call once before starting constrained generation.
   * Use resetGrammar() to reuse same grammar for new generation.
   *
   * Cost: ~0.1-1ms depending on grammar complexity
   *
   * @param grammarStr GBNF grammar string (EBNF-like syntax)
   * @example
   * ```typescript
   * // Force valid JSON
   * const grammar = ctx.jsonSchemaToGrammar(JSON.stringify({
   *   type: "object",
   *   properties: {
   *     name: { type: "string" },
   *     age: { type: "number" }
   *   }
   * }));
   *
   * ctx.initGrammar(grammar);
   *
   * // Now sample() will only generate valid JSON
   * const token = ctx.sample({ temperature: 0.7 });
   * ```
   */
  initGrammar(grammarStr: string): void;

  /**
   * Apply grammar constraints to token scores (modifies in-place)
   *
   * Masks invalid tokens with -Infinity based on parser state.
   * Call after getTokenScores(), before custom sampling.
   *
   * Flow: getTokenScores() → applyGrammar() → sample() → acceptToken()
   *
   * Thread safety: This method is synchronous and modifies the buffer
   * in-place on the JS thread. Safe because it's called sequentially
   * in the generation loop before any async operations.
   *
   * Cost: ~0.1-1ms depending on grammar complexity
   *
   * @param scoresBuffer Buffer from getTokenScores() (modified in-place)
   * @throws Error if grammar not initialized (call initGrammar first)
   * @example
   * ```typescript
   * // Custom sampling with grammar
   * const buffer = ctx.getTokenScores();
   * const scores = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
   *
   * // Apply grammar constraints
   * ctx.applyGrammar(buffer);
   *
   * // Now sample from constrained distribution
   * const token = customSample(scores);
   * ctx.acceptToken(token);
   * ```
   */
  applyGrammar(scoresBuffer: Buffer): void;

  /**
   * Advance grammar parser with chosen token
   *
   * Updates parser state after sampling.
   * MUST be called AFTER sampling, BEFORE next applyGrammar().
   *
   * This advances the stateful grammar parser through its rules.
   * Without this, grammar constraints will be incorrect.
   *
   * Cost: <0.01ms
   *
   * @param tokenId Token that was sampled
   * @example
   * ```typescript
   * const buffer = ctx.getTokenScores();
   * ctx.applyGrammar(buffer);
   *
   * const scores = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
   * const token = customSample(scores);
   *
   * // MUST call acceptToken to advance parser
   * ctx.acceptToken(token);
   *
   * // Now parser is ready for next token
   * ```
   */
  acceptToken(tokenId: number): void;

  /**
   * Reset grammar parser to initial state
   *
   * Call at start of each new generation with same grammar.
   * Parser returns to root state, ready to validate from beginning.
   *
   * Cost: <0.01ms
   *
   * @example
   * ```typescript
   * ctx.initGrammar(jsonGrammar);
   *
   * // First generation
   * while (!done) {
   *   const token = ctx.sample();
   *   // ... generate ...
   * }
   *
   * // Second generation - reuse same grammar
   * ctx.resetGrammar();
   * while (!done) {
   *   const token = ctx.sample();
   *   // ... generate ...
   * }
   * ```
   */
  resetGrammar(): void;

  /**
   * Free grammar resources
   *
   * Call when done with constrained generation.
   * Releases parser memory.
   *
   * Cost: <0.01ms
   *
   * @example
   * ```typescript
   * ctx.initGrammar(grammar);
   * // ... do constrained generation ...
   * ctx.freeGrammar();
   * ```
   */
  freeGrammar(): void;

  // ===== KV SEQUENCE OPERATIONS =====

  /**
   * Copy KV cache from one sequence to another
   *
   * Duplicates the KV cache state from source to destination sequence.
   * After copying, both sequences can continue independently.
   *
   * NOTE: llama.cpp only supports full sequence copies.
   * p0/p1 parameters are for API compatibility but partial
   * ranges will trigger an assertion failure in llama.cpp.
   *
   * Cost: ~1-5ms depending on sequence length
   *
   * @param srcSeqId Source sequence to copy from
   * @param dstSeqId Destination sequence to copy to
   * @param p0 Start position (default: 0, must be 0 for llama.cpp)
   * @param p1 End position (default: -1 = end, must be -1 for llama.cpp)
   * @example
   * ```typescript
   * // Decode initial prompt to seq 0
   * await ctx.decode(promptTokens, 0);
   *
   * // Copy seq 0 -> seq 1
   * ctx.kvSeqCopy(0, 1);
   *
   * // Now both sequences can continue independently
   * await ctx.decode([tokenA], position, 0);
   * await ctx.decode([tokenB], position, 1);
   * ```
   */
  kvSeqCopy(srcSeqId: number, dstSeqId: number, p0?: number, p1?: number): void;

  /**
   * Keep only specified sequence, remove all others
   *
   * NOTE: After kvSeqCopy, cells may be shared across sequences.
   * kvSeqKeep removes sequence associations but pos_max may
   * still return positions. For complete removal, use
   * kvCacheRemove(seqId, 0, -1) for each unwanted sequence.
   *
   * @param seqId Sequence ID to keep
   */
  kvSeqKeep(seqId: number): void;

  /**
   * Get max position in sequence
   *
   * Returns the highest position index in the specified sequence,
   * or -1 if the sequence is empty.
   *
   * Cost: <0.01ms (fast sync operation)
   *
   * @param seqId Sequence ID to query
   * @returns Max position index, or -1 if empty
   * @example
   * ```typescript
   * const pos = ctx.kvSeqPosMax(0);
   * if (pos === -1) {
   *   console.log('Sequence is empty');
   * } else {
   *   console.log(`Sequence has ${pos + 1} tokens`);
   * }
   * ```
   */
  kvSeqPosMax(seqId: number): number;

  // ===== HANDLE-BASED GRAMMAR =====

  /**
   * Create a new grammar sampler (returns handle)
   *
   * Creates an independent grammar sampler instance with its own state.
   *
   * Unlike initGrammar() which uses a single internal sampler, this returns
   * a handle that can be used with applySampler/acceptSamplerToken.
   * Multiple handles can coexist with independent parser states.
   *
   * Cost: ~0.1-1ms depending on grammar complexity
   *
   * @param grammarStr GBNF grammar string
   * @returns Handle to the created sampler
   * @example
   * ```typescript
   * const grammarHandle = ctx.createSampler(jsonGrammar);
   *
   * // Apply grammar constraints to logits
   * ctx.applySampler(grammarHandle, logitsBuffer);
   * ctx.acceptSamplerToken(grammarHandle, token);
   *
   * // Create independent copy with same grammar
   * const clonedHandle = ctx.cloneSampler(grammarHandle);
   *
   * // Cleanup when done
   * ctx.freeSamplerHandle(grammarHandle);
   * ctx.freeSamplerHandle(clonedHandle);
   * ```
   */
  createSampler(grammarStr: string): number;

  /**
   * Apply grammar constraints using handle-based sampler
   *
   * Masks invalid tokens with -Infinity based on parser state.
   * Modifies the logits buffer in-place.
   *
   * @param handle Sampler handle from createSampler()
   * @param logitsBuffer ArrayBuffer or TypedArray containing logits
   */
  applySampler(handle: number, logitsBuffer: ArrayBuffer | Float32Array): void;

  /**
   * Accept token to advance grammar parser state (handle-based)
   *
   * Must be called after sampling to advance the grammar parser.
   * This is the handle-based equivalent of acceptToken().
   *
   * @param handle Sampler handle from createSampler()
   * @param tokenId Token that was sampled
   */
  acceptSamplerToken(handle: number, tokenId: number): void;

  /**
   * Clone a grammar sampler
   *
   * Creates a copy of the sampler with identical parser state.
   * Both handles can then be used independently with their own state.
   *
   * @param handle Sampler handle to clone
   * @returns New handle to cloned sampler
   * @example
   * ```typescript
   * const original = ctx.createSampler(jsonGrammar);
   * ctx.acceptSamplerToken(original, openBrace);
   *
   * // Clone preserves parser state (already accepted openBrace)
   * const copy = ctx.cloneSampler(original);
   *
   * // Both can now continue independently
   * ctx.acceptSamplerToken(original, tokenA);
   * ctx.acceptSamplerToken(copy, tokenB);
   * ```
   */
  cloneSampler(handle: number): number;

  /**
   * Free a grammar sampler handle
   *
   * Releases memory for the specified sampler.
   * Handle becomes invalid after this call.
   *
   * @param handle Sampler handle to free
   */
  freeSamplerHandle(handle: number): void;

  // ===== METRICS API =====

  /**
   * Compute surprisal (negative log-likelihood) for a specific token.
   *
   * Measures how "surprising" the model finds the given token:
   * - Low surprisal: Model expected this token (high probability)
   * - High surprisal: Model didn't expect this token (low probability)
   *
   * @param pickedTokenId - Token ID to compute surprisal for
   * @param base - Logarithm base: "nats" (default) or "bits"
   * @returns Surprisal value in specified base
   *
   * @example
   * ```typescript
   * // After decode with logits=true
   * const token = ctx.sampleNextToken();
   * const surprisal = ctx.modelSurprisal(token, "bits");
   * console.log(`Model surprise: ${surprisal.toFixed(2)} bits`);
   * ```
   *
   * COST: O(1) - direct probability lookup from logits
   * REQUIRES: decode() called with logits=true
   */
  modelSurprisal(pickedTokenId: number, base?: "nats" | "bits"): number;

  /**
   * Compute entropy of the entire logits distribution.
   *
   * Measures model uncertainty:
   * - Low entropy: Model is confident (peaked distribution)
   * - High entropy: Model is uncertain (flat distribution)
   *
   * @param base - Logarithm base: "nats" (default), "bits", or "base10"
   * @returns Entropy value in specified base
   *
   * @example
   * ```typescript
   * // Check model confidence before sampling
   * const entropy = ctx.modelEntropy("bits");
   * if (entropy > 5.0) {
   *   console.log("Model is very uncertain - consider adjusting parameters");
   * }
   * ```
   *
   * COST: O(n_vocab) - must sum over all token probabilities
   * REQUIRES: decode() called with logits=true
   * ALGORITHM: Numerically stable log-sum-exp (metrics.hpp:73-81)
   */
  modelEntropy(base?: "nats" | "bits"): number;

  /**
   * Create a new perplexity tracker.
   *
   * @returns Integer handle to the tracker
   *
   * @example
   * ```typescript
   * const tracker = ctx.createPerplexityTracker();
   *
   * // Add surprisals during generation
   * for (let i = 0; i < tokens.length; i++) {
   *   const surprisal = ctx.modelSurprisal(tokens[i]);
   *   ctx.addSurprisal(tracker, surprisal);
   * }
   *
   * const ppl = ctx.getPerplexity(tracker);
   * console.log(`Sequence perplexity: ${ppl.toFixed(2)}`);
   *
   * ctx.freePerplexityTracker(tracker);
   * ```
   */
  createPerplexityTracker(): number;

  /**
   * Add a surprisal value to the rolling tracker.
   *
   * @param handle - Tracker handle from createPerplexityTracker()
   * @param surprisal - Surprisal value (from modelSurprisal or computed)
   *
   * @example
   * ```typescript
   * const surprisal = ctx.modelSurprisal(tokenId, "nats");
   * ctx.addSurprisal(tracker, surprisal);
   * ```
   *
   * COST: O(1) - numerically stable accumulation
   * THREAD-SAFETY: Not thread-safe (handle is session-local)
   */
  addSurprisal(handle: number, surprisal: number): void;

  /**
   * Get current perplexity value.
   *
   * @param handle - Tracker handle
   * @returns Perplexity = exp(average_surprisal_in_nats)
   *
   * @example
   * ```typescript
   * const ppl = ctx.getPerplexity(tracker);
   * console.log(`Current PPL: ${ppl.toFixed(2)}`);
   * ```
   *
   * FORMULA: PPL = exp(sum_surprisals / count)
   * RANGE: [1, ∞) where 1 = perfect prediction
   */
  getPerplexity(handle: number): number;

  /**
   * Clone a perplexity tracker (for fork/branch scenarios).
   *
   * @param sourceHandle - Handle to clone from
   * @returns New handle with same accumulated state
   *
   * @example
   * ```typescript
   * // Branch A and B start from same base perplexity
   * const baseTracker = ctx.createPerplexityTracker();
   * // ... accumulate base surprisals ...
   *
   * const branchA = ctx.clonePerplexityTracker(baseTracker);
   * const branchB = ctx.clonePerplexityTracker(baseTracker);
   *
   * // Branch A and B now track independently
   * ctx.addSurprisal(branchA, surprisalA);
   * ctx.addSurprisal(branchB, surprisalB);
   * ```
   */
  clonePerplexityTracker(sourceHandle: number): number;

  /**
   * Reset tracker to initial state (count=0, sum=0).
   *
   * @param handle - Tracker handle to reset
   *
   * @example
   * ```typescript
   * // Reuse tracker for multiple sequences
   * const tracker = ctx.createPerplexityTracker();
   *
   * for (const sequence of sequences) {
   *   ctx.resetPerplexityTracker(tracker);
   *   // ... process sequence ...
   *   const ppl = ctx.getPerplexity(tracker);
   * }
   * ```
   */
  resetPerplexityTracker(handle: number): void;

  /**
   * Get number of tokens tracked.
   *
   * @param handle - Tracker handle
   * @returns Number of surprisal values added
   */
  getPerplexityCount(handle: number): number;

  /**
   * Free perplexity tracker resources.
   *
   * @param handle - Tracker handle to free
   *
   * NOTE: Auto-freed in dispose() if not manually freed
   */
  freePerplexityTracker(handle: number): void;

  // ===== ATOMIC DECODE+CAPTURE =====

  /**
   * Decode tokens and capture logits atomically
   *
   * Mutex-protected operation that decodes tokens and immediately
   * copies logits to the destination buffer before releasing the lock.
   * Prevents race conditions in concurrent decode scenarios.
   *
   * Use this instead of separate decode() + getLogits() calls when
   * you need guaranteed consistency between decode and logits capture.
   *
   * @param tokens Token IDs to decode
   * @param position Start position in sequence
   * @param seqId Sequence ID
   * @param destBuffer Pre-allocated buffer to receive logits (vocabSize floats)
   * @example
   * ```typescript
   * // Pre-allocate buffer (reuse across calls)
   * const logitsBuffer = new Float32Array(ctx.vocabSize);
   *
   * // Atomic decode + capture
   * ctx.decodeAndCapture([token], position, seqId, logitsBuffer);
   *
   * // Safe to process logitsBuffer - it's an independent copy
   * const nextToken = sampleFromLogits(logitsBuffer);
   * ```
   */
  decodeAndCapture(
    tokens: number[],
    position: number,
    seqId: number,
    destBuffer: ArrayBuffer | Float32Array
  ): void;

  // ===== KV CACHE FILE PERSISTENCE =====

  /**
   * Write KV cache state + tokens to file
   *
   * Persists KV cache state for later restoration.
   * Useful for checkpointing long conversations.
   *
   * @param sequenceId Sequence ID to save
   * @param filepath Path to save file
   * @param tokens Tokens that were decoded into this sequence
   * @returns Promise resolving to bytes written
   */
  kvCacheWriteFile(
    sequenceId: number,
    filepath: string,
    tokens: number[]
  ): Promise<number>;

  /**
   * Read KV cache state + tokens from file
   *
   * Restores KV cache state from a previous kvCacheWriteFile call.
   *
   * @param sequenceId Sequence ID to restore to
   * @param filepath Path to saved file
   * @returns Promise resolving to tokens and bytes read
   */
  kvCacheReadFile(
    sequenceId: number,
    filepath: string
  ): Promise<{ tokens: number[]; bytesRead: number }>;

  // ===== HELPERS =====

  /**
   * Format messages using model's chat template
   *
   * Converts [{role, content}] → formatted prompt string.
   * Uses model's built-in template (ChatML, Llama, Mistral, etc.).
   *
   * Cost: ~1-5ms depending on message count
   *
   * @param messagesJson JSON string containing array of messages
   * @param templateOverride Optional custom template string
   * @returns Formatted prompt and stop tokens from template
   * @example
   * ```typescript
   * const result = await ctx.formatChat(JSON.stringify([
   *   { role: "system", content: "You are a helpful assistant" },
   *   { role: "user", content: "Hello!" }
   * ]));
   *
   * const tokens = await ctx.tokenize(result.prompt);
   * await ctx.decode(tokens, 0);
   * ```
   */
  formatChat(
    messagesJson: string,
    templateOverride?: string
  ): Promise<FormattedChatResult>;

  /**
   * Convert JSON schema to GBNF grammar
   *
   * Generates grammar string for constrained JSON generation.
   * Use with initGrammar() or sample({ grammar }).
   *
   * Cost: ~1-10ms depending on schema complexity
   *
   * @param schemaJson JSON schema string
   * @returns GBNF grammar string
   * @example
   * ```typescript
   * const schema = {
   *   type: "object",
   *   properties: {
   *     name: { type: "string" },
   *     age: { type: "number" }
   *   },
   *   required: ["name"]
   * };
   *
   * const grammar = ctx.jsonSchemaToGrammar(JSON.stringify(schema));
   * ctx.initGrammar(grammar);
   * ```
   */
  jsonSchemaToGrammar(schemaJson: string): string;

  /**
   * Validate chat template syntax
   *
   * Checks if template string is valid before using.
   *
   * Cost: ~0.1-1ms
   *
   * @param templateString Template string to validate
   * @returns True if template syntax is valid
   */
  validateChatTemplate(templateString: string): Promise<boolean>;

  // ===== EMBEDDING EXTRACTION =====

  /**
   * Encode tokens for embedding extraction
   *
   * Unlike decode(), this marks ALL tokens with logits=true which is
   * required for embedding extraction. Use with embeddings=true context.
   *
   * Workflow:
   * 1. Create context with { embeddings: true, poolingType: PoolingType.MEAN }
   * 2. Tokenize your text
   * 3. Clear KV cache (important between different texts!)
   * 4. Call encode() with tokens
   * 5. Call getEmbeddings() to get the vector
   *
   * Cost: ~5-50ms depending on text length and model
   *
   * @param tokens Token IDs from tokenize()
   * @example
   * ```typescript
   * // Create embedding context
   * const ctx = await createContext({
   *   modelPath: './nomic-embed.gguf',
   *   embeddings: true,
   *   poolingType: PoolingType.MEAN
   * });
   *
   * // Get embedding for text
   * const tokens = await ctx.tokenize("Hello world");
   * await ctx.kvCacheClear();  // Important between texts!
   * await ctx.encode(tokens);
   * const embedding = ctx.getEmbeddings();
   * ```
   */
  encode(tokens: number[]): Promise<void>;

  /**
   * Get embedding vector from context (after encode)
   *
   * Returns the embedding vector for the encoded text.
   * Call after encode() to extract embeddings.
   *
   * The vector dimension depends on the model (e.g., 768 for nomic-embed).
   * Use getEmbeddingDimension() to get the size.
   *
   * Cost: ~0.5ms (extraction from model state)
   *
   * @param normalize Apply L2 normalization (default: true for cosine similarity)
   * @returns Float32Array of embedding values
   * @example
   * ```typescript
   * await ctx.encode(tokens);
   *
   * // Get L2-normalized embedding (for cosine similarity)
   * const embedding = ctx.getEmbeddings();
   *
   * // Or raw embedding without normalization
   * const rawEmbedding = ctx.getEmbeddings(false);
   * ```
   */
  getEmbeddings(normalize?: boolean): Float32Array;

  /**
   * Get embedding dimension for model
   *
   * Returns the size of embedding vectors this model produces.
   * Common values: 768 (BERT-like), 1024, 2048, 4096.
   *
   * Cost: <0.01ms (fast model property lookup)
   *
   * @returns Embedding dimension
   * @example
   * ```typescript
   * const dim = ctx.getEmbeddingDimension();
   * console.log(`Model produces ${dim}-dimensional embeddings`);
   * ```
   */
  getEmbeddingDimension(): number;

  /**
   * Check if context has pooling enabled
   *
   * Returns true if context was created with embeddings=true and
   * a pooling type other than NONE.
   *
   * Cost: <0.01ms
   *
   * @returns True if pooling is enabled
   */
  hasPooling(): boolean;

  // ===== NATIVE REFERENCE IMPLEMENTATIONS (for testing) =====

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

  // ===== PROPERTIES =====

  /**
   * Model vocabulary size (number of possible tokens)
   *
   * This is the length of the scores buffer from getTokenScores().
   */
  readonly vocabSize: number;

  /**
   * Memory used by this context (bytes)
   *
   * Reports native memory for monitoring.
   * Includes model weights, KV cache, and context state.
   */
  readonly memorySize: number;

  // ===== LIFECYCLE =====

  /**
   * Free native resources
   *
   * Call when done with context to release model and KV cache memory.
   * Context becomes unusable after disposal.
   */
  dispose(): void;
}

/**
 * Create a new inference context
 *
 * @param options Context creation options
 * @returns Promise resolving to SessionContext instance
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
 *   const token = ctx.sample({ temperature: 0.7 });
 * } finally {
 *   ctx.dispose();
 * }
 * ```
 */
export function createContext(options: ContextOptions): Promise<SessionContext>;

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
 * Pattern: "Memoized Step-Scoped Views with Explicit Revocation"
 * - Memoization: If getLogits() called twice in same step, returns same buffer
 * - Revocation: On decode(), the previous buffer is detached
 *
 * @template T Return type of the callback
 * @param ctx The session context
 * @param fn Synchronous callback that uses logits - must not return a Promise
 * @returns The result from the callback
 * @throws Error if callback returns a Promise (async usage not allowed)
 *
 * @example Safe synchronous usage
 * ```typescript
 * // Compute entropy synchronously
 * const entropy = withLogits(ctx, (logits) => {
 *   let maxLogit = logits[0];
 *   for (let i = 1; i < logits.length; i++) {
 *     if (logits[i] > maxLogit) maxLogit = logits[i];
 *   }
 *
 *   let sumExp = 0;
 *   for (let i = 0; i < logits.length; i++) {
 *     sumExp += Math.exp(logits[i] - maxLogit);
 *   }
 *
 *   let entropy = 0;
 *   for (let i = 0; i < logits.length; i++) {
 *     const p = Math.exp(logits[i] - maxLogit) / sumExp;
 *     if (p > 0) entropy -= p * Math.log(p);
 *   }
 *   return entropy;
 * });
 *
 * // Now safe to decode (previous logits buffer is revoked)
 * await ctx.decode([nextToken], position++);
 * ```
 *
 * @example Error: async callback
 * ```typescript
 * // This will throw!
 * withLogits(ctx, async (logits) => {
 *   await something();  // NOT ALLOWED
 *   return logits[0];
 * });
 * ```
 */
export function withLogits<T>(
  ctx: SessionContext,
  fn: (logits: Float32Array) => T
): T;
