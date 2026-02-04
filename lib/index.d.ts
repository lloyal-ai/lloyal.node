/**
 * liblloyal-node TypeScript Definitions
 *
 * N-API bindings for liblloyal - Node.js native addon for llama.cpp inference
 */

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
 */
export type GpuVariant = 'default' | 'cuda' | 'vulkan';

/**
 * Options for binary loading
 *
 * Controls which native binary variant is loaded when creating a context.
 * Use this for explicit GPU variant selection with automatic fallback.
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
   *
   * @example
   * ```typescript
   * // Request CUDA with automatic fallback to CPU
   * const ctx = await createContext(
   *   { modelPath: './model.gguf' },
   *   { gpuVariant: 'cuda' }
   * );
   * ```
   */
  gpuVariant?: GpuVariant;
}

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
   * Batch size for token processing
   *
   * Controls how many tokens are processed per llama_decode call.
   * Higher values improve throughput for prompt prefill at the cost of memory.
   * Also sets llama_context_params.n_batch and n_ubatch at context creation.
   * Default: 512
   */
  nBatch?: number;

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
   * Useful for parallel decoding or conversation branching.
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
 *
 * Mirostat dynamically adjusts sampling to maintain target perplexity,
 * preventing both repetition and incoherence. Useful for long-form generation
 * where temperature alone produces inconsistent quality.
 *
 * Use Mirostat v2 (mode: 2) for most cases - it's more stable than v1.
 */
export interface MirostatParams {
  /** Mirostat mode (0 = disabled, 1 = v1, 2 = v2). Recommended: 2 */
  mode?: number;

  /** Target entropy (perplexity = exp(tau)). Default: 5.0. Lower = more focused */
  tau?: number;

  /** Learning rate for entropy adjustment. Default: 0.1. Higher = faster adaptation */
  eta?: number;
}

/**
 * DRY (Don't Repeat Yourself) sampling parameters
 *
 * Penalizes repetition of token sequences, more sophisticated than
 * simple repetition penalty. Useful for reducing loops and redundancy
 * in generated text.
 */
export interface DryParams {
  /** Penalty strength (0.0 = disabled, higher = stronger penalty) */
  multiplier?: number;

  /** Base penalty value (typically 1.75) */
  base?: number;

  /** Minimum sequence length to trigger penalty (typically 2) */
  allowedLength?: number;

  /** Number of recent tokens to scan for repetitions */
  penaltyLastN?: number;
}

/**
 * XTC (eXclude Top Choices) sampler parameters
 *
 * Excludes very high probability tokens to increase output diversity.
 * Useful when model is overly confident and produces repetitive text.
 */
export interface XtcParams {
  /** Probability of applying XTC (0.0 = disabled, 1.0 = always). Typical: 0.1 */
  probability?: number;

  /** Confidence threshold above which tokens are excluded. Typical: 0.1 */
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
 * Represents a loaded model with KV cache for maintaining conversation state.
 * Use createContext() to initialize, and dispose() when done to free memory.
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
   * @returns Float32Array of unnormalized logits (vocabSize elements)
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
   * // Constrained to valid JSON (handle-based API)
   * const grammarHandle = ctx.createSampler(grammar);
   * ctx.applySampler(grammarHandle, ctx.getLogits());
   * const token = ctx.sample({ temperature: 0.7 });
   * ctx.acceptSamplerToken(grammarHandle, token);
   * ```
   */
  sample(params?: SamplingParams): number;

  /**
   * Convert token ID to text piece
   *
   * Fast synchronous lookup in vocabulary table.
   * Call this on each generated token for streaming display.
   *
   * Optimized for per-token conversion during generation.
   * For batch conversion of many tokens, use detokenize() instead.
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
   * Optimized for batch conversion of many tokens.
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

  /**
   * Atomic clear+reseed operation
   *
   * Implements a KV cache compression strategy:
   * 1. Clear entire KV cache
   * 2. Re-decode original sinks (first N tokens from conversation start)
   * 3. Re-decode tail (last M recent tokens)
   *
   *
   * @param sinks - ORIGINAL first N tokens from conversation start (typically 4)
   * @param tail - Recent M tokens to preserve (typically 508-1020)
   * @returns Promise that resolves when reseed completes
   *
   * @example
   * ```typescript
   * const ORIGINAL_SINKS = allTokens.slice(0, 4);
   *
   * const tail = allTokens.slice(-508);  // Last 508 tokens
   * await ctx.clearAndReseed(ORIGINAL_SINKS, tail);
   *
   * const nextToken = ctx.greedySample();
   * await ctx.decode([nextToken], 512);
   * ```
   */
  clearAndReseed(sinks: number[], tail: number[]): Promise<void>;

  // ===== KV SEQUENCE OPERATIONS =====

  /**
   * Copy KV cache from one sequence to another
   *
   * Duplicates the KV cache state from source to destination sequence.
   * After copying, both sequences can continue independently.
   *
   * NOTE: Only full sequence copies are currently supported.
   * The p0/p1 parameters must use default values (0 and -1).
   *
   * Cost: ~1-5ms depending on sequence length
   *
   * @param srcSeqId Source sequence to copy from
   * @param dstSeqId Destination sequence to copy to
   * @param p0 Start position (must be 0, default: 0)
   * @param p1 End position (must be -1 for full copy, default: -1)
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
   * Removes all sequences except the one specified.
   * For complete cleanup of unwanted sequences, consider using
   * kvCacheRemove(seqId, 0, -1) on each sequence instead.
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
   * Returns a handle that can be used with applySampler/acceptSamplerToken.
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
   * Call after decode() to compute surprisal for any token based on
   * the current logits distribution, or pass captured logits for
   * offline computation (e.g., best-of-n scoring from prefill logits).
   *
   * @param pickedTokenId - Token ID to compute surprisal for
   * @param base - Logarithm base: "nats" (default) or "bits"
   * @param logits - Optional Float32Array of logits (uses current context logits if omitted)
   * @returns Surprisal value in specified base
   *
   * @example Current context logits (default)
   * ```typescript
   * await ctx.decode(tokens, position);
   * const token = ctx.sample();
   * const surprisal = ctx.modelSurprisal(token, "bits");
   * console.log(`Model surprise: ${surprisal.toFixed(2)} bits`);
   * ```
   *
   * @example Captured/arbitrary logits (for best-of-n, verification, etc.)
   * ```typescript
   * // Capture logits after prefill
   * const capturedLogits = new Float32Array(ctx.getLogits());
   *
   * // Later: compute surprisal from captured logits
   * const surprisal = ctx.modelSurprisal(token, "nats", capturedLogits);
   * ```
   *
   * COST: O(n_vocab) - softmax normalization required
   */
  modelSurprisal(pickedTokenId: number, base?: 'nats' | 'bits', logits?: Float32Array): number;

  /**
   * Compute entropy of the entire logits distribution.
   *
   * Measures model uncertainty:
   * - Low entropy: Model is confident (peaked distribution)
   * - High entropy: Model is uncertain (flat distribution)
   *
   * Call after decode() to analyze the current prediction distribution,
   * or pass captured logits for offline analysis.
   *
   * @param base - Logarithm base: "nats" (default), "bits", or "base10"
   * @param logits - Optional Float32Array of logits (uses current context logits if omitted)
   * @returns Entropy value in specified base
   *
   * @example Current context logits (default)
   * ```typescript
   * await ctx.decode(tokens, position);
   * const entropy = ctx.modelEntropy("bits");
   * if (entropy > 5.0) {
   *   console.log("Model is very uncertain - consider adjusting parameters");
   * }
   * ```
   *
   * @example Captured/arbitrary logits
   * ```typescript
   * const capturedLogits = new Float32Array(ctx.getLogits());
   * const entropy = ctx.modelEntropy("nats", capturedLogits);
   * ```
   *
   * COST: O(n_vocab) - must sum over all token probabilities
   */
  modelEntropy(base?: 'nats' | 'bits', logits?: Float32Array): number;

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
   * Performs decode and logits capture as a single atomic operation,
   * ensuring the captured logits correspond exactly to the decoded tokens.
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
   * Use with createSampler() for grammar-constrained generation.
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
   * const handle = ctx.createSampler(grammar);
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

  // ===== NATIVE REFERENCE IMPLEMENTATIONS =====

  /**
   * Sample greedily from current logits
   *
   * Selects token with highest logit value (deterministic).
   * Equivalent to sample() with temperature=0.
   *
   * @returns Token ID with highest probability
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

  // ===== BRANCH API (internal, wrapped by Branch class) =====

  /** @internal Create a new branch for parallel generation */
  _branchCreate(seqId: number, position: number, params?: SamplingParams, nBatch?: number): number;

  /** @internal Fork a branch to a new sequence */
  _branchFork(handle: number, newSeqId: number): number;

  /** @internal Capture logits into branch's snapshot */
  _branchCaptureLogits(handle: number): void;

  /** @internal Decode a single token and capture logits */
  _branchDecodeAndCaptureOne(handle: number, token: number): void;

  /** @internal Decode multiple tokens in n_batch-sized chunks and capture logits */
  _branchDecodeAndCaptureBatch(handle: number, tokens: number[]): void;

  /** @internal Sample next token from branch's logits snapshot */
  _branchSample(handle: number): number;

  /** @internal Accept token (update sampler state for penalties) */
  _branchAccept(handle: number, token: number): void;

  /** @internal Get branch's sequence ID */
  _branchGetSeqId(handle: number): number;

  /** @internal Get branch's current position */
  _branchGetPosition(handle: number): number;

  /** @internal Get branch's perplexity */
  _branchGetPerplexity(handle: number): number;

  /** @internal Prune branch (remove KV cache entries and free handle) */
  _branchPrune(handle: number): void;

  /** @internal Destroy branch (free handle without removing KV cache) */
  _branchDestroy(handle: number): void;

  /** @internal Reseed branch sampler PRNG for diversity after fork */
  _branchSamplerChainReseed(handle: number, seed: number): void;
}

/**
 * Create a new inference context
 *
 * Loads the appropriate native binary (with automatic GPU fallback) and
 * creates an inference context for the specified model.
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
 *   await ctx.decode(tokens, 0);
 *   const token = ctx.sample({ temperature: 0.7 });
 * } finally {
 *   ctx.dispose();
 * }
 * ```
 *
 * @example With GPU variant selection
 * ```typescript
 * // Request CUDA - falls back to CPU if unavailable
 * const ctx = await createContext(
 *   { modelPath: './model.gguf', nCtx: 4096 },
 *   { gpuVariant: 'cuda' }
 * );
 * ```
 *
 * @example Using environment variable
 * ```typescript
 * // Set LLOYAL_GPU=cuda before running
 * // createContext will automatically use CUDA if available
 * const ctx = await createContext({ modelPath: './model.gguf' });
 * ```
 */
export function createContext(
  options: ContextOptions,
  loadOptions?: LoadOptions
): Promise<SessionContext>;

/**
 * Load native binary for a specific GPU variant
 *
 * Loads the appropriate platform-specific binary with automatic fallback:
 * 1. Try requested GPU variant (if specified)
 * 2. Fall back to default (CPU) platform package
 * 3. Fall back to local build (development: build/Release/lloyal.node)
 *
 * Use this for advanced scenarios where you need direct binary access
 * or want to check variant availability before creating a context.
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
 */
export function loadBinary(variant?: GpuVariant): {
  createContext(options: ContextOptions): Promise<SessionContext>;
};

/**
 * Safe logits access with automatic lifetime management
 *
 * Ensures logits are only accessed synchronously within the callback.
 * The callback MUST NOT:
 * - Store the logits reference
 * - Return a Promise (will throw)
 * - Call decode() (would invalidate logits)
 *
 * This prevents common bugs where logits become invalid due to
 * async operations between access and usage.
 *
 * How it works:
 * - Memoization: Multiple getLogits() calls in same step return same buffer
 * - Revocation: Next decode() invalidates previous buffer
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

/**
 * Result from Branch.produce()
 */
export interface Produced {
  /** Sampled token ID */
  token: number;
  /** Text representation of the token */
  text: string;
  /** Whether this is a stop token (EOS) */
  isStop: boolean;
}

/**
 * Forkable inference handle for covalent generation
 *
 * A Branch owns everything needed for independent generation: a KV cache
 * sequence, sampler chain, logits snapshot, and perplexity tracker.
 *
 * Forking is cheap — the KV prefix is shared in memory (metadata-only operation under unified KV —
 * no KV tensor buffers are copied), so sibling branches read from the same physical KV entries.
 * Only tokens decoded after the fork point are exclusive to each branch.
 *
 * Branches form trees, not just flat lists. Fork from root for best-of-N,
 * fork from children for MCTS/beam search, fork from a draft for speculative
 * decoding.
 *
 * The produce/commit protocol separates sampling from state advancement:
 * produce() samples without writing to KV, letting you inspect the result
 * before deciding to commit().
 *
 * @example Best-of-N with perplexity selection
 * ```typescript
 * const root = Branch.create(ctx, 0, tokens.length, { temperature: 0.8 });
 * root.captureLogits();
 *
 * const candidates = [1, 2, 3, 4, 5].map((seqId, i) => {
 *   const branch = root.fork(seqId);
 *   branch.reseedSampler(1000 + i);
 *   return branch;
 * });
 *
 * for (let t = 0; t < 50; t++) {
 *   for (const branch of candidates) {
 *     const { token, isStop } = branch.produce();
 *     if (isStop) continue;
 *     branch.commit(token);
 *   }
 * }
 *
 * const best = candidates.reduce((a, b) => a.perplexity < b.perplexity ? a : b);
 * for (const c of candidates) { if (c !== best) c.prune(); }
 * ```
 */
export class Branch {
  /**
   * Create a root branch at the given position
   *
   * The branch takes ownership of the sequence and creates its own sampler
   * chain from the provided params. Call captureLogits() after prefill to
   * freeze the logit distribution before forking.
   *
   * @param ctx SessionContext to create branch on
   * @param seqId Sequence ID for this branch
   * @param position Starting position (typically prompt token count)
   * @param params Sampling parameters (temperature, topP, etc.)
   * @param nBatch Per-branch batch size override (defaults to context nBatch)
   */
  static create(
    ctx: SessionContext,
    seqId: number,
    position: number,
    params?: SamplingParams,
    nBatch?: number
  ): Branch;

  /**
   * Fork this branch to a new sequence
   *
   * The child shares the parent's KV prefix in memory (metadata-only under unified KV, no KV buffer copy).
   * Logits, sampler state, and perplexity tracker are cloned so the child
   * can diverge independently. Fork from any branch — root or intermediate —
   * to build arbitrarily deep trees.
   *
   * @param newSeqId Sequence ID for the forked branch
   */
  fork(newSeqId: number): Branch;

  /** Freeze the current logit distribution into this branch. Essential before fork(). */
  captureLogits(): void;

  /** Decode a single token, write to KV, and capture resulting logits */
  decodeAndCaptureOne(token: number): void;

  /**
   * Bulk-decode tokens into the branch's KV cache and capture logits.
   *
   * `tokens.length` is the total count to process; the branch's `nBatch`
   * (set at `Branch.create`) controls how many are sent per `llama_decode`
   * call. E.g. 500 tokens with `nBatch=64` → 8 calls (7×64 + 1×52).
   *
   * Advances `position` by `tokens.length`. Stores final logits into the
   * branch's internal snapshot — the next `produce()`/`sample()` reads
   * from it.
   *
   * Does NOT accept tokens into the repeat-penalty window — for external
   * tokens (user input between turns), not model-generated tokens.
   * For model output, use `commit()` which does accept + decode.
   *
   * Branch-level equivalent of `ctx.decode()`.
   *
   * @param tokens - Token IDs to decode
   */
  prefill(tokens: number[]): void;

  /** Sample next token from branch's frozen logits snapshot */
  sample(): number;

  /** Accept token for repeat-penalty tracking */
  accept(token: number): void;

  /** Discard branch — remove its divergent KV entries and free the handle (use for losers) */
  prune(): void;

  /** Release handle but keep KV entries intact (use for winners, continue with raw ops) */
  destroy(): void;

  /**
   * Reseed the sampler's PRNG for diversity after fork()
   *
   * CRITICAL for parallel generation: Without reseeding, all forked branches
   * produce identical outputs because they share the same PRNG state.
   *
   * Only affects stochastic samplers (temperature > 0). Greedy samplers are unchanged.
   *
   * @param seed - New seed for the PRNG
   */
  reseedSampler(seed: number): void;

  /** Sample next token without advancing state. Inspect before committing. */
  produce(): Produced;

  /** Accept and advance — write token to KV and update branch state. */
  commit(token: number): void;

  /** Branch's sequence ID */
  readonly seqId: number;

  /** Branch's current position */
  readonly position: number;

  /** Branch's perplexity */
  readonly perplexity: number;

  /** Internal handle (for debugging) */
  readonly handle: number;

  /** Whether this branch has been disposed */
  readonly disposed: boolean;
}
