/**
 * liblloyal-node TypeScript Definitions
 *
 * N-API bindings for liblloyal - Node.js native addon for llama.cpp inference
 *
 * @categoryDescription Core
 * Entry points, context lifecycle, and the main inference interface.
 *
 * @categoryDescription Sampling
 * Sampler chain configuration — temperature, penalties, nucleus sampling, and advanced filters.
 *
 * @categoryDescription Chat
 * Chat template formatting, output parsing, tool calls, and reasoning extraction.
 *
 * @categoryDescription Branching
 * Parallel and tree-structured generation with batched GPU dispatch.
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
 *
 * @category Core
 */
export type GpuVariant = 'default' | 'cuda' | 'vulkan';

/**
 * Options for binary loading
 *
 * Controls which native binary variant is loaded when creating a context.
 * Use this for explicit GPU variant selection with automatic fallback.
 *
 * @category Core
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
 *
 * @category Core
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
 * Configuration for context creation
 *
 * Controls the resource envelope for inference: context window size (`nCtx`),
 * batch throughput (`nBatch`), compute parallelism (`nThreads`), and
 * multi-sequence capacity (`nSeqMax`). These map directly to
 * `llama_context_params` and are fixed for the context's lifetime.
 *
 * Key tradeoffs:
 * - **nCtx**: Larger = longer conversations, but linear KV memory growth.
 * - **nBatch**: Larger = faster prompt prefill (more tokens per GPU dispatch),
 *   but higher peak memory. Also sets the bin-packing capacity for
 *   {@link BranchStore.prefill}.
 * - **nSeqMax**: Set ≥ your max concurrent branch count + 1 (root sequence).
 *   Each sequence shares the same KV cache memory pool — cost is metadata only
 *   under unified KV, not a per-sequence memory multiplier.
 *
 * @category Core
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
 * Chat format detected by the template engine
 *
 * Identifies how the model formats tool calls, reasoning blocks, and content.
 * Returned by {@link SessionContext.formatChat | formatChat()} in
 * {@link FormattedChatResult.format} and consumed by
 * {@link SessionContext.parseChatOutput | parseChatOutput()}.
 *
 * You generally don't need to inspect these values directly --
 * just pass them through from the formatChat result to parseChatOutput.
 *
 * Only commonly-used values are listed. The full set matches llama.cpp's
 * `common_chat_format` enum (30+ formats).
 *
 * @category Chat
 */
export enum ChatFormat {
  /** Plain content, no special formatting */
  CONTENT_ONLY = 0,
  /** Generic tool call format */
  GENERIC = 1,
}

/**
 * Reasoning/thinking block format
 *
 * Controls how `<think>` blocks are handled during formatting and parsing.
 *
 * @see {@link FormatChatOptions.reasoningFormat} for input-side usage
 * @see {@link ParseChatOutputOptions.reasoningFormat} for output-side usage
 *
 * @category Chat
 */
export enum ReasoningFormat {
  /** No reasoning extraction (default) */
  NONE = 0,
  /** Auto-detect reasoning format from model template */
  AUTO = 1,
  /** DeepSeek legacy format (`<think>...</think>` in content) */
  DEEPSEEK_LEGACY = 2,
  /** DeepSeek format (structured reasoning extraction) */
  DEEPSEEK = 3,
}

/**
 * Grammar trigger type
 *
 * Determines how lazy grammar activation is triggered during generation.
 *
 * @see {@link GrammarTrigger}
 * @see {@link FormattedChatResult.grammarTriggers}
 *
 * @category Chat
 */
export enum GrammarTriggerType {
  /** Trigger on a specific token ID */
  TOKEN = 0,
  /** Trigger on a word boundary match */
  WORD = 1,
  /** Trigger on a regex pattern match */
  PATTERN = 2,
  /** Trigger on a full-string regex pattern match */
  PATTERN_FULL = 3,
}

/**
 * Options for chat template formatting
 *
 * Controls format-awareness fields passed to the chat template engine.
 * All fields are optional -- sensible defaults are used when omitted.
 *
 * @example With tools and reasoning
 * ```typescript
 * const result = await ctx.formatChat(messagesJson, {
 *   tools: JSON.stringify(tools),
 *   toolChoice: 'auto',
 *   reasoningFormat: 'auto',
 * });
 * ```
 *
 * @category Chat
 */
export interface FormatChatOptions {
  /** Custom Jinja2 template override (bypasses model's built-in template) */
  templateOverride?: string;

  /**
   * JSON array of OpenAI-format tool definitions
   *
   * @example
   * ```typescript
   * const tools = [{ type: 'function', function: {
   *   name: 'get_weather',
   *   description: 'Get current weather',
   *   parameters: { type: 'object', properties: { location: { type: 'string' } } }
   * }}];
   * options.tools = JSON.stringify(tools);
   * ```
   */
  tools?: string;

  /** Tool choice strategy (default: "auto") */
  toolChoice?: 'auto' | 'required' | 'none';

  /** Allow parallel tool calls (default: false) */
  parallelToolCalls?: boolean;

  /**
   * Reasoning format (default: "none")
   *
   * Controls `<think>` block handling in the template.
   * Use "auto" to let the model's template decide.
   */
  reasoningFormat?: 'none' | 'auto' | 'deepseek' | 'deepseek_legacy';

  /** Enable `<think>` blocks (default: true). Pairs with reasoningFormat. */
  enableThinking?: boolean;

  /**
   * JSON schema for constrained output. Converted to GBNF grammar internally.
   * Mutually exclusive with `grammar`.
   *
   * @see {@link SessionContext.jsonSchemaToGrammar}
   */
  jsonSchema?: string;

  /**
   * Explicit GBNF grammar string for constrained generation.
   * Mutually exclusive with `jsonSchema`.
   */
  grammar?: string;

  /**
   * Append assistant prompt prefix (default: true).
   * Set false when formatting partial conversations or for
   * non-generation use cases like template validation.
   */
  addGenerationPrompt?: boolean;
}

/**
 * Grammar trigger from format-aware chat template
 *
 * Defines conditions for lazy grammar activation. When `grammarLazy` is true
 * in {@link FormattedChatResult}, generation runs unconstrained until one of
 * these triggers fires, at which point the grammar is activated.
 *
 * @category Chat
 */
export interface GrammarTrigger {
  /** Trigger type */
  type: GrammarTriggerType;
  /** Trigger value (token text, word, or regex pattern depending on type) */
  value: string;
  /** Token ID (for TOKEN-type triggers, -1 when not applicable) */
  token: number;
}

/**
 * Result from chat template formatting
 *
 * Includes format-awareness fields for proper output parsing.
 * Pass `format` and `reasoningFormat` directly to
 * {@link SessionContext.parseChatOutput | parseChatOutput()} to decode
 * the model's response.
 *
 * @example Roundtrip: format -> generate -> parse
 * ```typescript
 * const fmt = await ctx.formatChat(messagesJson, { tools: toolsJson });
 * // ... generate tokens using fmt.prompt and fmt.grammar ...
 * const parsed = ctx.parseChatOutput(output, fmt.format, {
 *   reasoningFormat: fmt.reasoningFormat,
 *   thinkingForcedOpen: fmt.thinkingForcedOpen,
 *   parser: fmt.parser,
 * });
 * ```
 *
 * @see {@link SessionContext.parseChatOutput}
 *
 * @category Chat
 */
export interface FormattedChatResult {
  /** Formatted prompt string ready for tokenization */
  prompt: string;
  /** Additional stop strings from the template */
  stopTokens: string[];

  /**
   * Detected chat format (pass to parseChatOutput)
   * @see {@link SessionContext.parseChatOutput}
   */
  format: ChatFormat;

  /** Grammar string for constrained generation (empty if no tools/schema) */
  grammar: string;
  /** Whether grammar should be applied lazily (only after triggers fire) */
  grammarLazy: boolean;
  /** Whether the thinking tag was forced open by the template */
  thinkingForcedOpen: boolean;

  /**
   * Reasoning format (pass to parseChatOutput options)
   * @see {@link ParseChatOutputOptions.reasoningFormat}
   */
  reasoningFormat: ReasoningFormat;

  /** PEG parser definition for PEG format models (pass to parseChatOutput options) */
  parser: string;
  /** Grammar trigger conditions for lazy grammar activation */
  grammarTriggers: GrammarTrigger[];
  /** Token strings preserved from grammar masking */
  preservedTokens: string[];
}

/**
 * Options for parsing chat output
 *
 * All fields are optional. For correct parsing, pass through the corresponding
 * fields from {@link FormattedChatResult}.
 *
 * @see {@link FormattedChatResult}
 *
 * @category Chat
 */
export interface ParseChatOutputOptions {
  /**
   * Reasoning format (from {@link FormattedChatResult.reasoningFormat})
   */
  reasoningFormat?: ReasoningFormat;

  /**
   * True if output is incomplete (streaming).
   * When true, the parser tolerates unterminated tool calls and open
   * thinking blocks, returning partial content as-is rather than
   * treating them as parse errors.
   */
  isPartial?: boolean;

  /** Whether thinking tag was forced open (from {@link FormattedChatResult.thinkingForcedOpen}) */
  thinkingForcedOpen?: boolean;

  /** PEG parser definition for PEG format models (from {@link FormattedChatResult.parser}) */
  parser?: string;
}

/**
 * A tool call extracted from model output
 *
 * @example
 * ```typescript
 * for (const tc of result.toolCalls) {
 *   const args = JSON.parse(tc.arguments);
 *   await executeTool(tc.name, args);
 * }
 * ```
 *
 * @category Chat
 */
export interface ParsedToolCall {
  /** Tool/function name */
  name: string;
  /** JSON string of arguments */
  arguments: string;
  /** Tool call ID (may be empty depending on model format) */
  id: string;
}

/**
 * Result from parsing chat output
 *
 * @example
 * ```typescript
 * const result = ctx.parseChatOutput(output, fmt.format);
 * if (result.toolCalls.length > 0) {
 *   for (const tc of result.toolCalls) {
 *     const args = JSON.parse(tc.arguments);
 *     await executeTool(tc.name, args);
 *   }
 * } else {
 *   console.log(result.content);
 * }
 * ```
 *
 * @category Chat
 */
export interface ParseChatOutputResult {
  /** Main response text */
  content: string;
  /**
   * Extracted thinking/reasoning content (empty string if none).
   * For thinking models (e.g. Qwen3), this contains the text inside
   * `<think>...</think>` blocks. Store as `reasoning_content` in your
   * messages array so formatChat() can reconstruct the template correctly
   * on subsequent turns.
   */
  reasoningContent: string;
  /** Extracted tool calls (empty array if none) */
  toolCalls: ParsedToolCall[];
}

/**
 * Penalty parameters for repetition control
 *
 * @category Sampling
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
 *
 * @category Sampling
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
 *
 * @category Sampling
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
 *
 * @category Sampling
 */
export interface XtcParams {
  /** Probability of applying XTC (0.0 = disabled, 1.0 = always). Typical: 0.1 */
  probability?: number;

  /** Confidence threshold above which tokens are excluded. Typical: 0.1 */
  threshold?: number;
}

/**
 * Advanced sampling parameters
 *
 * @category Sampling
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
 * Configures the sampler chain — a pipeline of composable filters and
 * transforms applied to raw logits before token selection. The chain is
 * built once at branch creation and persists across decode steps
 * (penalty state accumulates, PRNG advances).
 *
 * **Chain order**: penalties → top_k → typical_p → top_p → min_p →
 * temperature → dist (stochastic) or greedy (temperature ≤ 0).
 *
 * For tree search, each {@link Branch} owns an independent clone of the
 * chain. `reseedSampler()` replaces the terminal dist sampler's PRNG seed
 * so forked branches diverge. Greedy chains (temperature ≤ 0) are
 * deterministic and unaffected by reseeding.
 *
 * Common presets:
 * - Factual/Precise: `{ temperature: 0.1 }`
 * - Balanced: `{ temperature: 0.7 }`
 * - Creative: `{ temperature: 1.0 }`
 * - Deterministic greedy: `{ temperature: 0, topK: 0, topP: 1.0, minP: 0 }`
 *
 * @category Sampling
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
 * Inference context — the runtime surface for a loaded model
 *
 * A SessionContext owns a llama_context (KV cache + compute graph) bound to a
 * shared model. It provides tokenization, logit access, KV cache management,
 * chat template formatting, and embedding extraction.
 *
 * **All generation flows through {@link Branch}.** Create a branch at position 0,
 * prefill prompt tokens, then use the produce/commit loop or async iterator:
 *
 * ```typescript
 * const branch = Branch.create(ctx, 0, { temperature: 0.7 });
 * await branch.prefill(promptTokens);
 * for await (const { token, text } of branch) {
 *   process.stdout.write(text);
 * }
 * ```
 *
 * For tree-structured generation (best-of-N, beam search, speculative
 * decoding), use {@link Branch.fork} and {@link BranchStore} — they manage
 * per-branch KV sequences, sampler chains, and logits snapshots with O(1)
 * GPU dispatches via batched decode.
 *
 * **Logits lifetime**: `getLogits()` returns a zero-copy Float32Array wrapping
 * llama.cpp's internal buffer. It is invalidated (ArrayBuffer detached) on
 * the next `encode()` or `dispose()`. Use {@link withLogits} for safe scoped
 * access. For branch-level logits, use {@link Branch.getLogits} instead.
 *
 * **KV cache**: Supports multi-sequence operation (`nSeqMax > 1`), per-sequence
 * copy/clear/eviction, file-based persistence, and context compression via
 * `clearAndReseed()`.
 *
 * **Chat templates**: `formatChat()` and `parseChatOutput()` handle the full
 * round-trip of chat formatting, including tool calls, reasoning blocks, and
 * grammar-constrained generation — using the model's native Jinja template.
 *
 * Use {@link createContext} to initialize, and `dispose()` when done to free
 * GPU/CPU memory.
 *
 * @category Core
 */
export interface SessionContext {

  /**
   * Get logits (zero-copy view into model memory)
   *
   * Returns unnormalized scores for every possible next token.
   * Higher score = model thinks this token is more likely.
   * The returned Float32Array wraps llama.cpp's internal buffer directly
   * (zero-copy).
   *
   * LIFETIME CONSTRAINTS:
   * - Valid ONLY until the next encode() or dispose() call
   * - The ArrayBuffer is detached on invalidation — accessing a stale
   *   buffer throws a TypeError
   * - DO NOT retain references across async boundaries
   *
   * For a safe scoped access pattern, use {@link withLogits} instead.
   *
   * NOTE: This is the context-level logits view. For branch-level logits,
   * use {@link Branch.getLogits} which returns an independent copy of the
   * branch's snapshot (safe to hold, not invalidated by decode).
   *
   * Cost: ~0.5ms (zero-copy pointer, no data copied)
   *
   * @returns Float32Array of unnormalized logits (vocabSize elements)
   */
  getLogits(): Float32Array;

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
   * @param token Token ID
   * @returns Text string for this token
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
   */
  isStopToken(token: number): boolean;

  /**
   * Get the model's end-of-generation token ID
   *
   * Returns the EOT token (e.g. <|im_end|> for ChatML), falling back
   * to EOS (e.g. </s>) for Zephyr-style models. This is the inverse
   * of isStopToken() — "what IS the stop token?" vs "is this a stop token?"
   *
   * Use case: warm multi-turn continuation prepends this token to close
   * the previous assistant turn before injecting new user content.
   *
   * @returns Token ID (integer)
   * @throws If model has neither EOT nor EOS token
   */
  getEogToken(): number;

  /**
   * Get the model's turn separator token IDs
   *
   * Returns the tokens that close an assistant turn and transition to the
   * next message, as determined by the model's chat template. Computed once
   * per model, cached.
   *
   * For ChatML templates: [im_end_id, newline_id] (e.g., [2, 198])
   * For Llama 3 templates: [eot_id] (e.g., [128009])
   *
   * Use case: warm multi-turn prefill to achieve exact parity with cold path.
   *
   * @returns Array of token IDs (cached after first call)
   *
   * @example
   * ```typescript
   * const separator = ctx.getTurnSeparator();
   * console.log(separator.map(t => ctx.tokenToText(t)).join(''));  // "<|im_end|>\n"
   *
   * // Warm prefill with exact cold/warm parity
   * const deltaTokens = await ctx.tokenize(deltaPrompt, false);
   * await branch.prefill([...separator, ...deltaTokens]);
   * ```
   */
  getTurnSeparator(): number[];

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
   * @param addSpecial Whether to add special tokens (BOS/EOS). Defaults to
   *   model metadata setting (typically true). Pass false for mid-sequence
   *   tokenization (e.g., warm multi-turn continuation deltas).
   * @returns Array of token IDs
   * @example
   * ```typescript
   * // Full sequence (default — includes BOS)
   * const tokens = await ctx.tokenize("Hello world");
   *
   * // Mid-sequence delta (no BOS)
   * const delta = await ctx.tokenize("continuation text", false);
   * ```
   */
  tokenize(text: string, addSpecial?: boolean): Promise<number[]>;

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
   * Get max position in the KV cache for a sequence
   *
   * Returns the highest position index in the specified sequence,
   * or -1 if the sequence is empty. This is the same value as
   * {@link kvSeqPosMax}. To get the token count, add 1.
   *
   * Think of this as: "How much has the model read so far?"
   *
   * Cost: <0.01ms (fast sync operation - safe to call frequently)
   *
   * @param sequenceId Sequence ID (defaults to 0 for single conversation)
   * @returns Highest position index, or -1 if empty
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
   */
  kvCacheClear(): Promise<void>;

  /**
   * Blink KV — cache-local reconstruction for bounded-memory streaming
   *
   * Implements the [Blink KV](https://github.com/lloyal-ai/blink-kv/blob/main/blink_kv.pdf)
   * protocol (Naqvi, 2026): when the KV cache fills, clear it entirely and
   * re-decode retained tokens at contiguous positions `[0, 1, ..., N-1]`.
   * This achieves cache-local position IDs — the operative requirement for
   * stable bounded-memory streaming — without backend-specific knowledge of
   * key storage format. Works on post-RoPE engines (where StreamingLLM's
   * pos-shift is unavailable) and any backend exposing `clear()` + `decode()`.
   *
   * **Why not naive eviction?** Selective eviction (`kvCacheRemove`) preserves
   * original position IDs, which grow without bound. Across 5 architectures,
   * naive eviction produces PPL spanning 3 orders of magnitude — ranging from
   * 1.15× baseline (Llama, lucky config) to 198× (Phi, sinks present).
   * Under Blink KV reconstruction, all 5 converge to 3–16% of baseline.
   *
   * **Sinks are optional.** Under reconstruction, the 0+N (sinkless) config
   * matches 4+N (with sinks) within <2% across all tested architectures.
   * Pass an empty sinks array if you don't need them.
   *
   * **Algorithm:**
   * 1. Clear entire KV cache (zero fragmentation)
   * 2. Re-decode `sinks` at position 0 (optional attention anchors)
   * 3. Re-decode `tail` at position `sinks.length` (recent context)
   *
   * **Cost:** Re-decodes `sinks.length + tail.length` tokens. At per-boundary
   * trigger (reconstruct when cache reaches `nCtx`), amortized cost is
   * O(cacheSize / interval) decode ops per token — ~0.14 at typical settings.
   *
   * @param sinks First N tokens from conversation start (typically 4, or empty).
   *   Must be the same tokens every reseed — reusing different tokens degrades
   *   any attention-sink patterns the model may have learned for early positions.
   * @param tail Recent M tokens to preserve (typically 252–1020)
   * @returns Promise that resolves when reconstruction completes.
   *   Next decode continues at position `sinks.length + tail.length`.
   *
   * @example Per-boundary reconstruction
   * ```typescript
   * // Capture sinks once at conversation start
   * const SINKS = allTokens.slice(0, 4);
   *
   * // On cache fill: compress to 512 tokens (4 sinks + 508 tail)
   * if (position >= ctx.nCtx) {
   *   const tail = allTokens.slice(-508);
   *   await ctx.clearAndReseed(SINKS, tail);
   *   position = 512;  // sinks.length + tail.length
   * }
   * ```
   *
   * @example Sinkless reconstruction (equally effective)
   * ```typescript
   * const tail = allTokens.slice(-256);
   * await ctx.clearAndReseed([], tail);  // No sinks needed
   * position = 256;
   * ```
   *
   * @see [Blink KV paper](https://github.com/lloyal-ai/blink-kv/blob/main/blink_kv.pdf)
   */
  clearAndReseed(sinks: number[], tail: number[]): Promise<void>;

  // ===== KV SEQUENCE OPERATIONS =====

  /**
   * Fork a KV cache sequence — the primitive behind {@link Branch.fork}
   *
   * Copies all KV cache entries from `srcSeqId` to `dstSeqId`. Under
   * llama.cpp's unified KV cache, this is a **metadata-only operation** —
   * no key/value tensors are copied. Both sequences reference the same
   * physical KV entries for the shared prefix; only tokens decoded after
   * the fork point allocate new storage. This is what makes tree-structured
   * generation (best-of-N, beam search, speculative decoding) memory-efficient:
   * N branches sharing a 1000-token prefix cost ~1000 KV entries, not N×1000.
   *
   * The higher-level {@link Branch.fork} wraps this and additionally clones
   * the sampler chain, grammar state, logits snapshot, and perplexity tracker.
   * Use `kvSeqCopy` directly when you need raw sequence management without
   * the Branch abstraction.
   *
   * NOTE: Only full-sequence copies are supported. The p0/p1 parameters
   * must use default values (0 and -1).
   *
   * Cost: O(1) metadata — no tensor copy under unified KV
   *
   * @param srcSeqId Source sequence to copy from
   * @param dstSeqId Destination sequence to copy to
   * @param p0 Start position (must be 0, default: 0)
   * @param p1 End position (must be -1 for full copy, default: -1)
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

  // ===== METRICS API =====

  /**
   * Compute surprisal (negative log-likelihood) for a specific token.
   *
   * Measures how "surprising" the model finds the given token:
   * - Low surprisal: Model expected this token (high probability)
   * - High surprisal: Model didn't expect this token (low probability)
   *
   * Pass captured logits (e.g., from {@link Branch.getLogits}) for
   * offline computation, or omit to use the current context logits.
   *
   * @param pickedTokenId - Token ID to compute surprisal for
   * @param base - Logarithm base: "nats" (default) or "bits"
   * @param logits - Optional Float32Array of logits (uses current context logits if omitted)
   * @returns Surprisal value in specified base
   *
   * @example With branch logits
   * ```typescript
   * const { token } = await branch.produce();
   * const surprisal = ctx.modelSurprisal(token, "bits", branch.getLogits());
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
   * Pass captured logits (e.g., from {@link Branch.getLogits}) for
   * offline analysis, or omit to use the current context logits.
   *
   * @param base - Logarithm base: "nats" (default) or "bits"
   * @param logits - Optional Float32Array of logits (uses current context logits if omitted)
   * @returns Entropy value in specified base
   *
   * @example With branch logits
   * ```typescript
   * const entropy = ctx.modelEntropy("bits", branch.getLogits());
   * ```
   *
   * COST: O(n_vocab) - must sum over all token probabilities
   */
  modelEntropy(base?: 'nats' | 'bits', logits?: Float32Array): number;

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
   * Converts [{role, content}] → formatted prompt string with full format awareness.
   * Uses model's built-in template (ChatML, Llama, Mistral, etc.).
   *
   * The returned `format` and `reasoningFormat` fields should be passed to
   * `parseChatOutput()` after generation to correctly decode the response.
   *
   * Cost: ~1-5ms depending on message count
   *
   * @param messagesJson JSON string containing array of messages
   * @param options Formatting options (tools, reasoning, grammar, etc.)
   * @returns Formatted prompt with format-awareness metadata
   *
   * @see {@link parseChatOutput}
   *
   * @example Basic usage
   * ```typescript
   * const result = await ctx.formatChat(JSON.stringify([
   *   { role: "system", content: "You are a helpful assistant" },
   *   { role: "user", content: "Hello!" }
   * ]));
   *
   * const tokens = await ctx.tokenize(result.prompt);
   * const branch = Branch.create(ctx, 0, { temperature: 0.7 });
   * await branch.prefill(tokens);
   * ```
   */
  formatChat(
    messagesJson: string,
    options?: FormatChatOptions | string
  ): Promise<FormattedChatResult>;

  /**
   * Parse model output into structured content
   *
   * Extracts plain text, reasoning/thinking blocks, and tool calls from
   * raw model output. Uses the format detected by {@link formatChat} to apply
   * the correct parser for the model's output format.
   *
   * Cost: <0.1ms (synchronous string parsing, no I/O)
   *
   * @param output Raw model output text
   * @param format Chat format enum (from {@link FormattedChatResult.format})
   * @param options Optional parsing parameters
   * @returns Parsed content with tool calls and reasoning
   *
   * @see {@link formatChat}
   *
   * @example Basic parsing
   * ```typescript
   * const fmt = await ctx.formatChat(JSON.stringify(messages), { tools: toolsJson });
   * // ... generate tokens ...
   * const parsed = ctx.parseChatOutput(generatedText, fmt.format, {
   *   reasoningFormat: fmt.reasoningFormat,
   *   thinkingForcedOpen: fmt.thinkingForcedOpen,
   *   parser: fmt.parser
   * });
   * if (parsed.toolCalls.length > 0) {
   *   // Handle tool calls
   * }
   * ```
   *
   * @example Multi-turn warm continuation with reasoning models
   * ```typescript
   * // parseChatOutput separates <think>...</think> blocks into reasoningContent.
   * // This is REQUIRED for correct warm continuation on thinking models (e.g. Qwen3):
   * // if raw output containing <think> tags is stored as content, re-formatting
   * // the conversation produces different tokens, breaking cold/warm parity.
   *
   * const messages: Array<{role: string; content: string; reasoning_content?: string}> = [];
   * const sep = ctx.getTurnSeparator();
   * let branch: Branch | null = null;
   * let fmt: FormattedChatResult;
   *
   * async function handleTurn(userContent: string) {
   *   messages.push({ role: 'user', content: userContent });
   *
   *   if (!branch) {
   *     // Cold path: format full conversation, tokenize with BOS, prefill
   *     fmt = await ctx.formatChat(JSON.stringify(messages));
   *     const tokens = await ctx.tokenize(fmt.prompt);
   *     branch = Branch.create(ctx, 0, { temperature: 0.7 });
   *     await branch.prefill(tokens);
   *   } else {
   *     // Warm path: string-diff for delta tokens
   *     const { prompt: full } = await ctx.formatChat(JSON.stringify(messages));
   *     const { prompt: prefix } = await ctx.formatChat(
   *       JSON.stringify(messages.slice(0, -1)),
   *       { addGenerationPrompt: false }
   *     );
   *     const delta = await ctx.tokenize(full.substring(prefix.length), false);
   *     await branch.prefill([...sep, ...delta]);
   *   }
   *
   *   // Generate
   *   let rawOutput = '';
   *   while (true) {
   *     const { token, text, isStop } = await branch.produce();
   *     if (isStop) break;
   *     rawOutput += text;
   *     await branch.commit(token);
   *   }
   *
   *   // Parse output: separates reasoning from content
   *   const parsed = ctx.parseChatOutput(rawOutput, fmt.format, {
   *     reasoningFormat: fmt.reasoningFormat,
   *     thinkingForcedOpen: fmt.thinkingForcedOpen,
   *     parser: fmt.parser
   *   });
   *
   *   // Store parsed fields — formatChat reconstructs thinking blocks correctly
   *   messages.push({
   *     role: 'assistant',
   *     content: parsed.content,
   *     reasoning_content: parsed.reasoningContent || undefined
   *   });
   * }
   * ```
   */
  parseChatOutput(
    output: string,
    format: ChatFormat,
    options?: ParseChatOutputOptions
  ): ParseChatOutputResult;

  /**
   * Convert JSON schema to GBNF grammar
   *
   * Generates grammar string for constrained JSON generation.
   * Use with {@link Branch.create} grammar parameter for constrained generation.
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
   * const grammar = await ctx.jsonSchemaToGrammar(JSON.stringify(schema));
   * const branch = Branch.create(ctx, 0, params, undefined, grammar);
   * ```
   */
  jsonSchemaToGrammar(schemaJson: string): Promise<string>;

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

  // ===== PROPERTIES =====

  /**
   * Model vocabulary size (number of possible tokens)
   *
   * This is the length of the logits array from getLogits().
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
  _branchCreate(position: number, params?: SamplingParams, nBatch?: number): number;

  /** @internal Fork a branch to a new sequence */
  _branchFork(handle: number): number;

  /** @internal Decode multiple tokens in n_batch-sized chunks and capture logits */
  _branchPrefill(handle: number, tokens: number[]): Promise<void>;

  /** @internal Sample next token from branch's logits snapshot */
  _branchSample(handle: number): number;

  /** @internal Accept token (update sampler state for penalties) */
  _branchAccept(handle: number, token: number): void;

  /** @internal Get branch's current position */
  _branchGetPosition(handle: number): number;

  /** @internal Get branch's perplexity */
  _branchGetPerplexity(handle: number): number;

  /** @internal Get copy of branch's logits snapshot */
  _branchGetLogits(handle: number): Float32Array;

  /** @internal Prune branch (remove KV cache entries and free handle) — RESTRICT: throws if children */
  _branchPrune(handle: number): void;

  /** @internal Prune branch and all descendants — CASCADE */
  _branchPruneSubtree(handle: number): void;

  /** @internal Get parent branch handle (0 = INVALID_HANDLE if root) */
  _branchParent(handle: number): number;

  /** @internal Get child branch handles */
  _branchChildren(handle: number): number[];

  /** @internal Check if branch has no children */
  _branchIsLeaf(handle: number): boolean;

  /** @internal Check if branch holds a KV lease */
  _branchIsActive(handle: number): boolean;

  /** @internal Reseed branch sampler PRNG for diversity after fork */
  _branchSamplerChainReseed(handle: number, seed: number): void;

  /** @internal Set dynamic logit biases for a branch */
  _branchSteer(handle: number, biases: Array<{ token: number; bias: number }>): void;

  /** @internal Clear all dynamic logit biases from a branch */
  _branchClearSteer(handle: number): void;

  /** @internal Replace sampler chain with new parameters (memoized) */
  _branchSetSamplerParams(handle: number, params: SamplingParams): void;

  /** @internal Replace or remove grammar constraint */
  _branchSetGrammar(handle: number, grammarStr: string): void;

  // ===== STORE API (internal, wrapped by BranchStore) =====

  /** @internal Batched accept + decode_each + capture for N branches */
  _storeCommit(handles: number[], tokens: number[]): Promise<void>;

  /** @internal Batched decode_scatter + capture for N branches with variable token counts */
  _storePrefill(handles: number[], tokenArrays: number[][]): Promise<void>;

  /** @internal Retain winner branch, evict all others */
  _storeRetainOnly(handle: number): void;

  /** @internal Get number of available seq_id leases */
  _storeAvailable(): number;
}

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
export function createContext(
  options: ContextOptions,
  loadOptions?: LoadOptions
): Promise<SessionContext>;

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
 *
 * This prevents common bugs where logits become invalid due to
 * async operations between access and usage.
 *
 * @template T Return type of the callback
 * @param ctx The session context
 * @param fn Synchronous callback that uses logits - must not return a Promise
 * @returns The result from the callback
 * @throws Error if callback returns a Promise (async usage not allowed)
 *
 * @category Core
 */
export function withLogits<T>(
  ctx: SessionContext,
  fn: (logits: Float32Array) => T
): T;

/**
 * Result from Branch.produce()
 *
 * @category Branching
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
 * fork from children for tree search/beam search, fork from a draft for speculative
 * decoding.
 *
 * The produce/commit protocol separates sampling from state advancement:
 * produce() samples without writing to KV, letting you inspect the result
 * before deciding to commit().
 *
 * @example Best-of-N with perplexity selection
 * ```typescript
 * const root = Branch.create(ctx, tokens.length, { temperature: 0.8 });
 * await root.prefill(tokens);
 *
 * const results = [];
 * for (let i = 0; i < 5; i++) {
 *   const branch = await root.fork();
 *   branch.reseedSampler(1000 + i);
 *   const tokens = [];
 *   for await (const { token } of branch) tokens.push(token);
 *   results.push({ branch, tokens, ppl: branch.perplexity });
 * }
 *
 * const best = results.reduce((a, b) => a.ppl < b.ppl ? a : b);
 * for (const r of results) { if (r !== best) await r.branch.prune(); }
 * ```
 *
 * @category Branching
 */
export class Branch {
  /**
   * Create a root branch at the given position
   *
   * The branch takes ownership of the sequence and creates its own sampler
   * chain from the provided params. Call prefill() to decode prompt tokens
   * and capture the logit distribution before forking.
   *
   * @param ctx SessionContext to create branch on
   * @param position Starting position (typically prompt token count)
   * @param params Sampling parameters (temperature, topP, etc.)
   * @param nBatch Per-branch batch size override (defaults to context nBatch)
   * @param grammar GBNF grammar string for constrained generation. When provided,
   *   sample() returns only grammar-valid tokens. The grammar state is cloned on
   *   fork(), so sibling branches can diverge independently.
   */
  static create(
    ctx: SessionContext,
    position: number,
    params?: SamplingParams,
    nBatch?: number,
    grammar?: string
  ): Branch;

  /**
   * Fork this branch to a new sequence
   *
   * The child shares the parent's KV prefix in memory (metadata-only under unified KV, no KV buffer copy).
   * Logits, sampler state, and perplexity tracker are cloned so the child
   * can diverge independently. Fork from any branch — root or intermediate —
   * to build arbitrarily deep trees.
   *
   */
  fork(): Promise<Branch>;

  /**
   * Get a copy of this branch's captured logits snapshot.
   *
   * Returns n_vocab floats — the raw logit distribution from the last
   * prefill() or commit() call.
   *
   * Unlike {@link SessionContext.getLogits} (zero-copy view into shared
   * model memory, invalidated by next decode), this returns an independent
   * copy of the branch's internal snapshot. The returned Float32Array is
   * safe to hold across async boundaries and is not affected by subsequent
   * decode operations.
   *
   * @returns Independent copy of the logits snapshot (n_vocab elements)
   * @throws If no logits have been captured yet
   */
  getLogits(): Float32Array;

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
   * The primary way to feed tokens into a branch's KV cache.
   *
   * @param tokens - Token IDs to decode
   */
  prefill(tokens: number[]): Promise<void>;

  /** Sample next token from branch's frozen logits snapshot */
  sample(): number;

  /** Accept token for repeat-penalty tracking */
  accept(token: number): void;

  /**
   * Discard this branch — remove its divergent KV entries and free the handle
   *
   * Only removes KV entries divergent from the shared prefix; sibling branches
   * are unaffected. The disposed flag is set synchronously — any call to
   * produce(), commit(), etc. after prune() will throw immediately, even
   * before the returned promise resolves.
   *
   * RESTRICT mode: throws if children exist. Use {@link pruneSubtree} to
   * cascade-delete an entire subtree.
   */
  prune(): Promise<void>;

  /**
   * Discard this branch and all its descendants — CASCADE delete
   *
   * Iterative post-order traversal: prunes children first, then this branch.
   * Use when tearing down an entire subtree (e.g. abandoned search path).
   * Sets disposed synchronously, like {@link prune}.
   */
  pruneSubtree(): Promise<void>;

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

  /**
   * Apply dynamic logit adjustments for this branch only
   *
   * Unlike `logit_bias` in sampling params (which is cloned on fork), steer biases
   * are NOT inherited by child branches. Each branch manages its own steer state
   * independently. This makes steer ideal for path-dependent constraints.
   *
   * **Use cases:**
   * - **tsampler**: Block tokens that would create repeated N-grams based on
   *   this branch's specific generation history
   * - **Diverse beam search**: Penalize tokens already chosen by sibling beams
   *   to encourage output diversity across the beam
   * - **Dynamic constraints**: Apply token restrictions that change per-step
   *
   * **Sampling order:** Grammar → Logit Bias → Steer → Sampler Chain
   *
   * @param biases - Array of token adjustments. Use `-Infinity` to completely
   *   block a token, positive values to boost probability, negative to reduce.
   *
   * @example Block tokens for N-gram deduplication (tsampler pattern)
   * ```ts
   * // Compute which tokens would create repeated 4-grams
   * const blocked = computeNgramBlocks(generatedTokens, n=4);
   *
   * // Block those tokens for this sample only
   * branch.steer(blocked.map(t => ({ token: t, bias: -Infinity })));
   *
   * const { token } = await branch.produce();  // Blocked tokens won't be sampled
   * await branch.commit(token);
   *
   * // Clear for next iteration (recompute based on new history)
   * branch.clearSteer();
   * ```
   *
   * @example Diverse beam search
   * ```ts
   * // Each beam penalizes tokens chosen by siblings this step
   * for (const beam of beams) {
   *   // Collect tokens chosen by other beams
   *   const siblingTokens = beams
   *     .filter(b => b !== beam && b.lastToken !== undefined)
   *     .map(b => b.lastToken);
   *
   *   // Penalize sibling choices to encourage diversity
   *   beam.branch.steer(siblingTokens.map(t => ({ token: t, bias: -2.0 })));
   *
   *   const { token } = await beam.branch.produce();
   *   await beam.branch.commit(token);
   *   beam.lastToken = token;
   *   beam.branch.clearSteer();
   * }
   * ```
   *
   * @example Boost specific tokens
   * ```ts
   * // Boost "yes" and "no" tokens for a yes/no question
   * branch.steer([
   *   { token: yesTokenId, bias: 5.0 },
   *   { token: noTokenId, bias: 5.0 }
   * ]);
   * ```
   */
  steer(biases: Array<{ token: number; bias: number }>): void;

  /**
   * Clear all steer biases from this branch
   *
   * Removes any dynamic logit adjustments set by `steer()`. Call this after
   * each generation step if your steer constraints are computed per-step
   * (e.g., N-gram blocking where the blocked set changes as text grows).
   *
   * @example Per-step steer pattern
   * ```ts
   * for (let i = 0; i < maxTokens; i++) {
   *   // Compute constraints based on current state
   *   const blocked = computeConstraints(generatedTokens);
   *   branch.steer(blocked.map(t => ({ token: t, bias: -Infinity })));
   *
   *   const { token, isStop } = await branch.produce();
   *   if (isStop) break;
   *
   *   await branch.commit(token);
   *   branch.clearSteer();  // Reset for next iteration
   *   generatedTokens.push(token);
   * }
   * ```
   */
  clearSteer(): void;

  /**
   * Replace the sampler chain with new parameters (memoized)
   *
   * If the new params match the current chain's params, this is a no-op.
   * Otherwise the old chain is freed and a new one is created. Use for
   * Entropy-Driven Temperature (EDT) and other adaptive sampling strategies
   * that adjust parameters per-step.
   *
   * @param params - New sampling parameters
   *
   * @example Entropy-Driven Temperature
   * ```typescript
   * const entropy = ctx.modelEntropy('nats', branch.getLogits());
   * branch.setSamplerParams({ temperature: edtTemperature(entropy) });
   * const { token } = await branch.produce();
   * await branch.commit(token);
   * ```
   */
  setSamplerParams(params: SamplingParams): void;

  /**
   * Replace or remove the grammar constraint
   *
   * Pass a GBNF grammar string to constrain generation. Pass empty string
   * or undefined to remove the constraint. The grammar state is cloned on
   * fork(), so sibling branches can diverge independently after hot-swap.
   *
   * @param grammarStr - GBNF grammar string, or empty/undefined to remove
   *
   * @example Hot-swap grammar mid-generation
   * ```typescript
   * // Start unconstrained, then switch to JSON after detecting tool call
   * branch.setGrammar(jsonGrammar);
   * const { token } = await branch.produce();
   * ```
   */
  setGrammar(grammarStr?: string): void;

  /**
   * Sample next token without advancing state (async)
   *
   * Async contract: local branches resolve immediately; cloud branches
   * may perform an HTTP round-trip. Use {@link produceSync} when you know
   * the branch is local and want zero-overhead sampling.
   */
  produce(): Promise<Produced>;

  /**
   * Sample next token without advancing state (sync)
   *
   * Same as {@link produce} but synchronous. Use when you know the branch
   * is local and want to avoid the microtick overhead of a promise.
   */
  produceSync(): Produced;

  /**
   * Accept and decode — update branch state, then write token to KV
   *
   * Accepts the token into the sampler penalty window (for correct PPL
   * measurement), then decodes (writing to KV cache via AsyncWorker on
   * the libuv thread pool) and captures the resulting logits for the next
   * produce() call. Accept-first ordering with rollback: if decode throws,
   * sampler/grammar/metrics are restored from clones.
   *
   * @param token Token to commit (from produce())
   */
  commit(token: number): Promise<void>;

  /** Branch's current position */
  readonly position: number;

  /** Branch's perplexity */
  readonly perplexity: number;

  /** Internal handle (for debugging) */
  readonly handle: number;

  /** Whether this branch has been disposed */
  readonly disposed: boolean;

  /** Parent branch handle, or null if root */
  readonly parent: number | null;

  /** Child branch handles */
  readonly children: number[];

  /** True if this branch has no children */
  readonly isLeaf: boolean;

  /** True if this branch holds a KV lease */
  readonly isActive: boolean;

  /**
   * Async iterator — generate tokens until EOG
   *
   * Commit-before-yield semantics: every yielded token is already written
   * to KV and accepted into the sampler. Breaking out of the loop is clean —
   * no orphaned uncommitted tokens, perplexity reflects all yielded tokens.
   *
   * For inspect-before-commit (speculative decoding, tree search), use
   * the {@link produce}/{@link commit} protocol directly.
   *
   * @example Generate to completion
   * ```typescript
   * for await (const { token, text } of branch) {
   *   process.stdout.write(text);
   * }
   * ```
   *
   * @example Generate with consumer-side bound
   * ```typescript
   * const tokens = [];
   * for await (const { token } of branch) {
   *   tokens.push(token);
   *   if (tokens.length >= limit) break;
   * }
   * ```
   */
  [Symbol.asyncIterator](): AsyncIterableIterator<{ token: number; text: string }>;
}

/**
 * High-throughput multi-branch decode operations
 *
 * The naive approach to N-branch generation is N sequential llama_decode()
 * calls — each paying full GPU kernel launch overhead, memory barrier, and
 * PCIe round-trip. BranchStore eliminates this by packing all branches into
 * a single llama_batch and dispatching once: O(1) GPU round-trips regardless
 * of branch count. The GPU parallelizes across sequences within the batch,
 * so N branches approach the wall-time cost of 1.
 *
 * Two operations, two packing strategies:
 *
 * **commit()** — Generation step. Each branch contributes exactly 1 token.
 * Packs N tokens into a single batch via `decode_each` (one row per sequence,
 * all at their respective positions). Single `llama_decode()` call. Logits
 * captured per-branch at batch index `i`. O(N) total work, O(1) GPU
 * dispatches, O(1) amortized dispatch overhead per branch. Accept-first
 * ordering with rollback: accepts each token into its branch's repeat-penalty
 * window before decode, restores from clones if decode throws.
 *
 * **prefill()** — Bulk token injection. Each branch contributes a
 * variable-length token array. Uses a two-pass bin-packing algorithm:
 *
 * - *Pass 1 (planning)*: Greedy first-fit packs items into chunks ≤ nBatch.
 *   Items larger than nBatch get a dedicated chunk and fall through to
 *   decode_many's internal auto-chunking (ceil(nTokens / nBatch) calls).
 * - *Pass 2 (dispatch)*: Normal chunks dispatch via `decode_scatter` (one
 *   `llama_decode` per chunk). Logits are indexed by flattened cursor
 *   position: for item k in a chunk, logits live at `cursor + nTokens[k] - 1`.
 *
 * For T total tokens across N branches with batch capacity B:
 * - Best case (T ≤ B): 1 GPU dispatch, all branches in one batch.
 * - Worst case: ceil(T / B) dispatches. Each dispatch is fully packed.
 * - Amortized per-token GPU overhead: O(1/B) — vanishes as batch fills.
 *
 * Does NOT accept tokens into the sampler penalty window — use for
 * external/replayed tokens where repeat-penalty tracking is unwanted.
 * For model-generated tokens, use {@link commit} instead.
 *
 * Both methods take `[branch, token(s)]` tuples — the branch-to-token
 * binding is structural, not positional. After either call, each branch's
 * logits snapshot is updated with the output distribution from its decoded
 * token(s), ready for the next `produce()`/`sample()` call.
 *
 * @example 32-branch generation step — one GPU dispatch
 * ```typescript
 * const store = new BranchStore(ctx);
 * const entries = await Promise.all(branches.map(async b => [b, (await b.produce()).token] as [Branch, number]));
 * await store.commit(entries);  // 32 tokens, 1 llama_decode()
 * ```
 *
 * @example Best-of-N with batched commit
 * ```typescript
 * const store = new BranchStore(ctx);
 * const branches = [];
 * for (const _ of [1, 2, 3]) branches.push(await root.fork());
 *
 * for (let step = 0; step < 50; step++) {
 *   const produced = await Promise.all(branches.map(async b => [b, await b.produce()] as const));
 *   const live = produced.filter(([, p]) => !p.isStop);
 *   if (!live.length) break;
 *   await store.commit(live.map(([b, p]) => [b, p.token]));
 * }
 * ```
 *
 * @example Asymmetric prefill — variable-length injections, auto-chunked
 * ```typescript
 * await store.prefill([
 *   [branchA, systemPromptTokens],   // 200 tokens
 *   [branchB, shortQueryTokens],     //  12 tokens
 *   [branchC, longDocumentTokens],   // 800 tokens
 * ]);
 * // Bin-packed into ceil(1012 / nBatch) GPU dispatches
 * ```
 *
 * @category Branching
 */
export class BranchStore {
  constructor(ctx: SessionContext);

  /**
   * Batched single-token commit for model-generated tokens
   *
   * Each tuple `[branch, token]` binds one token to one branch.
   * Accepts each token into its branch's repeat-penalty window (for correct
   * PPL measurement), then decodes all N tokens in a single llama_decode()
   * call via decode_each and captures logits per-branch. Accept-first
   * ordering with rollback: if decode throws, sampler/grammar/metrics are
   * restored from clones taken before the accept.
   *
   * @param entries - Array of `[branch, token]` tuples (branches must not be disposed)
   * @throws If any branch is disposed
   */
  commit(entries: [Branch, number][]): Promise<void>;

  /**
   * Batched variable-length prefill for external tokens
   *
   * Each tuple `[branch, tokens]` binds a token array to one branch.
   * Each branch can receive a different number of tokens — decode_scatter
   * handles variable-length runs and auto-chunks to fit nBatch.
   *
   * Does NOT call accept_token — use for external/replayed tokens where
   * repeat-penalty tracking is unwanted. For model-generated tokens,
   * use {@link commit} instead.
   *
   * @param entries - Array of `[branch, tokens]` tuples (branches must not be disposed)
   * @throws If any branch is disposed
   */
  prefill(entries: [Branch, number[]][]): Promise<void>;

  /**
   * Retain only the winner branch — evict all other leases and free their slots.
   *
   * Nuclear operation: calls `kv::seq_keep` on the winner's seq_id (stripping all
   * other sequences from KV cache in a single pass), then frees all loser slots
   * and rebuilds the vacancy list. The winner's topology is reset (no parent, no children).
   *
   * @param winner - The branch to keep (must not be disposed, must hold a lease)
   * @throws If winner is disposed or has no lease
   */
  retainOnly(winner: Branch): Promise<void>;

  /** Number of available seq_id leases */
  readonly available: number;
}
