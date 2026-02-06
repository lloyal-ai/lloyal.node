#pragma once

#include <napi.h>
#include <lloyal/tokenizer.hpp>
#include <lloyal/metrics.hpp>
#include <lloyal/branch.hpp>
#include <llama/llama.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace liblloyal_node {

/**
 * Sampling parameters adapter for liblloyal compatibility
 *
 * liblloyal expects snake_case parameter names (top_k, penalty_repeat, etc.)
 * This struct satisfies liblloyal's SamplingParamsLike concept.
 */
struct LloyalSamplingParams {
  std::optional<float> temperature;
  std::optional<int32_t> top_k;
  std::optional<float> top_p;
  std::optional<float> typical_p;
  std::optional<float> min_p;
  std::optional<float> penalty_repeat;
  std::optional<float> penalty_freq;
  std::optional<float> penalty_present;
  std::optional<int32_t> penalty_last_n;
  std::optional<uint32_t> seed;

  // Equality operator for detecting param changes
  bool operator==(const LloyalSamplingParams& other) const {
    return temperature == other.temperature &&
           top_k == other.top_k &&
           top_p == other.top_p &&
           typical_p == other.typical_p &&
           min_p == other.min_p &&
           penalty_repeat == other.penalty_repeat &&
           penalty_freq == other.penalty_freq &&
           penalty_present == other.penalty_present &&
           penalty_last_n == other.penalty_last_n &&
           seed == other.seed;
  }

  bool operator!=(const LloyalSamplingParams& other) const {
    return !(*this == other);
  }
};

/**
 * Thin N-API wrapper over liblloyal
 *
 * Exposes raw llama.cpp inference primitives for testing and lightweight inference.
 * Design philosophy: Keep it thin, raw, and close to liblloyal APIs.
 *
 * Mirrors HybridSessionContext.cpp patterns but uses N-API instead of Nitro.
 */
class SessionContext : public Napi::ObjectWrap<SessionContext> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  SessionContext(const Napi::CallbackInfo& info);
  ~SessionContext();

  /**
   * Initialize context with model and llama_context
   * Called by CreateContext factory function after model loading completes
   * Pattern matches HybridSessionContext::initializeContext()
   */
  void initializeContext(
    std::shared_ptr<llama_model> model,
    llama_context* context,
    int32_t nBatch = lloyal::defaults::N_BATCH_INIT
  );

private:
  // ===== CORE PRIMITIVES =====

  /**
   * Get raw logits (zero-copy Float32Array)
   * Returns: Float32Array pointing directly to llama.cpp memory
   * Lifetime: Valid until next decode() call
   */
  Napi::Value getLogits(const Napi::CallbackInfo& info);

  /**
   * Decode tokens through model
   * Args: tokens (number[]), position (number), seqId? (number, default 0)
   * Returns: Promise<void>
   *
   * The seqId parameter specifies which KV cache sequence to update.
   * Use different seqIds for independent parallel sequences.
   */
  Napi::Value decode(const Napi::CallbackInfo& info);

  /**
   * Tokenize text to token IDs
   * Args: text (string)
   * Returns: Promise<number[]>
   */
  Napi::Value tokenize(const Napi::CallbackInfo& info);

  /**
   * Detokenize tokens to text
   * Args: tokens (number[])
   * Returns: Promise<string>
   */
  Napi::Value detokenize(const Napi::CallbackInfo& info);

  /**
   * Convert single token to text (sync, fast)
   * Args: token (number)
   * Returns: string
   */
  Napi::Value tokenToText(const Napi::CallbackInfo& info);

  /**
   * Check if token is a stop token (EOS)
   * Args: token (number)
   * Returns: boolean
   */
  Napi::Value isStopToken(const Napi::CallbackInfo& info);

  /**
   * Get the model's end-of-generation token ID
   * Returns EOT token, falling back to EOS for Zephyr-style models
   */
  Napi::Value getEogToken(const Napi::CallbackInfo& info);

  /**
   * Get the model's turn separator token IDs
   * Returns tokens that close an assistant turn and transition to the next message
   */
  Napi::Value getTurnSeparator(const Napi::CallbackInfo& info);

  /**
   * Format messages using model's chat template
   * Args: messagesJson (string), templateOverride (optional string)
   * Returns: Promise<{ prompt: string, stopTokens: string[] }>
   */
  Napi::Value formatChat(const Napi::CallbackInfo& info);

  /**
   * Get current KV cache position (number of tokens in cache)
   * Returns: number
   */
  Napi::Value kvCacheSize(const Napi::CallbackInfo& info);

  // ===== NATIVE REFERENCE IMPLEMENTATIONS =====

  /**
   * Native greedy sampling (for validation)
   * Returns: number (token ID)
   */
  Napi::Value greedySample(const Napi::CallbackInfo& info);

  /**
   * Native sampling with full parameters (for benchmarking)
   * Args: params (optional object with temperature, topK, topP, etc.)
   * Returns: number (token ID)
   */
  Napi::Value sample(const Napi::CallbackInfo& info);

  // ===== LIFECYCLE =====

  /**
   * Free native resources
   */
  Napi::Value dispose(const Napi::CallbackInfo& info);

  // ===== PROPERTIES =====

  Napi::Value getVocabSize(const Napi::CallbackInfo& info);
  Napi::Value getMemorySize(const Napi::CallbackInfo& info);

  // ===== KV CACHE MANAGEMENT =====

  Napi::Value kvCacheRemove(const Napi::CallbackInfo& info);
  Napi::Value kvCacheSave(const Napi::CallbackInfo& info);
  Napi::Value kvCacheLoad(const Napi::CallbackInfo& info);
  Napi::Value kvCacheClear(const Napi::CallbackInfo& info);

  /**
   * Atomic clear+reseed operation for KV cache compression
   * Args: sinks (Array<number>), tail (Array<number>)
   * Returns: void (Promise)
   */
  Napi::Value clearAndReseed(const Napi::CallbackInfo& info);

  // ===== KV SEQUENCE OPERATIONS =====

  /**
   * Copy KV cache from one sequence to another
   * Args: srcSeqId (number), dstSeqId (number), p0? (number), p1? (number)
   */
  Napi::Value kvSeqCopy(const Napi::CallbackInfo& info);

  /**
   * Keep only specified sequence, remove all others
   * Args: seqId (number)
   */
  Napi::Value kvSeqKeep(const Napi::CallbackInfo& info);

  /**
   * Get max position in sequence
   * Args: seqId (number)
   * Returns: number (-1 if empty)
   */
  Napi::Value kvSeqPosMax(const Napi::CallbackInfo& info);

  // ===== HANDLE-BASED GRAMMAR =====

  /**
   * Create a new grammar sampler, returns handle
   * Args: grammarStr (string)
   * Returns: number (handle)
   */
  Napi::Value createSampler(const Napi::CallbackInfo& info);

  /**
   * Apply grammar constraints to logits buffer
   * Args: handle (number), logitsBuffer (ArrayBuffer)
   */
  Napi::Value applySampler(const Napi::CallbackInfo& info);

  /**
   * Accept token to advance grammar parser state
   * Args: handle (number), tokenId (number)
   */
  Napi::Value acceptSamplerToken(const Napi::CallbackInfo& info);

  /**
   * Clone a grammar sampler
   * Args: handle (number)
   * Returns: number (new handle)
   */
  Napi::Value cloneSampler(const Napi::CallbackInfo& info);

  /**
   * Free a grammar sampler
   * Args: handle (number)
   */
  Napi::Value freeSamplerHandle(const Napi::CallbackInfo& info);

  // ===== ATOMIC DECODE+CAPTURE =====

  /**
   * Decode tokens and capture logits atomically (mutex protected)
   * Args: tokens (number[]), position (number), seqId (number), destBuffer (ArrayBuffer)
   * Returns: Promise<void>
   */
  Napi::Value decodeAndCapture(const Napi::CallbackInfo& info);

  /**
   * Write KV cache state + tokens to a file for disk persistence
   * Args: sequenceId (number), filepath (string), tokens (number[])
   * Returns: Promise<number> (bytes written)
   */
  Napi::Value kvCacheWriteFile(const Napi::CallbackInfo& info);

  /**
   * Read KV cache state + tokens from a file
   * Args: sequenceId (number), filepath (string)
   * Returns: Promise<{ tokens: number[], bytesRead: number }>
   */
  Napi::Value kvCacheReadFile(const Napi::CallbackInfo& info);

  // ===== HELPERS =====
  // Utility functions (not yet implemented)

  Napi::Value jsonSchemaToGrammar(const Napi::CallbackInfo& info);
  Napi::Value validateChatTemplate(const Napi::CallbackInfo& info);

  // ===== EMBEDDING EXTRACTION =====

  /**
   * Encode tokens for embedding extraction
   * Unlike decode(), marks ALL tokens with logits=true
   * Args: tokens (number[])
   * Returns: Promise<void>
   */
  Napi::Value encode(const Napi::CallbackInfo& info);

  /**
   * Get embeddings from context (after encode)
   * Args: normalize (optional boolean, default true for L2)
   * Returns: Float32Array
   */
  Napi::Value getEmbeddings(const Napi::CallbackInfo& info);

  /**
   * Get embedding dimension for model
   * Returns: number
   */
  Napi::Value getEmbeddingDimension(const Napi::CallbackInfo& info);

  /**
   * Check if context has pooling enabled
   * Returns: boolean
   */
  Napi::Value hasPooling(const Napi::CallbackInfo& info);

  // ===== METRICS API =====

  /**
   * Compute surprisal for a specific token
   * Args: pickedTokenId (number), base? (string: "nats" | "bits" | "base10")
   * Returns: number (surprisal in specified base)
   */
  Napi::Value modelSurprisal(const Napi::CallbackInfo& info);

  /**
   * Compute entropy of logits distribution
   * Args: base? (string: "nats" | "bits" | "base10")
   * Returns: number (entropy in specified base)
   */
  Napi::Value modelEntropy(const Napi::CallbackInfo& info);

  /**
   * Create a new perplexity tracker
   * Returns: number (handle)
   */
  Napi::Value createPerplexityTracker(const Napi::CallbackInfo& info);

  /**
   * Add surprisal value to tracker
   * Args: handle (number), surprisal (number)
   */
  Napi::Value addSurprisal(const Napi::CallbackInfo& info);

  /**
   * Get current perplexity value
   * Args: handle (number)
   * Returns: number (perplexity)
   */
  Napi::Value getPerplexity(const Napi::CallbackInfo& info);

  /**
   * Clone perplexity tracker
   * Args: sourceHandle (number)
   * Returns: number (new handle)
   */
  Napi::Value clonePerplexityTracker(const Napi::CallbackInfo& info);

  /**
   * Reset tracker to initial state
   * Args: handle (number)
   */
  Napi::Value resetPerplexityTracker(const Napi::CallbackInfo& info);

  /**
   * Get number of tokens tracked
   * Args: handle (number)
   * Returns: number (count)
   */
  Napi::Value getPerplexityCount(const Napi::CallbackInfo& info);

  /**
   * Free perplexity tracker resources
   * Args: handle (number)
   */
  Napi::Value freePerplexityTracker(const Napi::CallbackInfo& info);

  // ===== BRANCH API (internal, wrapped by lib/Branch.ts) =====

  Napi::Value _branchCreate(const Napi::CallbackInfo& info);
  Napi::Value _branchFork(const Napi::CallbackInfo& info);
  Napi::Value _branchCaptureLogits(const Napi::CallbackInfo& info);
  Napi::Value _branchDecodeAndCaptureOne(const Napi::CallbackInfo& info);
  Napi::Value _branchDecodeAndCaptureBatch(const Napi::CallbackInfo& info);
  Napi::Value _branchSample(const Napi::CallbackInfo& info);
  Napi::Value _branchAccept(const Napi::CallbackInfo& info);
  Napi::Value _branchGetSeqId(const Napi::CallbackInfo& info);
  Napi::Value _branchGetPosition(const Napi::CallbackInfo& info);
  Napi::Value _branchGetPerplexity(const Napi::CallbackInfo& info);
  Napi::Value _branchPrune(const Napi::CallbackInfo& info);
  Napi::Value _branchDestroy(const Napi::CallbackInfo& info);
  Napi::Value _branchSamplerChainReseed(const Napi::CallbackInfo& info);

private:
  // ===== INTERNAL STATE =====

  std::shared_ptr<llama_model> _model;
  llama_context* _context = nullptr;
  bool _disposed = false;
  int32_t _nBatch = lloyal::defaults::N_BATCH_INIT;

  // Persistent sampling chain (for repeat penalty tracking across tokens)
  // Pattern from branch.hpp: create once via sampler::create_chain(), reuse across samples.
  // Penalty sampler's history is updated via sampler::accept() after each sample.
  // This enables proper repeat penalty tracking across long generations and clearAndReseed().
  llama_sampler* _samplerChain = nullptr;
  LloyalSamplingParams _samplerParams;  // Track current params to detect changes

  // ===== HANDLE-BASED GRAMMAR =====
  std::unordered_map<int32_t, llama_sampler*> _samplerHandles;
  int32_t _nextSamplerHandle = 1;

  // ===== HANDLE-BASED PERPLEXITY TRACKING =====
  std::unordered_map<int32_t, lloyal::metrics::PerplexityHandle> _perplexityHandles;
  int32_t _nextPerplexityHandle = 1;

  // ===== BRANCH STORE =====
  lloyal::branch::BranchStore _branchStore{16};  // capacity 16

  // ===== TURN SEPARATOR CACHE =====
  std::vector<llama_token> _turnSeparatorCache;
  bool _turnSeparatorCached = false;

  // ===== DECODE MUTEX =====
  std::mutex _decodeMutex;

  // ===== LOGITS BUFFER MANAGEMENT (Memoization + Revocation) =====
  //
  // Pattern: "Memoized Step-Scoped Views with Explicit Revocation"
  //
  // - Memoization: If getLogits() called twice in same step, return same buffer
  // - Revocation: On decode(), detach previous buffer to prevent use-after-invalidation
  //
  // See: lloyal::logits::get() for the underlying safe wrapper
  uint64_t _decodeStepId = 0;                           // Incremented on each decode()
  uint64_t _logitsStepId = 0;                           // Step when _logitsBuffer was created
  Napi::Reference<Napi::ArrayBuffer> _logitsBufferRef;  // Strong reference - kept alive so we can Detach() on revocation

  // ===== INLINE HELPERS =====
  // Pattern matches HybridSessionContext.hpp:170-176

  inline void ensureNotDisposed() {
    if (_disposed) {
      throw Napi::Error::New(Env(), "Context has been disposed");
    }
  }

  inline llama_seq_id toSeqId(double id) {
    return static_cast<llama_seq_id>(id);
  }

  inline llama_pos toPos(double pos) {
    return static_cast<llama_pos>(pos);
  }

  // Parse base string ("nats", "bits", "base10") to lloyal::metrics::Base enum
  static lloyal::metrics::Base parseBase(const std::string& baseStr);

  /**
   * Invalidate any active logits buffer (The Kill Switch)
   *
   * Called before any operation that would invalidate the logits pointer:
   * - decode()
   * - encode()
   * - dispose()
   *
   * Detaches the ArrayBuffer so any JS code holding a reference
   * will get a TypeError when trying to access it.
   */
  void invalidateLogits();
};

/**
 * Factory function: createContext(options)
 *
 * Args: { modelPath: string, nCtx?: number, nThreads?: number }
 * Returns: SessionContext
 */
Napi::Value CreateContext(const Napi::CallbackInfo& info);

} // namespace liblloyal_node
