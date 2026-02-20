#pragma once

#include <napi.h>
#include <lloyal/tokenizer.hpp>
#include <lloyal/metrics.hpp>
#include <lloyal/branch.hpp>
#include <lloyal/chat_in.hpp>
#include <llama/llama.h>
#include <memory>
#include <optional>
#include <string>
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
  Napi::Value parseChatOutput(const Napi::CallbackInfo& info);

  /**
   * Get current KV cache position (number of tokens in cache)
   * Returns: number
   */
  Napi::Value kvCacheSize(const Napi::CallbackInfo& info);

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

  // ===== BRANCH API (internal, wrapped by lib/Branch.ts) =====

  Napi::Value _branchCreate(const Napi::CallbackInfo& info);
  Napi::Value _branchFork(const Napi::CallbackInfo& info);
  Napi::Value _branchPrefill(const Napi::CallbackInfo& info);
  Napi::Value _branchSample(const Napi::CallbackInfo& info);
  Napi::Value _branchAccept(const Napi::CallbackInfo& info);
  Napi::Value _branchGetPosition(const Napi::CallbackInfo& info);
  Napi::Value _branchGetPerplexity(const Napi::CallbackInfo& info);
  Napi::Value _branchGetLogits(const Napi::CallbackInfo& info);
  Napi::Value _branchPrune(const Napi::CallbackInfo& info);
  Napi::Value _branchPruneSubtree(const Napi::CallbackInfo& info);
  Napi::Value _branchParent(const Napi::CallbackInfo& info);
  Napi::Value _branchChildren(const Napi::CallbackInfo& info);
  Napi::Value _branchIsLeaf(const Napi::CallbackInfo& info);
  Napi::Value _branchIsActive(const Napi::CallbackInfo& info);
  Napi::Value _branchSamplerChainReseed(const Napi::CallbackInfo& info);
  Napi::Value _branchSteer(const Napi::CallbackInfo& info);
  Napi::Value _branchClearSteer(const Napi::CallbackInfo& info);
  Napi::Value _branchSetSamplerParams(const Napi::CallbackInfo& info);
  Napi::Value _branchSetGrammar(const Napi::CallbackInfo& info);
  Napi::Value _branchModelEntropy(const Napi::CallbackInfo& info);
  Napi::Value _branchModelSurprisal(const Napi::CallbackInfo& info);
  Napi::Value _branchGetSamplingPerplexity(const Napi::CallbackInfo& info);
  Napi::Value _branchSetLogitBias(const Napi::CallbackInfo& info);
  Napi::Value _branchClearLogitBias(const Napi::CallbackInfo& info);

  // ===== STORE API (internal, wrapped by lib/BranchStore.js) =====

  Napi::Value _storeCommit(const Napi::CallbackInfo& info);
  Napi::Value _storePrefill(const Napi::CallbackInfo& info);
  Napi::Value _storeRetainOnly(const Napi::CallbackInfo& info);
  Napi::Value _storeAvailable(const Napi::CallbackInfo& info);

private:
  // ===== INTERNAL STATE =====

  std::shared_ptr<llama_model> _model;
  llama_context* _context = nullptr;
  bool _disposed = false;
  int32_t _nBatch = lloyal::defaults::N_BATCH_INIT;

  // ===== BRANCH STORE =====
  lloyal::branch::BranchStore _branchStore{16};  // capacity 16

  // ===== TURN SEPARATOR CACHE =====
  std::vector<llama_token> _turnSeparatorCache;
  bool _turnSeparatorCached = false;

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
};

/**
 * Factory function: createContext(options)
 *
 * Args: { modelPath: string, nCtx?: number, nThreads?: number }
 * Returns: SessionContext
 */
Napi::Value CreateContext(const Napi::CallbackInfo& info);

} // namespace liblloyal_node
