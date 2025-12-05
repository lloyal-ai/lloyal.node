#pragma once

#include <napi.h>
#include <lloyal/tokenizer.hpp>
#include <llama/llama.h>
#include <memory>
#include <string>
#include <vector>

namespace liblloyal_node {

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
    llama_context* context
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
   * Args: tokens (number[]), position (number)
   * Returns: Promise<void>
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
   * Native entropy computation (for validation)
   * Returns: number (entropy in nats)
   */
  Napi::Value computeEntropy(const Napi::CallbackInfo& info);

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

  // ===== GRAMMAR-CONSTRAINED GENERATION =====
  // (To be implemented in Phase 4)

  Napi::Value getTokenScores(const Napi::CallbackInfo& info);
  Napi::Value initGrammar(const Napi::CallbackInfo& info);
  Napi::Value applyGrammar(const Napi::CallbackInfo& info);
  Napi::Value acceptToken(const Napi::CallbackInfo& info);
  Napi::Value resetGrammar(const Napi::CallbackInfo& info);
  Napi::Value freeGrammar(const Napi::CallbackInfo& info);

  // ===== KV CACHE MANAGEMENT =====

  Napi::Value kvCacheRemove(const Napi::CallbackInfo& info);
  Napi::Value kvCacheSave(const Napi::CallbackInfo& info);
  Napi::Value kvCacheLoad(const Napi::CallbackInfo& info);
  Napi::Value kvCacheClear(const Napi::CallbackInfo& info);

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
  // (To be implemented in Phase 6)

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

private:
  // ===== INTERNAL STATE =====

  std::shared_ptr<llama_model> _model;
  llama_context* _context = nullptr;
  bool _disposed = false;

  // Grammar sampler state (persistent across tokens within a generation)
  // Pattern matches HybridSessionContext.hpp:197-200
  llama_sampler* _grammarSampler = nullptr;
  std::string _currentGrammar;  // Track current grammar string to avoid re-initialization

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
  Napi::Reference<Napi::ArrayBuffer> _logitsBufferRef;  // Weak reference to detach on revocation

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
