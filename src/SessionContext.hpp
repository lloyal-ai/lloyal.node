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

private:
  // ===== INTERNAL STATE =====

  std::shared_ptr<llama_model> _model;
  llama_context* _context = nullptr;
  bool _disposed = false;

  // Helpers
  inline void ensureNotDisposed() {
    if (_disposed) {
      throw Napi::Error::New(Env(), "Context has been disposed");
    }
  }

  inline const llama_vocab* getVocabOrThrow() {
    const llama_vocab* vocab = lloyal::tokenizer::get_vocab(_model.get());
    if (!vocab) {
      throw Napi::Error::New(Env(), "Failed to get vocabulary");
    }
    return vocab;
  }
};

/**
 * Factory function: createContext(options)
 *
 * Args: { modelPath: string, nCtx?: number, nThreads?: number }
 * Returns: SessionContext
 */
Napi::Value CreateContext(const Napi::CallbackInfo& info);

} // namespace liblloyal_node
