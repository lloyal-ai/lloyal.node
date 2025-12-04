#include "SessionContext.hpp"
#include "BackendManager.hpp"
#include "FileSystem.h"
#include <lloyal/decoder.hpp>
#include <lloyal/sampler.hpp>
#include <lloyal/tokenizer.hpp>
#include <lloyal/common.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/chat_template.hpp>
#include <lloyal/grammar.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/embedding.hpp>
#include <cmath>

namespace liblloyal_node {

// ===== ADAPTER FOR LIBLLOYAL COMPATIBILITY =====
//
// liblloyal expects old snake_case parameter names (top_k, penalty_repeat, etc.)
// but our new API uses camelCase and nested groups (topK, penalties.repeat, etc.)
//
// This adapter struct satisfies liblloyal's SamplingParamsLike concept
// while allowing us to use the new developer-friendly API surface.
// Pattern matches HybridSessionContext.cpp:18-65
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
};

// Convert JS object params → liblloyal-compatible structure
// Note: For now this is a placeholder - Phase 5 will implement full conversion
// from the new nested API structure (penalties, advanced, etc.)
static LloyalSamplingParams adaptSamplingParamsFromJS(Napi::Object paramsObj) {
  LloyalSamplingParams adapted;

  // Direct mappings (camelCase → snake_case)
  if (paramsObj.Has("temperature") && paramsObj.Get("temperature").IsNumber()) {
    adapted.temperature = paramsObj.Get("temperature").As<Napi::Number>().FloatValue();
  }
  if (paramsObj.Has("topK") && paramsObj.Get("topK").IsNumber()) {
    adapted.top_k = paramsObj.Get("topK").As<Napi::Number>().Int32Value();
  }
  if (paramsObj.Has("topP") && paramsObj.Get("topP").IsNumber()) {
    adapted.top_p = paramsObj.Get("topP").As<Napi::Number>().FloatValue();
  }
  if (paramsObj.Has("minP") && paramsObj.Get("minP").IsNumber()) {
    adapted.min_p = paramsObj.Get("minP").As<Napi::Number>().FloatValue();
  }
  if (paramsObj.Has("seed") && paramsObj.Get("seed").IsNumber()) {
    adapted.seed = static_cast<uint32_t>(paramsObj.Get("seed").As<Napi::Number>().Int64Value());
  }

  // Extract from penalties group
  if (paramsObj.Has("penalties") && paramsObj.Get("penalties").IsObject()) {
    Napi::Object penalties = paramsObj.Get("penalties").As<Napi::Object>();

    if (penalties.Has("repeat") && penalties.Get("repeat").IsNumber()) {
      adapted.penalty_repeat = penalties.Get("repeat").As<Napi::Number>().FloatValue();
    }
    if (penalties.Has("frequency") && penalties.Get("frequency").IsNumber()) {
      adapted.penalty_freq = penalties.Get("frequency").As<Napi::Number>().FloatValue();
    }
    if (penalties.Has("presence") && penalties.Get("presence").IsNumber()) {
      adapted.penalty_present = penalties.Get("presence").As<Napi::Number>().FloatValue();
    }
    if (penalties.Has("lastN") && penalties.Get("lastN").IsNumber()) {
      adapted.penalty_last_n = penalties.Get("lastN").As<Napi::Number>().Int32Value();
    }
  }

  // TODO Phase 5: Extract from advanced group (mirostat, dry, xtc)
  // if (paramsObj.Has("advanced") && paramsObj.Get("advanced").IsObject()) {
  //   Napi::Object advanced = paramsObj.Get("advanced").As<Napi::Object>();
  //   adapted.typical_p = advanced.Get("typicalP").As<Napi::Number>().FloatValue();
  //   // Note: mirostat, dry, xtc not yet supported in liblloyal
  // }

  return adapted;
}

// ===== ASYNC WORKER CLASSES =====

/**
 * AsyncWorker for kvCacheRemove operation
 * Pattern matches HybridSessionContext.cpp:550-571
 */
class KVCacheRemoveWorker : public Napi::AsyncWorker {
public:
  KVCacheRemoveWorker(Napi::Env env, llama_context* ctx, double sequenceId, double start, double end)
    : AsyncWorker(env), _deferred(env), _ctx(ctx),
      _sequenceId(static_cast<llama_seq_id>(sequenceId)),
      _start(static_cast<llama_pos>(start)),
      _end(static_cast<llama_pos>(end)) {}

  void Execute() override {
    bool success = lloyal::kv::remove_range(_ctx, _sequenceId, _start, _end);
    if (!success) {
      SetError("Failed to remove KV range - see logs for details");
    }
  }

  void OnOK() override {
    _deferred.Resolve(Env().Undefined());
  }

  void OnError(const Napi::Error& err) override {
    _deferred.Reject(err.Value());
  }

  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  llama_context* _ctx;
  llama_seq_id _sequenceId;
  llama_pos _start;
  llama_pos _end;
};

/**
 * AsyncWorker for kvCacheSave operation
 * Pattern matches HybridSessionContext.cpp:585-614
 */
class KVCacheSaveWorker : public Napi::AsyncWorker {
public:
  KVCacheSaveWorker(Napi::Env env, llama_context* ctx, double sequenceId)
    : AsyncWorker(env), _deferred(env), _ctx(ctx),
      _sequenceId(static_cast<llama_seq_id>(sequenceId)) {}

  void Execute() override {
    // Get size needed for state buffer
    size_t size = lloyal::kv::state_size(_ctx, _sequenceId);
    if (size == 0) {
      SetError("Failed to get state size - both per-sequence and global queries failed");
      return;
    }

    // Allocate buffer
    _stateData.resize(size);

    // Save state to buffer
    size_t written = lloyal::kv::state_save(_ctx, _sequenceId, _stateData.data(), size);
    if (written == 0) {
      SetError("Failed to save state - both per-sequence and global save failed");
      return;
    }

    // Truncate to actual written size if needed
    if (written < size) {
      _stateData.resize(written);
    }
  }

  void OnOK() override {
    Napi::Env env = Env();
    // Create Buffer (Node.js version of ArrayBuffer)
    Napi::Buffer<uint8_t> buffer = Napi::Buffer<uint8_t>::Copy(env, _stateData.data(), _stateData.size());
    _deferred.Resolve(buffer);
  }

  void OnError(const Napi::Error& err) override {
    _deferred.Reject(err.Value());
  }

  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  llama_context* _ctx;
  llama_seq_id _sequenceId;
  std::vector<uint8_t> _stateData;
};

/**
 * AsyncWorker for kvCacheLoad operation
 * Pattern matches HybridSessionContext.cpp:616-642
 */
class KVCacheLoadWorker : public Napi::AsyncWorker {
public:
  KVCacheLoadWorker(Napi::Env env, llama_context* ctx, double sequenceId,
                    const uint8_t* stateData, size_t stateSize)
    : AsyncWorker(env), _deferred(env), _ctx(ctx),
      _sequenceId(static_cast<llama_seq_id>(sequenceId)),
      _stateData(stateData, stateData + stateSize) {}

  void Execute() override {
    if (_stateData.empty()) {
      SetError("Invalid state buffer - cannot restore");
      return;
    }

    // Restore state from buffer
    size_t read = lloyal::kv::state_load(_ctx, _sequenceId, _stateData.data(), _stateData.size());
    if (read == 0) {
      SetError("Failed to load state - both per-sequence and global restore failed");
    }
  }

  void OnOK() override {
    _deferred.Resolve(Env().Undefined());
  }

  void OnError(const Napi::Error& err) override {
    _deferred.Reject(err.Value());
  }

  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  llama_context* _ctx;
  llama_seq_id _sequenceId;
  std::vector<uint8_t> _stateData;
};

/**
 * AsyncWorker for kvCacheClear operation
 * Pattern matches HybridSessionContext.cpp:279-290
 */
class KVCacheClearWorker : public Napi::AsyncWorker {
public:
  KVCacheClearWorker(Napi::Env env, llama_context* ctx)
    : AsyncWorker(env), _deferred(env), _ctx(ctx) {}

  void Execute() override {
    // Use convenience overload
    lloyal::kv::clear_all(_ctx);
  }

  void OnOK() override {
    _deferred.Resolve(Env().Undefined());
  }

  void OnError(const Napi::Error& err) override {
    _deferred.Reject(err.Value());
  }

  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  llama_context* _ctx;
};

/**
 * AsyncWorker for kvCacheWriteFile operation
 * Writes KV cache state + tokens to a file for disk persistence
 */
class KVCacheWriteFileWorker : public Napi::AsyncWorker {
public:
  KVCacheWriteFileWorker(Napi::Env env, llama_context* ctx, llama_seq_id seq,
                         const std::string& filepath, std::vector<llama_token> tokens)
    : AsyncWorker(env), _deferred(env), _ctx(ctx), _seq(seq),
      _filepath(filepath), _tokens(std::move(tokens)) {}

  void Execute() override {
    _bytesWritten = lloyal::kv::write_file(_ctx, _seq, _filepath, _tokens);
    if (_bytesWritten == 0) {
      SetError("Failed to write KV cache to file");
    }
  }

  void OnOK() override {
    _deferred.Resolve(Napi::Number::New(Env(), static_cast<double>(_bytesWritten)));
  }

  void OnError(const Napi::Error& err) override {
    _deferred.Reject(err.Value());
  }

  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  llama_context* _ctx;
  llama_seq_id _seq;
  std::string _filepath;
  std::vector<llama_token> _tokens;
  size_t _bytesWritten = 0;
};

/**
 * AsyncWorker for kvCacheReadFile operation
 * Reads KV cache state + tokens from a file
 */
class KVCacheReadFileWorker : public Napi::AsyncWorker {
public:
  KVCacheReadFileWorker(Napi::Env env, llama_context* ctx, llama_seq_id seq,
                        const std::string& filepath)
    : AsyncWorker(env), _deferred(env), _ctx(ctx), _seq(seq), _filepath(filepath) {}

  void Execute() override {
    try {
      _result = lloyal::kv::read_file(_ctx, _seq, _filepath);
    } catch (const std::exception& e) {
      SetError(e.what());
    }
  }

  void OnOK() override {
    Napi::Env env = Env();
    Napi::Object result = Napi::Object::New(env);

    // Convert tokens to JS array
    Napi::Array jsTokens = Napi::Array::New(env, _result.tokens.size());
    for (size_t i = 0; i < _result.tokens.size(); i++) {
      jsTokens[i] = Napi::Number::New(env, static_cast<double>(_result.tokens[i]));
    }

    result.Set("tokens", jsTokens);
    result.Set("bytesRead", Napi::Number::New(env, static_cast<double>(_result.bytes_read)));

    _deferred.Resolve(result);
  }

  void OnError(const Napi::Error& err) override {
    _deferred.Reject(err.Value());
  }

  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  llama_context* _ctx;
  llama_seq_id _seq;
  std::string _filepath;
  lloyal::kv::FileData _result;
};

/**
 * AsyncWorker for tokenize operation
 */
class TokenizeWorker : public Napi::AsyncWorker {
public:
  TokenizeWorker(Napi::Env env, std::shared_ptr<llama_model> model, const std::string& text)
    : AsyncWorker(env), _deferred(env), _model(model), _text(text) {}

  void Execute() override {
    // Use convenience overload that auto-extracts vocab and handles add_bos
    _result = lloyal::tokenizer::tokenize(_model.get(), _text);
    if (_result.empty()) {
      SetError("Tokenization failed");
    }
  }

  void OnOK() override {
    Napi::Env env = Env();
    Napi::Array jsTokens = Napi::Array::New(env, _result.size());
    for (size_t i = 0; i < _result.size(); i++) {
      jsTokens[i] = Napi::Number::New(env, static_cast<double>(_result[i]));
    }
    _deferred.Resolve(jsTokens);
  }

  void OnError(const Napi::Error& err) override {
    _deferred.Reject(err.Value());
  }

  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  std::shared_ptr<llama_model> _model;
  std::string _text;
  std::vector<llama_token> _result;
};

/**
 * AsyncWorker for decode operation
 */
class DecodeWorker : public Napi::AsyncWorker {
public:
  DecodeWorker(Napi::Env env, llama_context* ctx, const std::vector<llama_token>& tokens, int32_t pos)
    : AsyncWorker(env), _deferred(env), _ctx(ctx), _tokens(tokens), _pos(pos) {}

  void Execute() override {
    try {
      lloyal::decoder::decode_tokens(_ctx, _tokens, _pos, lloyal::defaults::N_BATCH_PROCESS);
    } catch (const std::exception& e) {
      SetError(e.what());
    }
  }

  void OnOK() override {
    _deferred.Resolve(Env().Undefined());
  }

  void OnError(const Napi::Error& err) override {
    _deferred.Reject(err.Value());
  }

  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  llama_context* _ctx;
  std::vector<llama_token> _tokens;
  int32_t _pos;
};

/**
 * AsyncWorker for encode operation (embedding extraction)
 * Unlike DecodeWorker, marks ALL tokens with logits=true
 */
class EncodeWorker : public Napi::AsyncWorker {
public:
  EncodeWorker(Napi::Env env, llama_context* ctx, const std::vector<llama_token>& tokens)
    : AsyncWorker(env), _deferred(env), _ctx(ctx), _tokens(tokens) {}

  void Execute() override {
    try {
      lloyal::decoder::encode(_ctx, _tokens, lloyal::defaults::N_BATCH_PROCESS);
    } catch (const std::exception& e) {
      SetError(e.what());
    }
  }

  void OnOK() override {
    _deferred.Resolve(Env().Undefined());
  }

  void OnError(const Napi::Error& err) override {
    _deferred.Reject(err.Value());
  }

  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  llama_context* _ctx;
  std::vector<llama_token> _tokens;
};

/**
 * AsyncWorker for detokenize operation
 */
class DetokenizeWorker : public Napi::AsyncWorker {
public:
  DetokenizeWorker(Napi::Env env, std::shared_ptr<llama_model> model, const std::vector<llama_token>& tokens)
    : AsyncWorker(env), _deferred(env), _model(model), _tokens(tokens) {}

  void Execute() override {
    // Use convenience overload that auto-extracts vocab
    _result = lloyal::tokenizer::detokenize_batch(_model.get(), _tokens);
  }

  void OnOK() override {
    _deferred.Resolve(Napi::String::New(Env(), _result));
  }

  void OnError(const Napi::Error& err) override {
    _deferred.Reject(err.Value());
  }

  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  std::shared_ptr<llama_model> _model;
  std::vector<llama_token> _tokens;
  std::string _result;
};

/**
 * AsyncWorker for formatChat operation
 */
class FormatChatWorker : public Napi::AsyncWorker {
public:
  FormatChatWorker(Napi::Env env, std::shared_ptr<llama_model> model,
                   const std::string& messagesJson, const std::string& templateOverride)
    : AsyncWorker(env), _deferred(env), _model(model),
      _messagesJson(messagesJson), _templateOverride(templateOverride) {}

  void Execute() override {
    try {
      // Use lloyal::chat_template::format() from liblloyal
      lloyal::chat_template::FormatResult result = lloyal::chat_template::format(
        _model.get(),
        _messagesJson,
        _templateOverride
      );

      // Check if formatting failed completely
      if (result.prompt.empty()) {
        SetError("Chat template formatting failed");
        return;
      }

      _resultPrompt = result.prompt;
      _resultStopTokens = result.additional_stops;
    } catch (const std::exception& e) {
      SetError(e.what());
    }
  }

  void OnOK() override {
    Napi::Env env = Env();

    // Create result object { prompt: string, stopTokens: string[] }
    Napi::Object result = Napi::Object::New(env);
    result.Set("prompt", Napi::String::New(env, _resultPrompt));

    // Convert stopTokens vector to JS array
    Napi::Array stopTokens = Napi::Array::New(env, _resultStopTokens.size());
    for (size_t i = 0; i < _resultStopTokens.size(); i++) {
      stopTokens[i] = Napi::String::New(env, _resultStopTokens[i]);
    }
    result.Set("stopTokens", stopTokens);

    _deferred.Resolve(result);
  }

  void OnError(const Napi::Error& err) override {
    _deferred.Reject(err.Value());
  }

  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  std::shared_ptr<llama_model> _model;
  std::string _messagesJson;
  std::string _templateOverride;
  std::string _resultPrompt;
  std::vector<std::string> _resultStopTokens;
};

// ===== SESSIONCONTEXT IMPLEMENTATION =====

Napi::Object SessionContext::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(env, "SessionContext", {
    // ===== THE GENERATION LOOP =====
    InstanceMethod("decode", &SessionContext::decode),
    InstanceMethod("getLogits", &SessionContext::getLogits),
    InstanceMethod("sample", &SessionContext::sample),
    InstanceMethod("tokenToText", &SessionContext::tokenToText),
    InstanceMethod("isStopToken", &SessionContext::isStopToken),

    // ===== PROMPT PREPARATION =====
    InstanceMethod("tokenize", &SessionContext::tokenize),
    InstanceMethod("detokenize", &SessionContext::detokenize),

    // ===== KV CACHE MANAGEMENT =====
    InstanceMethod("kvCacheSize", &SessionContext::kvCacheSize),
    InstanceMethod("kvCacheRemove", &SessionContext::kvCacheRemove),
    InstanceMethod("kvCacheSave", &SessionContext::kvCacheSave),
    InstanceMethod("kvCacheLoad", &SessionContext::kvCacheLoad),
    InstanceMethod("kvCacheClear", &SessionContext::kvCacheClear),
    InstanceMethod("kvCacheWriteFile", &SessionContext::kvCacheWriteFile),
    InstanceMethod("kvCacheReadFile", &SessionContext::kvCacheReadFile),

    // ===== GRAMMAR-CONSTRAINED GENERATION =====
    InstanceMethod("getTokenScores", &SessionContext::getTokenScores),
    InstanceMethod("initGrammar", &SessionContext::initGrammar),
    InstanceMethod("applyGrammar", &SessionContext::applyGrammar),
    InstanceMethod("acceptToken", &SessionContext::acceptToken),
    InstanceMethod("resetGrammar", &SessionContext::resetGrammar),
    InstanceMethod("freeGrammar", &SessionContext::freeGrammar),

    // ===== HELPERS =====
    InstanceMethod("formatChat", &SessionContext::formatChat),
    InstanceMethod("jsonSchemaToGrammar", &SessionContext::jsonSchemaToGrammar),
    InstanceMethod("validateChatTemplate", &SessionContext::validateChatTemplate),

    // ===== EMBEDDING EXTRACTION =====
    InstanceMethod("encode", &SessionContext::encode),
    InstanceMethod("getEmbeddings", &SessionContext::getEmbeddings),
    InstanceMethod("getEmbeddingDimension", &SessionContext::getEmbeddingDimension),
    InstanceMethod("hasPooling", &SessionContext::hasPooling),

    // ===== NATIVE REFERENCE IMPLEMENTATIONS =====
    InstanceMethod("computeEntropy", &SessionContext::computeEntropy),
    InstanceMethod("greedySample", &SessionContext::greedySample),

    // ===== LIFECYCLE =====
    InstanceMethod("dispose", &SessionContext::dispose),

    // ===== PROPERTIES =====
    InstanceAccessor("vocabSize", &SessionContext::getVocabSize, nullptr),
    InstanceAccessor("memorySize", &SessionContext::getMemorySize, nullptr)
  });

  exports.Set("SessionContext", func);
  return exports;
}

SessionContext::SessionContext(const Napi::CallbackInfo& info)
  : Napi::ObjectWrap<SessionContext>(info) {
  // Constructor is called by CreateContext factory function
  // Model and context are set externally
}

SessionContext::~SessionContext() {
  if (!_disposed) {
    // Free grammar sampler first (pattern matches HybridSessionContext.cpp:72)
    if (_grammarSampler) {
      llama_sampler_free(_grammarSampler);
      _grammarSampler = nullptr;
    }

    // Free context (depends on model)
    if (_context) {
      llama_free(_context);
      _context = nullptr;
    }
    // _model freed automatically via shared_ptr
  }
}

void SessionContext::initializeContext(
  std::shared_ptr<llama_model> model,
  llama_context* context
) {
  _model = std::move(model);
  _context = context;

  std::cerr << "[SessionContext::initializeContext] Initialized:" << std::endl;
  std::cerr << "  Model ptr: " << static_cast<void*>(_model.get()) << std::endl;
  std::cerr << "  Context ptr: " << static_cast<void*>(_context) << std::endl;
  std::cerr << "  Shared refcount: " << _model.use_count() << std::endl;
}

Napi::Value SessionContext::getLogits(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (!_context) {
    throw Napi::Error::New(env, "Context not initialized");
  }

  // Get raw logits pointer (zero-copy)
  float* logits = llama_get_logits_ith(_context, -1);
  if (!logits) {
    throw Napi::Error::New(env, "Failed to get logits");
  }

  // Use model overload for vocab_size
  const int n_vocab = lloyal::tokenizer::vocab_size(_model.get());

  // Create Float32Array wrapping the logits (zero-copy!)
  // WARNING: This is only valid until next decode() call
  return Napi::Float32Array::New(
    env,
    n_vocab,
    Napi::ArrayBuffer::New(env, logits, n_vocab * sizeof(float)),
    0
  );
}

Napi::Value SessionContext::decode(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2 || !info[0].IsArray() || !info[1].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected (tokens: number[], position: number)");
  }

  // Extract tokens
  Napi::Array jsTokens = info[0].As<Napi::Array>();
  std::vector<llama_token> tokens;
  tokens.reserve(jsTokens.Length());
  for (uint32_t i = 0; i < jsTokens.Length(); i++) {
    Napi::Value val = jsTokens[i];
    if (!val.IsNumber()) {
      throw Napi::TypeError::New(env, "Token array must contain only numbers");
    }
    tokens.push_back(static_cast<llama_token>(val.As<Napi::Number>().Int32Value()));
  }

  int32_t position = info[1].As<Napi::Number>().Int32Value();

  // Run async
  auto* worker = new DecodeWorker(env, _context, tokens, position);
  worker->Queue();
  return worker->GetPromise();
}

Napi::Value SessionContext::tokenize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Expected (text: string)");
  }

  std::string text = info[0].As<Napi::String>().Utf8Value();

  // Run async
  auto* worker = new TokenizeWorker(env, _model, text);
  worker->Queue();
  return worker->GetPromise();
}

Napi::Value SessionContext::detokenize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsArray()) {
    throw Napi::TypeError::New(env, "Expected (tokens: number[])");
  }

  // Extract tokens
  Napi::Array jsTokens = info[0].As<Napi::Array>();
  std::vector<llama_token> tokens;
  tokens.reserve(jsTokens.Length());
  for (uint32_t i = 0; i < jsTokens.Length(); i++) {
    Napi::Value val = jsTokens[i];
    if (!val.IsNumber()) {
      throw Napi::TypeError::New(env, "Token array must contain only numbers");
    }
    tokens.push_back(static_cast<llama_token>(val.As<Napi::Number>().Int32Value()));
  }

  // Run async
  auto* worker = new DetokenizeWorker(env, _model, tokens);
  worker->Queue();
  return worker->GetPromise();
}

Napi::Value SessionContext::computeEntropy(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (!_context) {
    throw Napi::Error::New(env, "Context not initialized");
  }

  // Get logits
  float* logits = llama_get_logits_ith(_context, -1);
  if (!logits) {
    throw Napi::Error::New(env, "Failed to get logits");
  }

  // Use model overload for vocab_size
  const int n_vocab = lloyal::tokenizer::vocab_size(_model.get());

  // Compute entropy using log-sum-exp (numerically stable)
  // This is the native reference implementation for testing

  // Find max logit
  double max_logit = logits[0];
  for (int i = 1; i < n_vocab; ++i) {
    if (std::isfinite(logits[i]) && logits[i] > max_logit) {
      max_logit = logits[i];
    }
  }

  // Compute sum of exp(logit - max)
  double sum_exp = 0.0;
  for (int i = 0; i < n_vocab; ++i) {
    if (std::isfinite(logits[i])) {
      sum_exp += std::exp(logits[i] - max_logit);
    }
  }

  if (sum_exp == 0.0) {
    return Napi::Number::New(env, INFINITY);
  }

  double log_sum = max_logit + std::log(sum_exp);

  // H = -Σ p_i * log(p_i)
  double entropy = 0.0;
  for (int i = 0; i < n_vocab; ++i) {
    if (std::isfinite(logits[i])) {
      double p = std::exp(logits[i] - log_sum);
      if (p > 0) {
        entropy += -p * std::log(p);
      }
    }
  }

  return Napi::Number::New(env, entropy);
}

Napi::Value SessionContext::greedySample(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (!_context) {
    throw Napi::Error::New(env, "Context not initialized");
  }

  // Use liblloyal greedy sampler with model overload
  llama_token token = lloyal::sampler::greedy(_context, _model.get());

  return Napi::Number::New(env, static_cast<double>(token));
}

Napi::Value SessionContext::tokenToText(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected token ID (number)");
  }

  llama_token token = static_cast<llama_token>(info[0].As<Napi::Number>().Int32Value());

  // Use lloyal detokenize with model overload (optimized for single tokens)
  std::string text = lloyal::tokenizer::detokenize(_model.get(), token, true);

  return Napi::String::New(env, text);
}

// ===== EMBEDDING EXTRACTION =====

Napi::Value SessionContext::encode(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsArray()) {
    throw Napi::TypeError::New(env, "Expected (tokens: number[])");
  }

  // Extract tokens
  Napi::Array jsTokens = info[0].As<Napi::Array>();
  std::vector<llama_token> tokens;
  tokens.reserve(jsTokens.Length());
  for (uint32_t i = 0; i < jsTokens.Length(); i++) {
    Napi::Value val = jsTokens[i];
    if (!val.IsNumber()) {
      throw Napi::TypeError::New(env, "Token array must contain only numbers");
    }
    tokens.push_back(static_cast<llama_token>(val.As<Napi::Number>().Int32Value()));
  }

  // Run async
  auto* worker = new EncodeWorker(env, _context, tokens);
  worker->Queue();
  return worker->GetPromise();
}

Napi::Value SessionContext::getEmbeddings(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (!_context) {
    throw Napi::Error::New(env, "Context not initialized");
  }

  // Check for optional normalize parameter (default true = L2)
  bool normalize = true;
  if (info.Length() > 0 && info[0].IsBoolean()) {
    normalize = info[0].As<Napi::Boolean>().Value();
  }

  try {
    // Use liblloyal embedding::get
    auto normMode = normalize ? lloyal::embedding::Normalize::L2 : lloyal::embedding::Normalize::None;
    std::vector<float> embeddings = lloyal::embedding::get(_context, normMode);

    // Copy to Float32Array
    Napi::Float32Array result = Napi::Float32Array::New(env, embeddings.size());
    std::memcpy(result.Data(), embeddings.data(), embeddings.size() * sizeof(float));

    return result;
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, e.what());
  }
}

Napi::Value SessionContext::getEmbeddingDimension(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (!_model) {
    throw Napi::Error::New(env, "Model not initialized");
  }

  int32_t dim = lloyal::embedding::dimension(_model.get());
  return Napi::Number::New(env, static_cast<double>(dim));
}

Napi::Value SessionContext::hasPooling(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (!_context) {
    throw Napi::Error::New(env, "Context not initialized");
  }

  bool hasPool = lloyal::embedding::has_pooling(_context);
  return Napi::Boolean::New(env, hasPool);
}

Napi::Value SessionContext::isStopToken(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected token ID (number)");
  }

  llama_token token = static_cast<llama_token>(info[0].As<Napi::Number>().Int32Value());

  // Check if token is end-of-generation (EOS, EOT, etc.) using model overload
  bool isEog = lloyal::tokenizer::is_eog(_model.get(), token);

  return Napi::Boolean::New(env, isEog);
}

Napi::Value SessionContext::formatChat(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Expected (messagesJson: string, templateOverride?: string)");
  }

  std::string messagesJson = info[0].As<Napi::String>().Utf8Value();
  std::string templateOverride = "";

  if (info.Length() >= 2 && info[1].IsString()) {
    templateOverride = info[1].As<Napi::String>().Utf8Value();
  }

  // Run async
  auto* worker = new FormatChatWorker(env, _model, messagesJson, templateOverride);
  worker->Queue();
  return worker->GetPromise();
}

Napi::Value SessionContext::kvCacheSize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (!_context) {
    throw Napi::Error::New(env, "Context not initialized");
  }

  // Extract optional sequenceId parameter (defaults to 0)
  // Pattern matches HybridSessionContext.cpp:573-583
  double sequenceId = 0.0;
  if (info.Length() > 0 && info[0].IsNumber()) {
    sequenceId = info[0].As<Napi::Number>().DoubleValue();
  }

  // Get max position in KV cache for specified sequence
  // Returns -1 if empty (not 0!)
  llama_pos max_pos = lloyal::kv::pos_max(_context, toSeqId(sequenceId));

  return Napi::Number::New(env, static_cast<double>(max_pos));
}

Napi::Value SessionContext::sample(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (!_context) {
    throw Napi::Error::New(env, "Context not initialized");
  }

  llama_token next_token;

  // Use greedy if no params, otherwise use full sampling from liblloyal
  // Pattern matches HybridSessionContext.cpp:160-192
  if (info.Length() == 0 || !info[0].IsObject()) {
    // No params - use greedy sampling with model overload
    next_token = lloyal::sampler::greedy(_context, _model.get());
  } else {
    // Use adapter to convert JS params → liblloyal-compatible structure
    LloyalSamplingParams params = adaptSamplingParamsFromJS(info[0].As<Napi::Object>());

    // Use liblloyal sample_with_params with model overload
    // Note: Advanced params (mirostat, dry, xtc) not yet supported in liblloyal
    next_token = lloyal::sampler::sample_with_params(_context, _model.get(), params, _grammarSampler);
  }

  // Accept token to advance grammar parser state (if grammar active)
  if (_grammarSampler) {
    llama_sampler_accept(_grammarSampler, next_token);
  }

  return Napi::Number::New(env, static_cast<double>(next_token));
}

Napi::Value SessionContext::dispose(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (!_disposed) {
    // Free grammar sampler first
    if (_grammarSampler) {
      llama_sampler_free(_grammarSampler);
      _grammarSampler = nullptr;
    }

    // Free context
    if (_context) {
      llama_free(_context);
      _context = nullptr;
    }

    // Reset model
    _model.reset();
    _disposed = true;
  }

  return env.Undefined();
}

Napi::Value SessionContext::getVocabSize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Use model overload
  return Napi::Number::New(env, static_cast<double>(lloyal::tokenizer::vocab_size(_model.get())));
}

// ===== GRAMMAR-CONSTRAINED GENERATION =====
// Pattern matches HybridSessionContext.cpp:383-546
// All methods are SYNC (no AsyncWorker) per SessionContext.nitro.ts

Napi::Value SessionContext::getTokenScores(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (!_context) {
    throw Napi::Error::New(env, "Context not initialized");
  }

  // Get raw logits pointer from llama.cpp (last-step logits, index -1)
  // Returns mutable float* - we need to modify logits for grammar constraints
  float* logits = llama_get_logits_ith(_context, -1);
  if (!logits) {
    throw Napi::Error::New(env, "Failed to get logits (ensure decode had logits=true)");
  }

  // Get vocabulary size using model overload
  const int n_vocab = lloyal::tokenizer::vocab_size(_model.get());
  if (n_vocab <= 0) {
    throw Napi::Error::New(env, "Invalid vocabulary size");
  }

  // Create Buffer wrapping the logits (zero-copy!)
  // CRITICAL: Logits are valid only until next llama_decode() call
  size_t byte_size = n_vocab * sizeof(float);

  // Note: Using external Buffer - no copy, points directly to llama.cpp memory
  return Napi::Buffer<float>::New(
    env,
    logits,
    n_vocab,
    [](Napi::Env /*env*/, float* /*data*/) {
      // No-op finalizer: llama.cpp owns and manages this memory
    }
  );
}

Napi::Value SessionContext::initGrammar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Expected (grammarStr: string)");
  }

  std::string grammarStr = info[0].As<Napi::String>().Utf8Value();

  ensureNotDisposed();
  if (!_context) {
    throw Napi::Error::New(env, "Context not initialized");
  }

  // Reuse existing sampler if grammar unchanged (just reset parser state)
  if (_grammarSampler && _currentGrammar == grammarStr) {
    llama_sampler_reset(_grammarSampler);
    return env.Undefined();
  }

  // Free old sampler if grammar changed
  if (_grammarSampler) {
    llama_sampler_free(_grammarSampler);
    _grammarSampler = nullptr;
  }

  // Create new grammar sampler using liblloyal wrapper
  _grammarSampler = lloyal::grammar::init_sampler(_model.get(), grammarStr);
  if (!_grammarSampler) {
    throw Napi::Error::New(env, "Failed to initialize grammar sampler - grammar may be invalid");
  }

  _currentGrammar = grammarStr;
  return env.Undefined();
}

Napi::Value SessionContext::applyGrammar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsBuffer()) {
    throw Napi::TypeError::New(env, "Expected (scoresBuffer: Buffer)");
  }

  if (!_grammarSampler) {
    throw Napi::Error::New(env, "Grammar not initialized - call initGrammar() first");
  }

  if (!_context) {
    throw Napi::Error::New(env, "Context not initialized");
  }

  // Use model overload for vocab_size
  const int n_vocab = lloyal::tokenizer::vocab_size(_model.get());

  // Get mutable access to logits
  Napi::Buffer<float> scoresBuffer = info[0].As<Napi::Buffer<float>>();
  float* logits = scoresBuffer.Data();
  if (!logits) {
    throw Napi::Error::New(env, "Invalid scores buffer");
  }

  // Build token data array from logits
  std::vector<llama_token_data> candidates(n_vocab);
  for (int i = 0; i < n_vocab; i++) {
    candidates[i] = llama_token_data{
      static_cast<llama_token>(i),
      logits[i],
      0.0f
    };
  }

  llama_token_data_array arr = {
    candidates.data(),
    static_cast<size_t>(n_vocab),
    -1,  // sorted = -1 (unsorted)
    false
  };

  // Apply grammar constraint (modifies candidates in-place)
  llama_sampler_apply(_grammarSampler, &arr);

  // Write modified logits back to buffer
  for (int i = 0; i < n_vocab; i++) {
    logits[i] = candidates[i].logit;
  }

  return env.Undefined();
}

Napi::Value SessionContext::acceptToken(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected (tokenId: number)");
  }

  llama_token token = static_cast<llama_token>(info[0].As<Napi::Number>().Int32Value());

  // Advance grammar parser state (if grammar active)
  if (_grammarSampler) {
    llama_sampler_accept(_grammarSampler, token);
  }

  return env.Undefined();
}

Napi::Value SessionContext::resetGrammar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (_grammarSampler) {
    llama_sampler_reset(_grammarSampler);
  }

  return env.Undefined();
}

Napi::Value SessionContext::freeGrammar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (_grammarSampler) {
    llama_sampler_free(_grammarSampler);
    _grammarSampler = nullptr;
    _currentGrammar.clear();
  }

  return env.Undefined();
}

// ===== HELPER METHODS =====
// Pattern matches HybridSessionContext.cpp:103-106, 365-379

Napi::Value SessionContext::getMemorySize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (!_context || !_model) {
    return Napi::Number::New(env, 0.0);
  }

  // Return model size as memory usage estimate
  // Pattern matches HybridSessionContext.cpp:103-106 (placeholder approach)
  // More precise KV cache tracking would require llama.cpp API additions
  size_t modelSize = 0;
  if (_model) {
    modelSize = llama_model_size(_model.get());
  }

  return Napi::Number::New(env, static_cast<double>(modelSize));
}

Napi::Value SessionContext::jsonSchemaToGrammar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Expected (schemaJson: string)");
  }

  std::string schemaJson = info[0].As<Napi::String>().Utf8Value();

  // Use liblloyal (handles parsing, conversion, and error logging)
  // Pattern matches HybridSessionContext.cpp:374-379
  std::string grammar = lloyal::grammar::from_json_schema(schemaJson);

  return Napi::String::New(env, grammar);
}

Napi::Value SessionContext::validateChatTemplate(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Expected (templateString: string)");
  }

  std::string templateString = info[0].As<Napi::String>().Utf8Value();

  // Create AsyncWorker for validation
  class ValidateChatTemplateWorker : public Napi::AsyncWorker {
  public:
    ValidateChatTemplateWorker(Napi::Env env, const std::string& templateStr)
      : AsyncWorker(env), _deferred(env), _templateString(templateStr) {}

    void Execute() override {
      // Use lloyal::chat_template from liblloyal (handles error logging)
      // Pattern matches HybridSessionContext.cpp:365-372
      _result = lloyal::chat_template::validate(_templateString);
    }

    void OnOK() override {
      _deferred.Resolve(Napi::Boolean::New(Env(), _result));
    }

    void OnError(const Napi::Error& err) override {
      _deferred.Reject(err.Value());
    }

    Napi::Promise GetPromise() { return _deferred.Promise(); }

  private:
    Napi::Promise::Deferred _deferred;
    std::string _templateString;
    bool _result = false;
  };

  auto* worker = new ValidateChatTemplateWorker(env, templateString);
  worker->Queue();
  return worker->GetPromise();
}

// ===== KV CACHE OPERATIONS =====
// Pattern matches HybridSessionContext.cpp:550-642

Napi::Value SessionContext::kvCacheRemove(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 3 || !info[0].IsNumber() || !info[1].IsNumber() || !info[2].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected (sequenceId: number, start: number, end: number)");
  }

  double sequenceId = info[0].As<Napi::Number>().DoubleValue();
  double start = info[1].As<Napi::Number>().DoubleValue();
  double end = info[2].As<Napi::Number>().DoubleValue();

  auto* worker = new KVCacheRemoveWorker(env, _context, sequenceId, start, end);
  worker->Queue();
  return worker->GetPromise();
}

Napi::Value SessionContext::kvCacheSave(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Extract optional sequenceId parameter (defaults to 0)
  double sequenceId = 0.0;
  if (info.Length() > 0 && info[0].IsNumber()) {
    sequenceId = info[0].As<Napi::Number>().DoubleValue();
  }

  auto* worker = new KVCacheSaveWorker(env, _context, sequenceId);
  worker->Queue();
  return worker->GetPromise();
}

Napi::Value SessionContext::kvCacheLoad(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsBuffer()) {
    throw Napi::TypeError::New(env, "Expected (sequenceId: number, state: Buffer)");
  }

  double sequenceId = info[0].As<Napi::Number>().DoubleValue();
  Napi::Buffer<uint8_t> stateBuffer = info[1].As<Napi::Buffer<uint8_t>>();

  auto* worker = new KVCacheLoadWorker(env, _context, sequenceId, stateBuffer.Data(), stateBuffer.Length());
  worker->Queue();
  return worker->GetPromise();
}

Napi::Value SessionContext::kvCacheClear(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  auto* worker = new KVCacheClearWorker(env, _context);
  worker->Queue();
  return worker->GetPromise();
}

Napi::Value SessionContext::kvCacheWriteFile(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Args: sequenceId, filepath, tokens
  if (info.Length() < 3 || !info[0].IsNumber() || !info[1].IsString() || !info[2].IsArray()) {
    throw Napi::TypeError::New(env, "Expected (sequenceId: number, filepath: string, tokens: number[])");
  }

  llama_seq_id seq = static_cast<llama_seq_id>(info[0].As<Napi::Number>().Int32Value());
  std::string filepath = info[1].As<Napi::String>().Utf8Value();

  // Extract tokens
  Napi::Array jsTokens = info[2].As<Napi::Array>();
  std::vector<llama_token> tokens;
  tokens.reserve(jsTokens.Length());
  for (uint32_t i = 0; i < jsTokens.Length(); i++) {
    Napi::Value val = jsTokens.Get(i);
    tokens.push_back(static_cast<llama_token>(val.As<Napi::Number>().Int32Value()));
  }

  auto* worker = new KVCacheWriteFileWorker(env, _context, seq, filepath, std::move(tokens));
  worker->Queue();
  return worker->GetPromise();
}

Napi::Value SessionContext::kvCacheReadFile(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Args: sequenceId, filepath
  if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsString()) {
    throw Napi::TypeError::New(env, "Expected (sequenceId: number, filepath: string)");
  }

  llama_seq_id seq = static_cast<llama_seq_id>(info[0].As<Napi::Number>().Int32Value());
  std::string filepath = info[1].As<Napi::String>().Utf8Value();

  auto* worker = new KVCacheReadFileWorker(env, _context, seq, filepath);
  worker->Queue();
  return worker->GetPromise();
}

// ===== FACTORY FUNCTION =====

Napi::Value CreateContext(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsObject()) {
    throw Napi::TypeError::New(env, "Expected options object");
  }

  Napi::Object options = info[0].As<Napi::Object>();

  // Extract modelPath (required)
  if (!options.Has("modelPath") || !options.Get("modelPath").IsString()) {
    throw Napi::TypeError::New(env, "options.modelPath is required");
  }
  std::string modelPath = options.Get("modelPath").As<Napi::String>().Utf8Value();

  // Extract nCtx (optional, use liblloyal defaults)
  int32_t nCtx = lloyal::defaults::N_CTX;
  if (options.Has("nCtx") && options.Get("nCtx").IsNumber()) {
    nCtx = options.Get("nCtx").As<Napi::Number>().Int32Value();
  }

  // Extract nThreads (optional, 0 = auto)
  int32_t nThreads = 0;  // 0 = llama.cpp auto-detects
  if (options.Has("nThreads") && options.Get("nThreads").IsNumber()) {
    nThreads = options.Get("nThreads").As<Napi::Number>().Int32Value();
  }

  // Extract embeddings mode (optional, default false)
  bool embeddingsMode = false;
  if (options.Has("embeddings") && options.Get("embeddings").IsBoolean()) {
    embeddingsMode = options.Get("embeddings").As<Napi::Boolean>().Value();
  }

  // Extract pooling type (optional, default MEAN for embeddings, NONE otherwise)
  // 0 = NONE, 1 = MEAN, 2 = CLS, 3 = LAST
  int32_t poolingType = embeddingsMode ? 1 : 0;  // Default to MEAN for embedding contexts
  if (options.Has("poolingType") && options.Get("poolingType").IsNumber()) {
    poolingType = options.Get("poolingType").As<Napi::Number>().Int32Value();
  }

  // Ensure llama backend is initialized on main thread (thread-safe, once)
  BackendManager::ensureInitialized();

  // Normalize and validate path BEFORE queuing async work
  std::string fsPath = margelo::nitro::nitrollama::FileSystem::normalizePath(modelPath);
  if (fsPath != modelPath) {
    std::cout << "[CreateContext] Normalized " << modelPath << " → " << fsPath << std::endl;
  }

  if (!margelo::nitro::nitrollama::FileSystem::exists(fsPath)) {
    std::cout << "[CreateContext] File does not exist: " << fsPath << std::endl;
    throw Napi::Error::New(env, "Model file not found: " + fsPath);
  }

  size_t fileSize = margelo::nitro::nitrollama::FileSystem::getSize(fsPath);
  std::cout << "[CreateContext] File validated: " << fsPath << " (" << fileSize << " bytes)" << std::endl;

  // Load model on main thread
  // Note: With XCFramework build, this works reliably on main thread
  // (async loading was failing with CMake build due to binary incompatibility)
  std::cout << "[CreateContext] Loading model from XCFramework..." << std::endl;

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;

  std::cout << "[CreateContext] Acquiring from ModelRegistry..." << std::endl;
  auto sharedModel = lloyal::ModelRegistry::acquire(fsPath, model_params);

  if (!sharedModel) {
    throw Napi::Error::New(env, "Failed to load model from " + fsPath);
  }

  std::cout << "[CreateContext] Model loaded (refcount: " << sharedModel.use_count() << ")" << std::endl;

  // Create context
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = static_cast<uint32_t>(nCtx);
  ctx_params.n_batch = lloyal::defaults::N_BATCH_INIT;
  ctx_params.n_ubatch = lloyal::defaults::N_BATCH_INIT;
  ctx_params.n_threads = static_cast<uint32_t>(nThreads);

  // Apply embedding-specific params
  ctx_params.embeddings = embeddingsMode;
  ctx_params.pooling_type = static_cast<enum llama_pooling_type>(poolingType);

  std::cout << "[CreateContext] Creating context (embeddings=" << embeddingsMode
            << ", pooling=" << poolingType << ")..." << std::endl;
  llama_context* ctx = llama_init_from_model(sharedModel.get(), ctx_params);

  if (!ctx) {
    throw Napi::Error::New(env, "Failed to create context");
  }

  std::cout << "[CreateContext] Context created successfully" << std::endl;

  // Create SessionContext instance
  Napi::Function ctor = env.GetInstanceData<Napi::FunctionReference>()->Value();
  Napi::Object instance = ctor.New({});
  SessionContext* obj = SessionContext::Unwrap(instance);

  // Initialize
  obj->initializeContext(std::move(sharedModel), ctx);

  std::cout << "[CreateContext] SessionContext initialized" << std::endl;
  return instance;
}

} // namespace liblloyal_node
