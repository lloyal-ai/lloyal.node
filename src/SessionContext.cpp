#include "SessionContext.hpp"
#include "BackendManager.hpp"
#include <lloyal/decoder.hpp>
#include <lloyal/sampler.hpp>
#include <lloyal/tokenizer.hpp>
#include <lloyal/common.hpp>
#include <lloyal/model_registry.hpp>
#include <cmath>

namespace liblloyal_node {

// ===== ASYNC WORKER CLASSES =====

/**
 * AsyncWorker for tokenize operation
 */
class TokenizeWorker : public Napi::AsyncWorker {
public:
  TokenizeWorker(Napi::Env env, std::shared_ptr<llama_model> model, const std::string& text)
    : AsyncWorker(env), _deferred(env), _model(model), _text(text) {}

  void Execute() override {
    const llama_vocab* vocab = lloyal::tokenizer::get_vocab(_model.get());
    if (!vocab) {
      SetError("Failed to get vocabulary");
      return;
    }

    bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = lloyal::tokenizer::tokenize(vocab, _text, add_bos, true);

    if (tokens.empty()) {
      SetError("Tokenization failed");
      return;
    }

    _result = std::move(tokens);
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
 * AsyncWorker for detokenize operation
 */
class DetokenizeWorker : public Napi::AsyncWorker {
public:
  DetokenizeWorker(Napi::Env env, std::shared_ptr<llama_model> model, const std::vector<llama_token>& tokens)
    : AsyncWorker(env), _deferred(env), _model(model), _tokens(tokens) {}

  void Execute() override {
    const llama_vocab* vocab = lloyal::tokenizer::get_vocab(_model.get());
    if (!vocab) {
      SetError("Failed to get vocabulary");
      return;
    }

    _result = lloyal::tokenizer::detokenize_batch(
      vocab,
      _tokens.data(),
      static_cast<int32_t>(_tokens.size()),
      false,  // remove_special
      false   // unparse_special
    );
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

// ===== SESSIONCONTEXT IMPLEMENTATION =====

Napi::Object SessionContext::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(env, "SessionContext", {
    InstanceMethod("getLogits", &SessionContext::getLogits),
    InstanceMethod("decode", &SessionContext::decode),
    InstanceMethod("tokenize", &SessionContext::tokenize),
    InstanceMethod("detokenize", &SessionContext::detokenize),
    InstanceMethod("computeEntropy", &SessionContext::computeEntropy),
    InstanceMethod("greedySample", &SessionContext::greedySample),
    InstanceMethod("dispose", &SessionContext::dispose),
    InstanceAccessor("vocabSize", &SessionContext::getVocabSize, nullptr)
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
    if (_context) {
      llama_free(_context);
      _context = nullptr;
    }
    // _model freed automatically via shared_ptr
  }
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

  const llama_vocab* vocab = getVocabOrThrow();
  const int n_vocab = llama_vocab_n_tokens(vocab);

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

  const llama_vocab* vocab = getVocabOrThrow();
  const int n_vocab = llama_vocab_n_tokens(vocab);

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

  const llama_vocab* vocab = getVocabOrThrow();

  // Use liblloyal greedy sampler
  llama_token token = lloyal::sampler::greedy(_context, vocab);

  return Napi::Number::New(env, static_cast<double>(token));
}

Napi::Value SessionContext::dispose(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (!_disposed) {
    if (_context) {
      llama_free(_context);
      _context = nullptr;
    }
    _model.reset();
    _disposed = true;
  }

  return env.Undefined();
}

Napi::Value SessionContext::getVocabSize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  const llama_vocab* vocab = getVocabOrThrow();
  return Napi::Number::New(env, static_cast<double>(lloyal::tokenizer::vocab_size(vocab)));
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

  // Ensure llama backend is initialized (thread-safe, once)
  BackendManager::ensureInitialized();

  // Acquire model from registry (enables caching/sharing)
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;  // CPU only for testing

  auto sharedModel = lloyal::ModelRegistry::acquire(modelPath, model_params);
  if (!sharedModel) {
    throw Napi::Error::New(env, "Failed to load model from " + modelPath);
  }

  // Create context
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = static_cast<uint32_t>(nCtx);
  ctx_params.n_batch = lloyal::defaults::N_BATCH_INIT;
  ctx_params.n_ubatch = lloyal::defaults::N_BATCH_INIT;
  ctx_params.n_threads = static_cast<uint32_t>(nThreads);

  llama_context* ctx = llama_init_from_model(sharedModel.get(), ctx_params);
  if (!ctx) {
    // sharedModel freed automatically by shared_ptr
    throw Napi::Error::New(env, "Failed to create context");
  }

  // Create SessionContext instance
  Napi::Function ctor = env.GetInstanceData<Napi::FunctionReference>()->Value();
  Napi::Object instance = ctor.New({});
  SessionContext* obj = SessionContext::Unwrap(instance);

  // Initialize with shared model and exclusive context
  obj->_model = sharedModel;  // Shared ownership via ModelRegistry
  obj->_context = ctx;        // Exclusive ownership

  return instance;
}

} // namespace liblloyal_node
