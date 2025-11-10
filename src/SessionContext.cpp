#include "SessionContext.hpp"
#include "BackendManager.hpp"
#include "FileSystem.h"
#include <lloyal/decoder.hpp>
#include <lloyal/sampler.hpp>
#include <lloyal/tokenizer.hpp>
#include <lloyal/common.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/chat_template.hpp>
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
    InstanceMethod("getLogits", &SessionContext::getLogits),
    InstanceMethod("decode", &SessionContext::decode),
    InstanceMethod("tokenize", &SessionContext::tokenize),
    InstanceMethod("detokenize", &SessionContext::detokenize),
    InstanceMethod("tokenToText", &SessionContext::tokenToText),
    InstanceMethod("isStopToken", &SessionContext::isStopToken),
    InstanceMethod("formatChat", &SessionContext::formatChat),
    InstanceMethod("sample", &SessionContext::sample),
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

Napi::Value SessionContext::tokenToText(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected token ID (number)");
  }

  llama_token token = static_cast<llama_token>(info[0].As<Napi::Number>().Int32Value());
  const llama_vocab* vocab = getVocabOrThrow();

  // Use lloyal detokenize (optimized for single tokens)
  std::string text = lloyal::tokenizer::detokenize(vocab, token, true);

  return Napi::String::New(env, text);
}

Napi::Value SessionContext::isStopToken(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected token ID (number)");
  }

  llama_token token = static_cast<llama_token>(info[0].As<Napi::Number>().Int32Value());
  const llama_vocab* vocab = getVocabOrThrow();

  // Check if token is end-of-generation (EOS, EOT, etc.)
  bool isEog = llama_vocab_is_eog(vocab, token);

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

Napi::Value SessionContext::sample(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (!_context) {
    throw Napi::Error::New(env, "Context not initialized");
  }

  const llama_vocab* vocab = getVocabOrThrow();

  // Simple struct conforming to SamplingParamsLike concept
  struct SimpleSamplingParams {
    float temperature = 0.8f;
    int32_t top_k = 40;
    float top_p = 0.95f;
    float typical_p = 1.0f;
    float min_p = 0.05f;
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    int32_t penalty_last_n = 64;
    uint32_t seed = static_cast<uint32_t>(std::time(nullptr));
  };

  SimpleSamplingParams params;

  // Parse parameters from JS object
  if (info.Length() > 0 && info[0].IsObject()) {
    Napi::Object paramsObj = info[0].As<Napi::Object>();

    if (paramsObj.Has("temperature") && paramsObj.Get("temperature").IsNumber()) {
      params.temperature = paramsObj.Get("temperature").As<Napi::Number>().FloatValue();
    }
    if (paramsObj.Has("topK") && paramsObj.Get("topK").IsNumber()) {
      params.top_k = paramsObj.Get("topK").As<Napi::Number>().Int32Value();
    }
    if (paramsObj.Has("topP") && paramsObj.Get("topP").IsNumber()) {
      params.top_p = paramsObj.Get("topP").As<Napi::Number>().FloatValue();
    }
    if (paramsObj.Has("minP") && paramsObj.Get("minP").IsNumber()) {
      params.min_p = paramsObj.Get("minP").As<Napi::Number>().FloatValue();
    }
    if (paramsObj.Has("seed") && paramsObj.Get("seed").IsNumber()) {
      params.seed = static_cast<uint32_t>(paramsObj.Get("seed").As<Napi::Number>().Int64Value());
    }

    // Penalty params
    if (paramsObj.Has("penalties") && paramsObj.Get("penalties").IsObject()) {
      Napi::Object penalties = paramsObj.Get("penalties").As<Napi::Object>();

      if (penalties.Has("repeat") && penalties.Get("repeat").IsNumber()) {
        params.penalty_repeat = penalties.Get("repeat").As<Napi::Number>().FloatValue();
      }
      if (penalties.Has("frequency") && penalties.Get("frequency").IsNumber()) {
        params.penalty_freq = penalties.Get("frequency").As<Napi::Number>().FloatValue();
      }
      if (penalties.Has("presence") && penalties.Get("presence").IsNumber()) {
        params.penalty_present = penalties.Get("presence").As<Napi::Number>().FloatValue();
      }
      if (penalties.Has("lastN") && penalties.Get("lastN").IsNumber()) {
        params.penalty_last_n = penalties.Get("lastN").As<Napi::Number>().Int32Value();
      }
    }
  }

  // Use liblloyal sample_with_params
  llama_token token = lloyal::sampler::sample_with_params(_context, vocab, params);

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

  std::cout << "[CreateContext] Creating context..." << std::endl;
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
