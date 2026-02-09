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
#include <lloyal/logits.hpp>
#include <lloyal/metrics.hpp>
#include <cmath>

namespace liblloyal_node {

// ===== ADAPTER FOR LIBLLOYAL COMPATIBILITY =====
//
// LloyalSamplingParams is now defined in SessionContext.hpp
// This function converts JS object params to that structure.

// Convert JS object params → liblloyal-compatible structure
// Currently supports basic parameters (temperature, topK, topP, minP, seed)
// and penalty group (repeat, frequency, presence, lastN).
// Advanced parameters (mirostat, dry, xtc, typical_p) to be added as liblloyal adds support.
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

  // Future: Extract from advanced group when liblloyal adds support
  // - typical_p (Locally Typical Sampling)
  // - mirostat (Mirostat 1.0/2.0)
  // - dry (Don't Repeat Yourself)
  // - xtc (Extended Temperature Scaling)

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
 * AsyncWorker for clearAndReseed operation (StreamingLLM)
 * Uses lloyal::kv::clear_and_reseed() - the validated API
 */
class ClearAndReseedWorker : public Napi::AsyncWorker {
public:
  ClearAndReseedWorker(Napi::Env env, llama_context* ctx,
                       std::vector<llama_token> sinks,
                       std::vector<llama_token> tail,
                       int32_t n_batch)
    : AsyncWorker(env), _deferred(env), _ctx(ctx),
      _sinks(std::move(sinks)), _tail(std::move(tail)), _n_batch(n_batch) {}

  void Execute() override {
    // Use lloyal::kv::clear_and_reseed() - handles clear+decode atomically
    lloyal::kv::clear_and_reseed(_ctx, _sinks, _tail, _n_batch);
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
  std::vector<llama_token> _sinks;
  std::vector<llama_token> _tail;
  int32_t _n_batch;
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
  TokenizeWorker(Napi::Env env, std::shared_ptr<llama_model> model,
                 const std::string& text, bool addSpecial, bool addSpecialOverridden)
    : AsyncWorker(env), _deferred(env), _model(model), _text(text),
      _addSpecial(addSpecial), _addSpecialOverridden(addSpecialOverridden) {}

  void Execute() override {
    if (_addSpecialOverridden) {
      const llama_vocab* vocab = llama_model_get_vocab(_model.get());
      _result = lloyal::tokenizer::tokenize(vocab, _text, _addSpecial, true);
    } else {
      // Use convenience overload that auto-extracts vocab and handles add_bos
      _result = lloyal::tokenizer::tokenize(_model.get(), _text);
    }
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
  bool _addSpecial;
  bool _addSpecialOverridden;
  std::vector<llama_token> _result;
};

/**
 * AsyncWorker for decode operation
 */
class DecodeWorker : public Napi::AsyncWorker {
public:
  DecodeWorker(Napi::Env env, llama_context* ctx, const std::vector<llama_token>& tokens,
               int32_t pos, llama_seq_id seqId, int32_t nBatch)
    : AsyncWorker(env), _deferred(env), _ctx(ctx), _tokens(tokens), _pos(pos), _seqId(seqId), _nBatch(nBatch) {}

  void Execute() override {
    try {
      lloyal::decoder::decode_tokens(_ctx, _tokens, _pos, _nBatch, _seqId);
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
  llama_seq_id _seqId;
  int32_t _nBatch;
};

/**
 * AsyncWorker for encode operation (embedding extraction)
 * Unlike DecodeWorker, marks ALL tokens with logits=true
 */
class EncodeWorker : public Napi::AsyncWorker {
public:
  EncodeWorker(Napi::Env env, llama_context* ctx, const std::vector<llama_token>& tokens, int32_t nBatch)
    : AsyncWorker(env), _deferred(env), _ctx(ctx), _tokens(tokens), _nBatch(nBatch) {}

  void Execute() override {
    try {
      lloyal::embedding::encode(_ctx, _tokens, _nBatch);
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
  int32_t _nBatch;
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
    InstanceMethod("getEogToken", &SessionContext::getEogToken),
    InstanceMethod("getTurnSeparator", &SessionContext::getTurnSeparator),

    // ===== PROMPT PREPARATION =====
    InstanceMethod("tokenize", &SessionContext::tokenize),
    InstanceMethod("detokenize", &SessionContext::detokenize),

    // ===== KV CACHE MANAGEMENT =====
    InstanceMethod("kvCacheSize", &SessionContext::kvCacheSize),
    InstanceMethod("kvCacheRemove", &SessionContext::kvCacheRemove),
    InstanceMethod("kvCacheSave", &SessionContext::kvCacheSave),
    InstanceMethod("kvCacheLoad", &SessionContext::kvCacheLoad),
    InstanceMethod("kvCacheClear", &SessionContext::kvCacheClear),
    InstanceMethod("clearAndReseed", &SessionContext::clearAndReseed),
    InstanceMethod("kvCacheWriteFile", &SessionContext::kvCacheWriteFile),
    InstanceMethod("kvCacheReadFile", &SessionContext::kvCacheReadFile),

    // ===== KV SEQUENCE OPERATIONS =====
    InstanceMethod("kvSeqCopy", &SessionContext::kvSeqCopy),
    InstanceMethod("kvSeqKeep", &SessionContext::kvSeqKeep),
    InstanceMethod("kvSeqPosMax", &SessionContext::kvSeqPosMax),

    // ===== HANDLE-BASED GRAMMAR =====
    InstanceMethod("createSampler", &SessionContext::createSampler),
    InstanceMethod("applySampler", &SessionContext::applySampler),
    InstanceMethod("acceptSamplerToken", &SessionContext::acceptSamplerToken),
    InstanceMethod("cloneSampler", &SessionContext::cloneSampler),
    InstanceMethod("freeSamplerHandle", &SessionContext::freeSamplerHandle),

    // ===== ATOMIC DECODE+CAPTURE =====
    InstanceMethod("decodeAndCapture", &SessionContext::decodeAndCapture),

    // ===== HELPERS =====
    InstanceMethod("formatChat", &SessionContext::formatChat),
    InstanceMethod("jsonSchemaToGrammar", &SessionContext::jsonSchemaToGrammar),
    InstanceMethod("validateChatTemplate", &SessionContext::validateChatTemplate),

    // ===== EMBEDDING EXTRACTION =====
    InstanceMethod("encode", &SessionContext::encode),
    InstanceMethod("getEmbeddings", &SessionContext::getEmbeddings),
    InstanceMethod("getEmbeddingDimension", &SessionContext::getEmbeddingDimension),
    InstanceMethod("hasPooling", &SessionContext::hasPooling),

    // ===== METRICS API =====
    InstanceMethod("modelSurprisal", &SessionContext::modelSurprisal),
    InstanceMethod("modelEntropy", &SessionContext::modelEntropy),
    InstanceMethod("createPerplexityTracker", &SessionContext::createPerplexityTracker),
    InstanceMethod("addSurprisal", &SessionContext::addSurprisal),
    InstanceMethod("getPerplexity", &SessionContext::getPerplexity),
    InstanceMethod("clonePerplexityTracker", &SessionContext::clonePerplexityTracker),
    InstanceMethod("resetPerplexityTracker", &SessionContext::resetPerplexityTracker),
    InstanceMethod("getPerplexityCount", &SessionContext::getPerplexityCount),
    InstanceMethod("freePerplexityTracker", &SessionContext::freePerplexityTracker),

    // ===== NATIVE REFERENCE IMPLEMENTATIONS =====
    InstanceMethod("greedySample", &SessionContext::greedySample),

    // ===== LIFECYCLE =====
    InstanceMethod("dispose", &SessionContext::dispose),

    // ===== BRANCH API (internal, wrapped by lib/Branch.ts) =====
    InstanceMethod("_branchCreate", &SessionContext::_branchCreate),
    InstanceMethod("_branchFork", &SessionContext::_branchFork),
    InstanceMethod("_branchCaptureLogits", &SessionContext::_branchCaptureLogits),
    InstanceMethod("_branchDecodeAndCaptureOne", &SessionContext::_branchDecodeAndCaptureOne),
    InstanceMethod("_branchDecodeAndCaptureBatch", &SessionContext::_branchDecodeAndCaptureBatch),
    InstanceMethod("_branchSample", &SessionContext::_branchSample),
    InstanceMethod("_branchAccept", &SessionContext::_branchAccept),
    InstanceMethod("_branchGetSeqId", &SessionContext::_branchGetSeqId),
    InstanceMethod("_branchGetPosition", &SessionContext::_branchGetPosition),
    InstanceMethod("_branchGetPerplexity", &SessionContext::_branchGetPerplexity),
    InstanceMethod("_branchPrune", &SessionContext::_branchPrune),
    InstanceMethod("_branchDestroy", &SessionContext::_branchDestroy),
    InstanceMethod("_branchSamplerChainReseed", &SessionContext::_branchSamplerChainReseed),
    InstanceMethod("_branchSteer", &SessionContext::_branchSteer),
    InstanceMethod("_branchClearSteer", &SessionContext::_branchClearSteer),

    // ===== PROPERTIES =====
    InstanceAccessor("vocabSize", &SessionContext::getVocabSize, nullptr),
    InstanceAccessor("memorySize", &SessionContext::getMemorySize, nullptr)
  });

  exports.Set("SessionContext", func);
  return exports;
}

// ===== HELPERS =====

lloyal::metrics::Base SessionContext::parseBase(const std::string& baseStr) {
  if (baseStr == "bits") return lloyal::metrics::Base::Bits;
  return lloyal::metrics::Base::Nats;  // Default (matches metrics.hpp)
}

SessionContext::SessionContext(const Napi::CallbackInfo& info)
  : Napi::ObjectWrap<SessionContext>(info) {
  // Constructor is called by CreateContext factory function
  // Model and context are set externally
}

SessionContext::~SessionContext() {
  if (!_disposed) {
    // Free handle-based grammar samplers first
    for (auto& [handle, sampler] : _samplerHandles) {
      if (sampler) {
        llama_sampler_free(sampler);
      }
    }
    _samplerHandles.clear();

    // Free handle-based perplexity trackers
    for (auto& [napiHandle, pplHandle] : _perplexityHandles) {
      lloyal::metrics::free_perplexity(pplHandle);
    }
    _perplexityHandles.clear();

    // Free persistent sampler chain (pattern from branch.hpp)
    if (_samplerChain) {
      lloyal::sampler::free_chain(_samplerChain);
      _samplerChain = nullptr;
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
  llama_context* context,
  int32_t nBatch
) {
  _model = std::move(model);
  _context = context;
  _nBatch = nBatch;

  std::cerr << "[SessionContext::initializeContext] Initialized:" << std::endl;
  std::cerr << "  Model ptr: " << static_cast<void*>(_model.get()) << std::endl;
  std::cerr << "  Context ptr: " << static_cast<void*>(_context) << std::endl;
  std::cerr << "  Shared refcount: " << _model.use_count() << std::endl;
}

// ===== LOGITS BUFFER MANAGEMENT =====

void SessionContext::invalidateLogits() {
  // The Kill Switch: Detach any active logits buffer
  //
  // This is called before any operation that invalidates the logits pointer:
  // - decode() - new forward pass overwrites logits
  // - encode() - embedding pass overwrites logits
  // - dispose() - context is destroyed
  //
  // After detach, any JS code holding a reference to the buffer will get
  // a TypeError when trying to access it - exactly what we want.
  if (!_logitsBufferRef.IsEmpty()) {
    try {
      Napi::ArrayBuffer buffer = _logitsBufferRef.Value();
      if (!buffer.IsDetached()) {
        buffer.Detach();
      }
    } catch (...) {
      // Buffer may have been garbage collected - that's fine
    }
    _logitsBufferRef.Reset();
  }

  // Increment step counter - any new getLogits() call will create fresh buffer
  _decodeStepId++;
}

Napi::Value SessionContext::getLogits(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (!_context) {
    throw Napi::Error::New(env, "Context not initialized");
  }

  // ===== MEMOIZATION: Return same buffer if already created for this step =====
  //
  // Pattern: "Memoized Step-Scoped Views"
  // If caller calls getLogits() twice in the same step, return the same buffer.
  // This avoids creating multiple views into the same memory.
  if (_logitsStepId == _decodeStepId && !_logitsBufferRef.IsEmpty()) {
    // Same step, reuse existing buffer
    Napi::ArrayBuffer existingBuffer = _logitsBufferRef.Value();
    const int n_vocab = lloyal::tokenizer::vocab_size(_model.get());
    return Napi::Float32Array::New(env, n_vocab, existingBuffer, 0);
  }

  // ===== NEW BUFFER: Get logits via lloyal wrapper (handles null checks) =====
  //
  // lloyal::logits::get() throws descriptive errors if:
  // - Context is null
  // - Logits unavailable (decode() not called with logits=true)
  float* logits;
  try {
    logits = lloyal::logits::get(_context, -1);
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, e.what());
  }

  const int n_vocab = lloyal::tokenizer::vocab_size(_model.get());

  // Create ArrayBuffer wrapping the logits (zero-copy!)
  // Store reference for memoization and future revocation
  Napi::ArrayBuffer buffer = Napi::ArrayBuffer::New(env, logits, n_vocab * sizeof(float));

  // Store weak reference for memoization
  _logitsBufferRef = Napi::Reference<Napi::ArrayBuffer>::New(buffer, 1);
  _logitsStepId = _decodeStepId;

  // Return Float32Array view
  return Napi::Float32Array::New(env, n_vocab, buffer, 0);
}

Napi::Value SessionContext::decode(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2 || !info[0].IsArray() || !info[1].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected (tokens: number[], position: number[, seqId: number])");
  }

  // Revoke any active logits buffer before decode
  invalidateLogits();

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

  // Extract optional seqId (default 0 for backward compatibility)
  llama_seq_id seqId = 0;
  if (info.Length() >= 3 && info[2].IsNumber()) {
    seqId = static_cast<llama_seq_id>(info[2].As<Napi::Number>().Int32Value());
  }

  // Run async
  auto* worker = new DecodeWorker(env, _context, tokens, position, seqId, _nBatch);
  worker->Queue();
  return worker->GetPromise();
}

Napi::Value SessionContext::tokenize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Expected (text: string[, addSpecial: boolean])");
  }

  std::string text = info[0].As<Napi::String>().Utf8Value();

  bool addSpecial = true;
  bool addSpecialOverridden = false;
  if (info.Length() >= 2 && info[1].IsBoolean()) {
    addSpecial = info[1].As<Napi::Boolean>().Value();
    addSpecialOverridden = true;
  }

  // Run async
  auto* worker = new TokenizeWorker(env, _model, text, addSpecial, addSpecialOverridden);
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

// ===== METRICS API =====

Napi::Value SessionContext::modelSurprisal(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Argument validation
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected number (pickedTokenId)");
  }

  int32_t pickedTokenId = info[0].As<Napi::Number>().Int32Value();

  // Optional base parameter (default: "nats")
  std::string baseStr = "nats";
  if (info.Length() >= 2 && info[1].IsString()) {
    baseStr = info[1].As<Napi::String>().Utf8Value();
  }

  lloyal::metrics::Base base = parseBase(baseStr);

  // Get logits - either from provided Float32Array or from current context
  float* logits;
  int n_vocab;

  if (info.Length() >= 3 && info[2].IsTypedArray()) {
    // Use provided logits (for captured/arbitrary logits)
    auto arr = info[2].As<Napi::TypedArray>();
    if (arr.TypedArrayType() != napi_float32_array) {
      throw Napi::TypeError::New(env, "Expected Float32Array for logits parameter");
    }
    auto float32Arr = info[2].As<Napi::Float32Array>();
    logits = float32Arr.Data();
    n_vocab = static_cast<int>(float32Arr.ElementLength());
  } else {
    // Use current context logits (default behavior)
    try {
      logits = lloyal::logits::get(_context, -1);
    } catch (const std::exception& e) {
      throw Napi::Error::New(env, e.what());
    }
    n_vocab = lloyal::tokenizer::vocab_size(_model.get());
  }

  // Compute surprisal
  float surprisal = lloyal::metrics::model_surprisal(logits, n_vocab, pickedTokenId, base);

  return Napi::Number::New(env, static_cast<double>(surprisal));
}

Napi::Value SessionContext::modelEntropy(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Optional base parameter (default: "nats")
  std::string baseStr = "nats";
  if (info.Length() >= 1 && info[0].IsString()) {
    baseStr = info[0].As<Napi::String>().Utf8Value();
  }

  lloyal::metrics::Base base = parseBase(baseStr);

  // Get logits - either from provided Float32Array or from current context
  float* logits;
  int n_vocab;

  if (info.Length() >= 2 && info[1].IsTypedArray()) {
    // Use provided logits (for captured/arbitrary logits)
    auto arr = info[1].As<Napi::TypedArray>();
    if (arr.TypedArrayType() != napi_float32_array) {
      throw Napi::TypeError::New(env, "Expected Float32Array for logits parameter");
    }
    auto float32Arr = info[1].As<Napi::Float32Array>();
    logits = float32Arr.Data();
    n_vocab = static_cast<int>(float32Arr.ElementLength());
  } else {
    // Use current context logits (default behavior)
    try {
      logits = lloyal::logits::get(_context, -1);
    } catch (const std::exception& e) {
      throw Napi::Error::New(env, e.what());
    }
    n_vocab = lloyal::tokenizer::vocab_size(_model.get());
  }

  // Compute entropy using metrics.hpp
  float entropy = lloyal::metrics::model_entropy(logits, n_vocab, base);

  return Napi::Number::New(env, static_cast<double>(entropy));
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

  // Revoke any active logits buffer before encode
  invalidateLogits();

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
  auto* worker = new EncodeWorker(env, _context, tokens, _nBatch);
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

Napi::Value SessionContext::getEogToken(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  const llama_vocab* vocab = llama_model_get_vocab(_model.get());
  llama_token eot = llama_vocab_eot(vocab);
  if (eot == LLAMA_TOKEN_NULL) {
    eot = llama_vocab_eos(vocab);  // Fallback: Zephyr-style models use EOS
  }
  if (eot == LLAMA_TOKEN_NULL) {
    throw Napi::Error::New(env, "Model has no EOT or EOS token");
  }
  return Napi::Number::New(env, static_cast<double>(eot));
}

Napi::Value SessionContext::getTurnSeparator(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Compute once, cache thereafter
  if (!_turnSeparatorCached) {
    _turnSeparatorCache = lloyal::chat_template::get_turn_separator(_model.get());
    _turnSeparatorCached = true;
  }

  Napi::Array result = Napi::Array::New(env, _turnSeparatorCache.size());
  for (size_t i = 0; i < _turnSeparatorCache.size(); i++) {
    result[i] = Napi::Number::New(env, static_cast<double>(_turnSeparatorCache[i]));
  }
  return result;
}

Napi::Value SessionContext::formatChat(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Expected (messagesJson: string[, templateOverride: string])");
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

  // Use greedy if no params, otherwise use persistent sampler chain
  // Pattern from branch.hpp: create chain once, reuse across samples, call accept() after
  if (info.Length() == 0 || !info[0].IsObject()) {
    // No params - use greedy sampling (stateless, no chain needed)
    next_token = lloyal::sampler::greedy(_context, _model.get());
  } else {
    // Use adapter to convert JS params → liblloyal-compatible structure
    LloyalSamplingParams params = adaptSamplingParamsFromJS(info[0].As<Napi::Object>());

    // Create or rebuild sampler chain if params changed
    // Pattern from branch.hpp: persistent chain enables repeat penalty tracking
    if (!_samplerChain || params != _samplerParams) {
      if (_samplerChain) {
        lloyal::sampler::free_chain(_samplerChain);
      }
      _samplerChain = lloyal::sampler::create_chain(params);
      _samplerParams = params;
    }

    // Get logits and build candidate array (pattern from branch.hpp::sample)
    const int n_vocab = lloyal::tokenizer::vocab_size(_model.get());
    float* logits = lloyal::logits::get(_context, -1);

    std::vector<llama_token_data> candidates(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
      candidates[i] = llama_token_data{static_cast<llama_token>(i), logits[i], 0.0f};
    }

    llama_token_data_array cur_p = {
      candidates.data(),
      static_cast<size_t>(n_vocab),
      -1,    // selected
      false  // sorted
    };

    // Apply persistent sampler chain (includes penalties, filters, temp, dist)
    lloyal::sampler::apply(_samplerChain, &cur_p);

    if (cur_p.selected == -1) {
      throw Napi::Error::New(env, "Sampling failed - no token selected");
    }

    next_token = cur_p.data[cur_p.selected].id;

    // Update penalty history in persistent chain (KEY CHANGE from old stateless approach)
    // This enables repeat penalty to track ALL tokens across the generation,
    // not just what's visible in the current KV cache window after clearAndReseed()
    lloyal::sampler::accept(_samplerChain, next_token);
  }

  return Napi::Number::New(env, static_cast<double>(next_token));
}

Napi::Value SessionContext::dispose(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (!_disposed) {
    // Revoke any active logits buffer before disposing
    invalidateLogits();

    // Free handle-based grammar samplers
    for (auto& [handle, sampler] : _samplerHandles) {
      if (sampler) {
        llama_sampler_free(sampler);
      }
    }
    _samplerHandles.clear();

    // Free handle-based perplexity trackers
    for (auto& [napiHandle, pplHandle] : _perplexityHandles) {
      lloyal::metrics::free_perplexity(pplHandle);
    }
    _perplexityHandles.clear();

    // Free persistent sampler chain (pattern from branch.hpp)
    if (_samplerChain) {
      lloyal::sampler::free_chain(_samplerChain);
      _samplerChain = nullptr;
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

// ===== KV SEQUENCE OPERATIONS =====

Napi::Value SessionContext::kvSeqCopy(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2) {
    throw Napi::TypeError::New(env, "Expected (srcSeqId: number, dstSeqId: number[, p0: number, p1: number])");
  }

  llama_seq_id src = toSeqId(info[0].As<Napi::Number>().DoubleValue());
  llama_seq_id dst = toSeqId(info[1].As<Napi::Number>().DoubleValue());
  llama_pos p0 = info.Length() > 2 ? toPos(info[2].As<Napi::Number>().DoubleValue()) : 0;
  llama_pos p1 = info.Length() > 3 ? toPos(info[3].As<Napi::Number>().DoubleValue()) : -1;

  lloyal::kv::seq_cp(_context, src, dst, p0, p1);
  return env.Undefined();
}

Napi::Value SessionContext::kvSeqKeep(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::TypeError::New(env, "Expected (seqId)");
  }

  llama_seq_id seq = toSeqId(info[0].As<Napi::Number>().DoubleValue());
  lloyal::kv::seq_keep(_context, seq);
  return env.Undefined();
}

Napi::Value SessionContext::kvSeqPosMax(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::TypeError::New(env, "Expected (seqId)");
  }

  llama_seq_id seq = toSeqId(info[0].As<Napi::Number>().DoubleValue());
  llama_pos pos = lloyal::kv::pos_max(_context, seq);
  return Napi::Number::New(env, static_cast<double>(pos));
}

// ===== HANDLE-BASED GRAMMAR =====

Napi::Value SessionContext::createSampler(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Expected (grammarStr: string)");
  }

  std::string grammarStr = info[0].As<Napi::String>().Utf8Value();
  llama_sampler* sampler = lloyal::grammar::init_sampler(_model.get(), grammarStr);

  if (!sampler) {
    throw Napi::Error::New(env, "Failed to create grammar sampler");
  }

  int32_t handle = _nextSamplerHandle++;
  _samplerHandles[handle] = sampler;

  return Napi::Number::New(env, static_cast<double>(handle));
}

Napi::Value SessionContext::applySampler(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2) {
    throw Napi::TypeError::New(env, "Expected (handle, logitsBuffer)");
  }

  int32_t handle = static_cast<int32_t>(info[0].As<Napi::Number>().Int32Value());

  auto it = _samplerHandles.find(handle);
  if (it == _samplerHandles.end()) {
    throw Napi::Error::New(env, "Invalid sampler handle");
  }

  // Get logits buffer
  Napi::ArrayBuffer buffer;
  if (info[1].IsArrayBuffer()) {
    buffer = info[1].As<Napi::ArrayBuffer>();
  } else if (info[1].IsTypedArray()) {
    buffer = info[1].As<Napi::TypedArray>().ArrayBuffer();
  } else {
    throw Napi::TypeError::New(env, "Expected ArrayBuffer or TypedArray");
  }

  float* logits = static_cast<float*>(buffer.Data());
  int n_vocab = lloyal::tokenizer::vocab_size(_model.get());

  // Build candidates array
  std::vector<llama_token_data> candidates(n_vocab);
  for (int i = 0; i < n_vocab; i++) {
    candidates[i] = llama_token_data{static_cast<llama_token>(i), logits[i], 0.0f};
  }

  llama_token_data_array arr = {candidates.data(), static_cast<size_t>(n_vocab), -1, false};

  // Apply grammar (modifies candidates)
  llama_sampler_apply(it->second, &arr);

  // Write back to buffer
  for (int i = 0; i < n_vocab; i++) {
    logits[i] = candidates[i].logit;
  }

  return env.Undefined();
}

Napi::Value SessionContext::acceptSamplerToken(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2) {
    throw Napi::TypeError::New(env, "Expected (handle, tokenId)");
  }

  int32_t handle = static_cast<int32_t>(info[0].As<Napi::Number>().Int32Value());
  llama_token token = static_cast<llama_token>(info[1].As<Napi::Number>().Int32Value());

  auto it = _samplerHandles.find(handle);
  if (it == _samplerHandles.end()) {
    throw Napi::Error::New(env, "Invalid sampler handle");
  }

  llama_sampler_accept(it->second, token);
  return env.Undefined();
}

Napi::Value SessionContext::cloneSampler(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::TypeError::New(env, "Expected (handle)");
  }

  int32_t handle = static_cast<int32_t>(info[0].As<Napi::Number>().Int32Value());

  auto it = _samplerHandles.find(handle);
  if (it == _samplerHandles.end()) {
    throw Napi::Error::New(env, "Invalid sampler handle");
  }

  llama_sampler* cloned = lloyal::grammar::clone_sampler(it->second);
  if (!cloned) {
    throw Napi::Error::New(env, "Failed to clone sampler");
  }

  int32_t newHandle = _nextSamplerHandle++;
  _samplerHandles[newHandle] = cloned;

  return Napi::Number::New(env, static_cast<double>(newHandle));
}

Napi::Value SessionContext::freeSamplerHandle(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::TypeError::New(env, "Expected (handle)");
  }

  int32_t handle = static_cast<int32_t>(info[0].As<Napi::Number>().Int32Value());

  auto it = _samplerHandles.find(handle);
  if (it != _samplerHandles.end()) {
    llama_sampler_free(it->second);
    _samplerHandles.erase(it);
  }

  return env.Undefined();
}

// ===== PERPLEXITY TRACKING =====

Napi::Value SessionContext::createPerplexityTracker(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Create new perplexity tracker via metrics.hpp
  lloyal::metrics::PerplexityHandle handle = lloyal::metrics::create_perplexity();

  // Generate N-API handle
  int32_t napiHandle = _nextPerplexityHandle++;
  _perplexityHandles[napiHandle] = handle;

  return Napi::Number::New(env, static_cast<double>(napiHandle));
}

Napi::Value SessionContext::addSurprisal(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Argument validation
  if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected (handle: number, surprisal: number)");
  }

  int32_t napiHandle = info[0].As<Napi::Number>().Int32Value();
  double surprisal = info[1].As<Napi::Number>().DoubleValue();

  // Lookup handle
  auto it = _perplexityHandles.find(napiHandle);
  if (it == _perplexityHandles.end()) {
    throw Napi::Error::New(env, "Invalid perplexity tracker handle");
  }

  // Add surprisal to tracker
  lloyal::metrics::add_surprisal(it->second, static_cast<float>(surprisal));

  return env.Undefined();
}

Napi::Value SessionContext::getPerplexity(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Argument validation
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected handle: number");
  }

  int32_t napiHandle = info[0].As<Napi::Number>().Int32Value();

  // Lookup handle
  auto it = _perplexityHandles.find(napiHandle);
  if (it == _perplexityHandles.end()) {
    throw Napi::Error::New(env, "Invalid perplexity tracker handle");
  }

  // Get perplexity value
  float ppl = lloyal::metrics::get_ppl(it->second);

  return Napi::Number::New(env, static_cast<double>(ppl));
}

Napi::Value SessionContext::clonePerplexityTracker(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Argument validation
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected handle: number");
  }

  int32_t sourceHandle = info[0].As<Napi::Number>().Int32Value();

  // Lookup source handle
  auto it = _perplexityHandles.find(sourceHandle);
  if (it == _perplexityHandles.end()) {
    throw Napi::Error::New(env, "Invalid source perplexity tracker handle");
  }

  // Clone via metrics.hpp
  lloyal::metrics::PerplexityHandle clonedHandle =
      lloyal::metrics::clone_perplexity(it->second);

  // Generate new N-API handle
  int32_t newNapiHandle = _nextPerplexityHandle++;
  _perplexityHandles[newNapiHandle] = clonedHandle;

  return Napi::Number::New(env, static_cast<double>(newNapiHandle));
}

Napi::Value SessionContext::resetPerplexityTracker(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Argument validation
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected handle: number");
  }

  int32_t napiHandle = info[0].As<Napi::Number>().Int32Value();

  // Lookup handle
  auto it = _perplexityHandles.find(napiHandle);
  if (it == _perplexityHandles.end()) {
    throw Napi::Error::New(env, "Invalid perplexity tracker handle");
  }

  // Reset tracker
  lloyal::metrics::reset_perplexity(it->second);

  return env.Undefined();
}

Napi::Value SessionContext::getPerplexityCount(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Argument validation
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected handle: number");
  }

  int32_t napiHandle = info[0].As<Napi::Number>().Int32Value();

  // Lookup handle
  auto it = _perplexityHandles.find(napiHandle);
  if (it == _perplexityHandles.end()) {
    throw Napi::Error::New(env, "Invalid perplexity tracker handle");
  }

  // Get token count
  int count = lloyal::metrics::get_count(it->second);

  return Napi::Number::New(env, static_cast<double>(count));
}

Napi::Value SessionContext::freePerplexityTracker(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Argument validation
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected handle: number");
  }

  int32_t napiHandle = info[0].As<Napi::Number>().Int32Value();

  // Lookup and remove handle
  auto it = _perplexityHandles.find(napiHandle);
  if (it == _perplexityHandles.end()) {
    throw Napi::Error::New(env, "Invalid perplexity tracker handle");
  }

  // Free via metrics.hpp
  lloyal::metrics::free_perplexity(it->second);

  // Remove from map
  _perplexityHandles.erase(it);

  return env.Undefined();
}

// ===== ATOMIC DECODE+CAPTURE =====

Napi::Value SessionContext::decodeAndCapture(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 4) {
    throw Napi::TypeError::New(env, "Expected (tokens, position, seqId, destBuffer)");
  }

  // Parse tokens
  Napi::Array tokensArray = info[0].As<Napi::Array>();
  std::vector<llama_token> tokens(tokensArray.Length());
  for (uint32_t i = 0; i < tokensArray.Length(); i++) {
    tokens[i] = static_cast<llama_token>(tokensArray.Get(i).As<Napi::Number>().Int32Value());
  }

  int32_t position = info[1].As<Napi::Number>().Int32Value();
  llama_seq_id seqId = toSeqId(info[2].As<Napi::Number>().DoubleValue());

  // Get dest buffer
  Napi::ArrayBuffer destBuffer;
  if (info[3].IsArrayBuffer()) {
    destBuffer = info[3].As<Napi::ArrayBuffer>();
  } else if (info[3].IsTypedArray()) {
    destBuffer = info[3].As<Napi::TypedArray>().ArrayBuffer();
  } else {
    throw Napi::TypeError::New(env, "destBuffer must be ArrayBuffer or TypedArray");
  }

  float* dest = static_cast<float*>(destBuffer.Data());
  int n_vocab = lloyal::tokenizer::vocab_size(_model.get());

  // Atomic: lock mutex through decode + logits copy
  {
    std::lock_guard<std::mutex> lock(_decodeMutex);

    // Invalidate any existing logits views
    invalidateLogits();
    _decodeStepId++;

    // Decode
    lloyal::decoder::decode_tokens(_context, tokens, position, _nBatch, seqId);

    // Capture logits immediately
    float* logits = lloyal::logits::get(_context, -1);
    std::memcpy(dest, logits, n_vocab * sizeof(float));
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

  // CRITICAL: Invalidate logits before KV cache modification
  // Logits may reference positions that will be evicted
  // (matches pattern from decode() line 801, encode() line 1035)
  invalidateLogits();

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

Napi::Value SessionContext::clearAndReseed(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Args: sinks (Array<number>), tail (Array<number>)
  if (info.Length() < 2 || !info[0].IsArray() || !info[1].IsArray()) {
    throw Napi::TypeError::New(env, "Expected (sinks: number[], tail: number[])");
  }

  // Extract sinks array
  Napi::Array jsSinks = info[0].As<Napi::Array>();
  std::vector<llama_token> sinks;
  sinks.reserve(jsSinks.Length());
  for (uint32_t i = 0; i < jsSinks.Length(); i++) {
    Napi::Value val = jsSinks.Get(i);
    if (!val.IsNumber()) {
      throw Napi::TypeError::New(env, "sinks array must contain only numbers");
    }
    sinks.push_back(static_cast<llama_token>(val.As<Napi::Number>().Int32Value()));
  }

  // Extract tail array
  Napi::Array jsTail = info[1].As<Napi::Array>();
  std::vector<llama_token> tail;
  tail.reserve(jsTail.Length());
  for (uint32_t i = 0; i < jsTail.Length(); i++) {
    Napi::Value val = jsTail.Get(i);
    if (!val.IsNumber()) {
      throw Napi::TypeError::New(env, "tail array must contain only numbers");
    }
    tail.push_back(static_cast<llama_token>(val.As<Napi::Number>().Int32Value()));
  }

  auto* worker = new ClearAndReseedWorker(env, _context, std::move(sinks), std::move(tail), _nBatch);
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

  // Extract nBatch (optional, default N_BATCH_INIT = 512)
  int32_t nBatch = lloyal::defaults::N_BATCH_INIT;
  if (options.Has("nBatch") && options.Get("nBatch").IsNumber()) {
    nBatch = options.Get("nBatch").As<Napi::Number>().Int32Value();
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

  // Extract nSeqMax (optional, default 1)
  // Set > 1 to enable multiple independent KV cache sequences (kvSeqCopy, etc.)
  // NOTE: When sequences share a common prefix, kv_unified=true (default) is optimal.
  //       If sequences don't share a prefix, consider setting kv_unified=false for perf.
  int32_t nSeqMax = 1;
  if (options.Has("nSeqMax") && options.Get("nSeqMax").IsNumber()) {
    nSeqMax = options.Get("nSeqMax").As<Napi::Number>().Int32Value();
  }

  // Ensure llama backend is initialized on main thread (thread-safe, once)
  BackendManager::ensureInitialized();

  // Normalize and validate path BEFORE queuing async work
  std::string fsPath = liblloyal_node::FileSystem::normalizePath(modelPath);
  if (fsPath != modelPath) {
    std::cout << "[CreateContext] Normalized " << modelPath << " → " << fsPath << std::endl;
  }

  if (!liblloyal_node::FileSystem::exists(fsPath)) {
    std::cout << "[CreateContext] File does not exist: " << fsPath << std::endl;
    throw Napi::Error::New(env, "Model file not found: " + fsPath);
  }

  size_t fileSize = liblloyal_node::FileSystem::getSize(fsPath);
  std::cout << "[CreateContext] File validated: " << fsPath << " (" << fileSize << " bytes)" << std::endl;

  // Load model on main thread
  // Note: With XCFramework build, this works reliably on main thread
  // (async loading was failing with CMake build due to binary incompatibility)
  std::cout << "[CreateContext] Loading model from XCFramework..." << std::endl;

  llama_model_params model_params = llama_model_default_params();
  // -1 = offload all layers to GPU (auto-detect), 0 = CPU only
  model_params.n_gpu_layers = -1;

  std::cout << "[CreateContext] Acquiring from ModelRegistry..." << std::endl;
  auto sharedModel = lloyal::ModelRegistry::acquire(fsPath, model_params);

  if (!sharedModel) {
    throw Napi::Error::New(env, "Failed to load model from " + fsPath);
  }

  std::cout << "[CreateContext] Model loaded (refcount: " << sharedModel.use_count() << ")" << std::endl;

  // Create context
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = static_cast<uint32_t>(nCtx);
  ctx_params.n_batch = static_cast<uint32_t>(nBatch);
  ctx_params.n_ubatch = static_cast<uint32_t>(nBatch);
  ctx_params.n_threads = static_cast<uint32_t>(nThreads);
  ctx_params.n_seq_max = static_cast<uint32_t>(nSeqMax);
  ctx_params.kv_unified = true;  // Share KV across sequences (efficient for branching)

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
  obj->initializeContext(std::move(sharedModel), ctx, nBatch);

  std::cout << "[CreateContext] SessionContext initialized" << std::endl;
  return instance;
}

// ===== BRANCH API IMPLEMENTATION =====

Napi::Value SessionContext::_branchCreate(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2) {
    throw Napi::Error::New(env, "_branchCreate requires (seqId, position[, params[, nBatch[, grammar]]])");
  }

  auto seqId = static_cast<llama_seq_id>(info[0].As<Napi::Number>().Int32Value());
  auto position = static_cast<llama_pos>(info[1].As<Napi::Number>().Int32Value());

  // Extract sampling params from JS object (optional third arg)
  LloyalSamplingParams params;
  if (info.Length() >= 3 && info[2].IsObject()) {
    params = adaptSamplingParamsFromJS(info[2].As<Napi::Object>());
  }

  // Per-branch nBatch override (optional 4th arg), falls back to context default
  int32_t nBatch = _nBatch;
  if (info.Length() >= 4 && info[3].IsNumber()) {
    nBatch = info[3].As<Napi::Number>().Int32Value();
  }

  // Grammar string (optional 5th arg)
  const char* grammar_str = nullptr;
  std::string grammar_storage;  // Keep string alive for duration of create()
  if (info.Length() >= 5 && info[4].IsString()) {
    grammar_storage = info[4].As<Napi::String>().Utf8Value();
    grammar_str = grammar_storage.c_str();
  }

  // Create branch using lloyal::branch::create
  auto handle = lloyal::branch::create(
    _context,
    _model.get(),
    seqId,
    position,
    params,
    nBatch,       // per-branch override or context default
    grammar_str,  // grammar GBNF string (or nullptr)
    nullptr,      // boundary_tracker
    &_branchStore
  );

  if (handle == lloyal::branch::INVALID_HANDLE) {
    throw Napi::Error::New(env, "Failed to create branch");
  }

  return Napi::Number::New(env, handle);
}

Napi::Value SessionContext::_branchFork(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2) {
    throw Napi::Error::New(env, "_branchFork requires (handle, newSeqId)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  auto newSeqId = static_cast<llama_seq_id>(info[1].As<Napi::Number>().Int32Value());

  auto newHandle = lloyal::branch::fork(handle, newSeqId, &_branchStore);

  if (newHandle == lloyal::branch::INVALID_HANDLE) {
    throw Napi::Error::New(env, "Failed to fork branch");
  }

  return Napi::Number::New(env, newHandle);
}

Napi::Value SessionContext::_branchCaptureLogits(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchCaptureLogits requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  lloyal::branch::capture_logits(handle, &_branchStore);

  return env.Undefined();
}

Napi::Value SessionContext::_branchDecodeAndCaptureOne(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2) {
    throw Napi::Error::New(env, "_branchDecodeAndCaptureOne requires (handle, token)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  auto token = static_cast<llama_token>(info[1].As<Napi::Number>().Int32Value());

  lloyal::branch::decode_and_capture_one(handle, token, &_branchStore);

  return env.Undefined();
}

// Bulk-decode tokens into a branch's KV cache and capture final logits.
//
// tokens.size() is the total token count (n_tokens).  The branch's n_batch
// (set at Branch.create via the nBatch parameter, stored on BranchState)
// controls the chunk size — decode_and_capture_batch passes both to
// decoder::decode_tokens which loops: min(n_tokens - processed, n_batch)
// tokens per llama_decode call.
//
// Does NOT accept tokens into the sampler's penalty window.
// Wrapped by Branch.prefill() on the JS side.
Napi::Value SessionContext::_branchDecodeAndCaptureBatch(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsArray()) {
    throw Napi::Error::New(env, "_branchDecodeAndCaptureBatch requires (handle, tokens[])");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());

  Napi::Array jsTokens = info[1].As<Napi::Array>();
  std::vector<llama_token> tokens;
  tokens.reserve(jsTokens.Length());
  for (uint32_t i = 0; i < jsTokens.Length(); i++) {
    tokens.push_back(static_cast<llama_token>(jsTokens.Get(i).As<Napi::Number>().Int32Value()));
  }

  if (!tokens.empty()) {
    lloyal::branch::decode_and_capture_batch(handle, tokens.data(), tokens.size(), &_branchStore);
  }

  return env.Undefined();
}

Napi::Value SessionContext::_branchSample(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchSample requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  auto token = lloyal::branch::sample(handle, &_branchStore);

  return Napi::Number::New(env, token);
}

Napi::Value SessionContext::_branchAccept(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2) {
    throw Napi::Error::New(env, "_branchAccept requires (handle, token)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  auto token = static_cast<llama_token>(info[1].As<Napi::Number>().Int32Value());

  lloyal::branch::accept_token(handle, token, &_branchStore);

  return env.Undefined();
}

Napi::Value SessionContext::_branchGetSeqId(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchGetSeqId requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  auto seqId = lloyal::branch::get_seq_id(handle, &_branchStore);

  return Napi::Number::New(env, seqId);
}

Napi::Value SessionContext::_branchGetPosition(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchGetPosition requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  auto position = lloyal::branch::get_position(handle, &_branchStore);

  return Napi::Number::New(env, position);
}

Napi::Value SessionContext::_branchGetPerplexity(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchGetPerplexity requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  auto ppl = lloyal::branch::get_perplexity(handle, &_branchStore);

  return Napi::Number::New(env, ppl);
}

Napi::Value SessionContext::_branchPrune(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchPrune requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  lloyal::branch::prune(handle, &_branchStore);

  return env.Undefined();
}

Napi::Value SessionContext::_branchDestroy(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchDestroy requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  lloyal::branch::destroy(handle, &_branchStore);

  return env.Undefined();
}

Napi::Value SessionContext::_branchSamplerChainReseed(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2) {
    throw Napi::Error::New(env, "_branchSamplerChainReseed requires (handle, seed)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  auto seed = static_cast<uint32_t>(info[1].As<Napi::Number>().Uint32Value());

  // Get branch state to access sampler chain
  auto* state = _branchStore.get(handle);
  if (!state) {
    throw Napi::Error::New(env, "_branchSamplerChainReseed: invalid handle");
  }

  // Only reseed stochastic chains (has_dist_sampler=true)
  // Reseeding greedy chains would corrupt them
  if (state->sampler_chain && state->has_dist_sampler) {
    lloyal::sampler::reseed_chain(state->sampler_chain, seed);
  }

  return env.Undefined();
}

Napi::Value SessionContext::_branchSteer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2) {
    throw Napi::Error::New(env, "_branchSteer requires (handle, biases[])");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());

  if (!info[1].IsArray()) {
    throw Napi::Error::New(env, "_branchSteer: biases must be an array");
  }

  Napi::Array biasArray = info[1].As<Napi::Array>();
  uint32_t length = biasArray.Length();

  // Build vector of biases from JS [{token, bias}, ...]
  std::vector<llama_logit_bias> biases;
  biases.reserve(length);

  for (uint32_t i = 0; i < length; i++) {
    Napi::Value item = biasArray[i];
    if (!item.IsObject()) {
      throw Napi::Error::New(env, "_branchSteer: each bias must be {token, bias}");
    }
    Napi::Object obj = item.As<Napi::Object>();

    if (!obj.Has("token") || !obj.Has("bias")) {
      throw Napi::Error::New(env, "_branchSteer: each bias must have 'token' and 'bias' properties");
    }

    llama_logit_bias bias;
    bias.token = static_cast<llama_token>(obj.Get("token").As<Napi::Number>().Int32Value());
    bias.bias = obj.Get("bias").As<Napi::Number>().FloatValue();
    biases.push_back(bias);
  }

  // Create steer function that applies these biases
  // Capture by value so the vector is owned by the lambda
  lloyal::branch::set_steer(handle, [biases](llama_token_data_array& cur_p) {
    for (const auto& bias : biases) {
      for (size_t i = 0; i < cur_p.size; ++i) {
        if (cur_p.data[i].id == bias.token) {
          cur_p.data[i].logit += bias.bias;
          break;
        }
      }
    }
  }, &_branchStore);

  return env.Undefined();
}

Napi::Value SessionContext::_branchClearSteer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchClearSteer requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());

  lloyal::branch::clear_steer(handle, &_branchStore);

  return env.Undefined();
}

} // namespace liblloyal_node
