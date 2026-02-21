#include "SessionContext.hpp"
#include "BackendManager.hpp"
#include "FileSystem.h"
#include <lloyal/sampler.hpp>
#include <lloyal/tokenizer.hpp>
#include <lloyal/common.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/chat_in.hpp>
#include <lloyal/chat_out.hpp>
#include <lloyal/grammar.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/embedding.hpp>
#include <lloyal/metrics.hpp>
#include <cmath>
#include <iostream>

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
                   const lloyal::chat_in::FormatInputs& inputs)
    : AsyncWorker(env), _deferred(env), _model(model), _inputs(inputs) {}

  void Execute() override {
    try {
      lloyal::chat_in::FormatResult result = lloyal::chat_in::format(
        _model.get(), _inputs
      );

      if (result.prompt.empty()) {
        SetError("Chat template formatting failed");
        return;
      }

      _result = result;
    } catch (const std::exception& e) {
      SetError(e.what());
    }
  }

  void OnOK() override {
    Napi::Env env = Env();

    Napi::Object result = Napi::Object::New(env);
    result.Set("prompt", Napi::String::New(env, _result.prompt));

    // stopTokens (backward compat)
    Napi::Array stopTokens = Napi::Array::New(env, _result.additional_stops.size());
    for (size_t i = 0; i < _result.additional_stops.size(); i++) {
      stopTokens[i] = Napi::String::New(env, _result.additional_stops[i]);
    }
    result.Set("stopTokens", stopTokens);

    // Format awareness fields
    result.Set("format", Napi::Number::New(env, static_cast<double>(_result.format)));
    result.Set("grammar", Napi::String::New(env, _result.grammar));
    result.Set("grammarLazy", Napi::Boolean::New(env, _result.grammar_lazy));
    result.Set("thinkingForcedOpen", Napi::Boolean::New(env, _result.thinking_forced_open));
    result.Set("reasoningFormat", Napi::Number::New(env, static_cast<double>(_result.reasoning_format)));
    result.Set("parser", Napi::String::New(env, _result.parser));

    // grammarTriggers: Array<{ type: number, value: string, token: number }>
    Napi::Array triggers = Napi::Array::New(env, _result.grammar_triggers.size());
    for (size_t i = 0; i < _result.grammar_triggers.size(); i++) {
      Napi::Object trigger = Napi::Object::New(env);
      trigger.Set("type", Napi::Number::New(env, static_cast<double>(_result.grammar_triggers[i].type)));
      trigger.Set("value", Napi::String::New(env, _result.grammar_triggers[i].value));
      trigger.Set("token", Napi::Number::New(env, static_cast<double>(_result.grammar_triggers[i].token)));
      triggers[i] = trigger;
    }
    result.Set("grammarTriggers", triggers);

    // preservedTokens: string[]
    Napi::Array preserved = Napi::Array::New(env, _result.preserved_tokens.size());
    for (size_t i = 0; i < _result.preserved_tokens.size(); i++) {
      preserved[i] = Napi::String::New(env, _result.preserved_tokens[i]);
    }
    result.Set("preservedTokens", preserved);

    _deferred.Resolve(result);
  }

  void OnError(const Napi::Error& err) override {
    _deferred.Reject(err.Value());
  }

  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  std::shared_ptr<llama_model> _model;
  lloyal::chat_in::FormatInputs _inputs;
  lloyal::chat_in::FormatResult _result;
};

// ===== BRANCH / STORE / DECODE ASYNC WORKERS =====

/**
 * AsyncWorker for bulk branch decode + logits capture (prompt injection)
 * Wraps lloyal::branch::prefill on libuv pool thread
 */
class BranchPrefillWorker : public Napi::AsyncWorker {
public:
  BranchPrefillWorker(Napi::Env env,
                      lloyal::branch::BranchStore& store,
                      lloyal::branch::BranchHandle handle,
                      std::vector<llama_token> tokens)
    : AsyncWorker(env), _deferred(env), _store(store), _handle(handle),
      _tokens(std::move(tokens)) {}

  void Execute() override {
    try {
      lloyal::branch::prefill(_handle, _tokens.data(), _tokens.size(), _store);
    } catch (const std::exception& e) { SetError(e.what()); }
  }

  void OnOK() override { _deferred.Resolve(Env().Undefined()); }
  void OnError(const Napi::Error& err) override { _deferred.Reject(err.Value()); }
  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  lloyal::branch::BranchStore& _store;
  lloyal::branch::BranchHandle _handle;
  std::vector<llama_token> _tokens;
};

/**
 * AsyncWorker for batched multi-branch commit (accept + decode_each)
 * Accept-first ordering with rollback: accepts tokens for correct PPL measurement,
 * then decodes. On decode failure, restores sampler/grammar/metrics from clones.
 */
class StoreCommitWorker : public Napi::AsyncWorker {
public:
  StoreCommitWorker(Napi::Env env,
                    lloyal::branch::BranchStore& store,
                    std::vector<lloyal::branch::DecodeEachItem> items)
    : AsyncWorker(env), _deferred(env), _store(store), _items(std::move(items)) {}

  void Execute() override {
    // RAII snapshot of accept-mutable state. Destructor frees anything still
    // owned, so partial clones from a throwing OOM don't leak.
    struct Snapshot {
      lloyal::branch::SamplerChainHandle sampler = 0;
      lloyal::branch::GrammarHandle grammar = 0;
      lloyal::branch::MetricsHandle metrics = 0;
      lloyal::branch::BranchStore* store = nullptr;

      ~Snapshot() {
        if (!store) return;
        if (sampler) store->free_sampler(sampler);
        if (grammar) store->free_grammar(grammar);
        if (metrics) store->free_metrics(metrics);
      }

      void restore_into(lloyal::branch::BranchState& st) {
        std::swap(sampler, st.sampler_chain);
        std::swap(grammar, st.grammar);
        std::swap(metrics, st.metrics);
      }
    };

    // Pre-size with unique_ptr so the vector never needs to move elements
    std::vector<std::unique_ptr<Snapshot>> snaps(_items.size());

    try {
      // Phase 1: snapshot all accept-mutable state (no mutations yet)
      for (size_t i = 0; i < _items.size(); ++i) {
        auto* st = _store.get(_items[i].handle);
        if (!st) throw std::runtime_error("StoreCommitWorker: invalid handle");

        auto s = std::make_unique<Snapshot>();
        s->store = &_store;
        s->sampler = st->sampler_chain != 0
            ? _store.clone_sampler(st->sampler_chain) : 0;
        s->grammar = st->grammar != 0
            ? _store.clone_grammar(st->grammar) : 0;
        s->metrics = st->metrics != 0
            ? _store.clone_metrics(st->metrics) : 0;
        snaps[i] = std::move(s);
      }

      // Phase 2: accept all tokens (in-memory state mutation, won't throw)
      for (auto& item : _items)
        lloyal::branch::accept_token(item.handle, item.token, _store);

      // Phase 3: decode (single GPU batch — the only realistic failure point)
      _store.decode_each(_items);

      // Success — discard snapshots
      snaps.clear();

    } catch (const std::exception& e) {
      // Restore all branches — un-mutated branches get a harmless equivalent swap
      for (size_t i = 0; i < _items.size(); ++i) {
        auto* st = _store.get(_items[i].handle);
        if (st && snaps[i]) snaps[i]->restore_into(*st);
      }
      // ~Snapshot frees the swapped-out (post-accept) state

      SetError(e.what());
    }
  }

  void OnOK() override { _deferred.Resolve(Env().Undefined()); }
  void OnError(const Napi::Error& err) override { _deferred.Reject(err.Value()); }
  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  lloyal::branch::BranchStore& _store;
  std::vector<lloyal::branch::DecodeEachItem> _items;
};

/**
 * AsyncWorker for multi-branch prefill (decode_scatter)
 * Owns token storage; rebuilds spans in Execute() from owned vectors
 */
class StorePrefillWorker : public Napi::AsyncWorker {
public:
  StorePrefillWorker(Napi::Env env,
                     lloyal::branch::BranchStore& store,
                     std::vector<lloyal::branch::BranchHandle> handles,
                     std::vector<std::vector<llama_token>> tokenStorage)
    : AsyncWorker(env), _deferred(env), _store(store),
      _handles(std::move(handles)), _tokenStorage(std::move(tokenStorage)) {}

  void Execute() override {
    try {
      // Rebuild DecodeScatterItems with spans into owned storage
      std::vector<lloyal::branch::DecodeScatterItem> items(_handles.size());
      for (size_t i = 0; i < _handles.size(); i++) {
        items[i].handle = _handles[i];
        items[i].tokens = _tokenStorage[i];
      }
      _store.decode_scatter(items);
    } catch (const std::exception& e) { SetError(e.what()); }
  }

  void OnOK() override { _deferred.Resolve(Env().Undefined()); }
  void OnError(const Napi::Error& err) override { _deferred.Reject(err.Value()); }
  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  lloyal::branch::BranchStore& _store;
  std::vector<lloyal::branch::BranchHandle> _handles;
  std::vector<std::vector<llama_token>> _tokenStorage;
};

/**
 * AsyncWorker for JSON schema → GBNF grammar conversion
 * Pure CPU, no shared state — cleanest worker
 */
class JsonSchemaToGrammarWorker : public Napi::AsyncWorker {
public:
  JsonSchemaToGrammarWorker(Napi::Env env, std::string schemaJson)
    : AsyncWorker(env), _deferred(env), _schemaJson(std::move(schemaJson)) {}

  void Execute() override {
    try {
      _result = lloyal::grammar::from_json_schema(_schemaJson);
    } catch (const std::exception& e) { SetError(e.what()); }
  }

  void OnOK() override { _deferred.Resolve(Napi::String::New(Env(), _result)); }
  void OnError(const Napi::Error& err) override { _deferred.Reject(err.Value()); }
  Napi::Promise GetPromise() { return _deferred.Promise(); }

private:
  Napi::Promise::Deferred _deferred;
  std::string _schemaJson;
  std::string _result;
};

// ===== SESSIONCONTEXT IMPLEMENTATION =====

Napi::Object SessionContext::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(env, "SessionContext", {
    // ===== CORE =====
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

    // ===== HELPERS =====
    InstanceMethod("formatChat", &SessionContext::formatChat),
    InstanceMethod("parseChatOutput", &SessionContext::parseChatOutput),
    InstanceMethod("jsonSchemaToGrammar", &SessionContext::jsonSchemaToGrammar),
    InstanceMethod("validateChatTemplate", &SessionContext::validateChatTemplate),

    // ===== EMBEDDING EXTRACTION =====
    InstanceMethod("encode", &SessionContext::encode),
    InstanceMethod("getEmbeddings", &SessionContext::getEmbeddings),
    InstanceMethod("getEmbeddingDimension", &SessionContext::getEmbeddingDimension),
    InstanceMethod("hasPooling", &SessionContext::hasPooling),

    // ===== LIFECYCLE =====
    InstanceMethod("dispose", &SessionContext::dispose),

    // ===== BRANCH API (internal, wrapped by lib/Branch.ts) =====
    InstanceMethod("_branchCreate", &SessionContext::_branchCreate),
    InstanceMethod("_branchFork", &SessionContext::_branchFork),
    InstanceMethod("_branchPrefill", &SessionContext::_branchPrefill),
    InstanceMethod("_branchSample", &SessionContext::_branchSample),
    InstanceMethod("_branchAccept", &SessionContext::_branchAccept),
    InstanceMethod("_branchGetPosition", &SessionContext::_branchGetPosition),
    InstanceMethod("_branchGetPerplexity", &SessionContext::_branchGetPerplexity),
    InstanceMethod("_branchGetLogits", &SessionContext::_branchGetLogits),
    InstanceMethod("_branchPrune", &SessionContext::_branchPrune),
    InstanceMethod("_branchPruneSubtree", &SessionContext::_branchPruneSubtree),
    InstanceMethod("_branchParent", &SessionContext::_branchParent),
    InstanceMethod("_branchChildren", &SessionContext::_branchChildren),
    InstanceMethod("_branchIsLeaf", &SessionContext::_branchIsLeaf),
    InstanceMethod("_branchIsActive", &SessionContext::_branchIsActive),
    InstanceMethod("_branchSamplerChainReseed", &SessionContext::_branchSamplerChainReseed),
    InstanceMethod("_branchSteer", &SessionContext::_branchSteer),
    InstanceMethod("_branchClearSteer", &SessionContext::_branchClearSteer),
    InstanceMethod("_branchSetSamplerParams", &SessionContext::_branchSetSamplerParams),
    InstanceMethod("_branchSetGrammar", &SessionContext::_branchSetGrammar),
    InstanceMethod("_branchModelEntropy", &SessionContext::_branchModelEntropy),
    InstanceMethod("_branchModelSurprisal", &SessionContext::_branchModelSurprisal),
    InstanceMethod("_branchGetSamplingPerplexity", &SessionContext::_branchGetSamplingPerplexity),
    InstanceMethod("_branchSetLogitBias", &SessionContext::_branchSetLogitBias),
    InstanceMethod("_branchClearLogitBias", &SessionContext::_branchClearLogitBias),

    // ===== STORE API (internal, wrapped by lib/BranchStore.js) =====
    InstanceMethod("_storeCommit", &SessionContext::_storeCommit),
    InstanceMethod("_storePrefill", &SessionContext::_storePrefill),
    InstanceMethod("_storeRetainOnly", &SessionContext::_storeRetainOnly),
    InstanceMethod("_storeAvailable", &SessionContext::_storeAvailable),

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
  _branchStore.init_tenancy(_context);

  std::cerr << "[SessionContext::initializeContext] Initialized:" << std::endl;
  std::cerr << "  Model ptr: " << static_cast<void*>(_model.get()) << std::endl;
  std::cerr << "  Context ptr: " << static_cast<void*>(_context) << std::endl;
  std::cerr << "  Shared refcount: " << _model.use_count() << std::endl;
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
    _turnSeparatorCache = lloyal::chat_in::get_turn_separator(_model.get());
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
    throw Napi::TypeError::New(env, "Expected (messagesJson: string[, options: object])");
  }

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = info[0].As<Napi::String>().Utf8Value();

  // Second argument: options object (or string for backward compat)
  if (info.Length() >= 2) {
    if (info[1].IsString()) {
      // Backward compat: formatChat(messagesJson, templateOverride)
      inputs.template_override = info[1].As<Napi::String>().Utf8Value();
    } else if (info[1].IsObject()) {
      Napi::Object opts = info[1].As<Napi::Object>();

      if (opts.Has("templateOverride") && opts.Get("templateOverride").IsString()) {
        inputs.template_override = opts.Get("templateOverride").As<Napi::String>().Utf8Value();
      }
      if (opts.Has("tools") && opts.Get("tools").IsString()) {
        inputs.tools_json = opts.Get("tools").As<Napi::String>().Utf8Value();
      }
      if (opts.Has("toolChoice") && opts.Get("toolChoice").IsString()) {
        inputs.tool_choice = opts.Get("toolChoice").As<Napi::String>().Utf8Value();
      }
      if (opts.Has("parallelToolCalls") && opts.Get("parallelToolCalls").IsBoolean()) {
        inputs.parallel_tool_calls = opts.Get("parallelToolCalls").As<Napi::Boolean>().Value();
      }
      if (opts.Has("reasoningFormat") && opts.Get("reasoningFormat").IsString()) {
        inputs.reasoning_format = opts.Get("reasoningFormat").As<Napi::String>().Utf8Value();
      }
      if (opts.Has("enableThinking") && opts.Get("enableThinking").IsBoolean()) {
        inputs.enable_thinking = opts.Get("enableThinking").As<Napi::Boolean>().Value();
      }
      if (opts.Has("jsonSchema") && opts.Get("jsonSchema").IsString()) {
        inputs.json_schema = opts.Get("jsonSchema").As<Napi::String>().Utf8Value();
      }
      if (opts.Has("grammar") && opts.Get("grammar").IsString()) {
        inputs.grammar = opts.Get("grammar").As<Napi::String>().Utf8Value();
      }
      if (opts.Has("addGenerationPrompt") && opts.Get("addGenerationPrompt").IsBoolean()) {
        inputs.add_generation_prompt = opts.Get("addGenerationPrompt").As<Napi::Boolean>().Value();
      }
    }
  }

  auto* worker = new FormatChatWorker(env, _model, inputs);
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

Napi::Value SessionContext::dispose(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (!_disposed) {
    // Drain branch store while context is still alive
    _branchStore.drain();

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

  auto* worker = new JsonSchemaToGrammarWorker(env, std::move(schemaJson));
  worker->Queue();
  return worker->GetPromise();
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
      // Use lloyal::chat_in from liblloyal (handles error logging)
      // Pattern matches HybridSessionContext.cpp:365-372
      _result = lloyal::chat_in::validate(_templateString);
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

// ===== CHAT OUTPUT PARSING =====

Napi::Value SessionContext::parseChatOutput(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  // Args: output (string), format (number), options? (object)
  if (info.Length() < 2 || !info[0].IsString() || !info[1].IsNumber()) {
    throw Napi::TypeError::New(env, "Expected (output: string, format: number[, options: object])");
  }

  std::string output = info[0].As<Napi::String>().Utf8Value();
  auto format = static_cast<common_chat_format>(info[1].As<Napi::Number>().Int32Value());

  // Optional params
  auto reasoning_format = COMMON_REASONING_FORMAT_NONE;
  bool is_partial = false;
  bool thinking_forced_open = false;
  std::string parser_data;

  if (info.Length() >= 3 && info[2].IsObject()) {
    Napi::Object opts = info[2].As<Napi::Object>();

    if (opts.Has("reasoningFormat") && opts.Get("reasoningFormat").IsNumber()) {
      reasoning_format = static_cast<common_reasoning_format>(
        opts.Get("reasoningFormat").As<Napi::Number>().Int32Value());
    }
    if (opts.Has("isPartial") && opts.Get("isPartial").IsBoolean()) {
      is_partial = opts.Get("isPartial").As<Napi::Boolean>().Value();
    }
    if (opts.Has("thinkingForcedOpen") && opts.Get("thinkingForcedOpen").IsBoolean()) {
      thinking_forced_open = opts.Get("thinkingForcedOpen").As<Napi::Boolean>().Value();
    }
    if (opts.Has("parser") && opts.Get("parser").IsString()) {
      parser_data = opts.Get("parser").As<Napi::String>().Utf8Value();
    }
  }

  // Synchronous — parsing is fast, no I/O
  auto result = lloyal::chat_out::parse(output, format, reasoning_format,
                                         is_partial, thinking_forced_open, parser_data);

  // Build return object
  Napi::Object obj = Napi::Object::New(env);
  obj.Set("content", Napi::String::New(env, result.content));
  obj.Set("reasoningContent", Napi::String::New(env, result.reasoning_content));

  Napi::Array toolCalls = Napi::Array::New(env, result.tool_calls.size());
  for (size_t i = 0; i < result.tool_calls.size(); i++) {
    Napi::Object tc = Napi::Object::New(env);
    tc.Set("name", Napi::String::New(env, result.tool_calls[i].name));
    tc.Set("arguments", Napi::String::New(env, result.tool_calls[i].arguments));
    tc.Set("id", Napi::String::New(env, result.tool_calls[i].id));
    toolCalls[i] = tc;
  }
  obj.Set("toolCalls", toolCalls);

  return obj;
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

  // Extract typeK (optional, default F16)
  ggml_type typeK = GGML_TYPE_F16;
  if (options.Has("typeK") && options.Get("typeK").IsString()) {
    std::string s = options.Get("typeK").As<Napi::String>().Utf8Value();
    ggml_type t = lloyal::kv::cache_type::from_str(s);
    if (t == GGML_TYPE_COUNT) {
      throw Napi::Error::New(env, "Unsupported typeK: " + s);
    }
    typeK = t;
  }

  // Extract typeV (optional, default F16)
  ggml_type typeV = GGML_TYPE_F16;
  if (options.Has("typeV") && options.Get("typeV").IsString()) {
    std::string s = options.Get("typeV").As<Napi::String>().Utf8Value();
    ggml_type t = lloyal::kv::cache_type::from_str(s);
    if (t == GGML_TYPE_COUNT) {
      throw Napi::Error::New(env, "Unsupported typeV: " + s);
    }
    typeV = t;
  }

  // Ensure llama backend is initialized on main thread (thread-safe, once)
  BackendManager::ensureInitialized();

  // Normalize and validate path BEFORE queuing async work
  std::string fsPath = liblloyal_node::FileSystem::normalizePath(modelPath);
  if (fsPath != modelPath) {
    std::cerr << "[CreateContext] Normalized " << modelPath << " → " << fsPath << std::endl;
  }

  if (!liblloyal_node::FileSystem::exists(fsPath)) {
    std::cerr << "[CreateContext] File does not exist: " << fsPath << std::endl;
    throw Napi::Error::New(env, "Model file not found: " + fsPath);
  }

  size_t fileSize = liblloyal_node::FileSystem::getSize(fsPath);
  std::cerr << "[CreateContext] File validated: " << fsPath << " (" << fileSize << " bytes)" << std::endl;

  // Load model on main thread
  std::cerr << "[CreateContext] Loading model..." << std::endl;

  llama_model_params model_params = llama_model_default_params();
  // -1 = offload all layers to GPU (auto-detect), 0 = CPU only
  model_params.n_gpu_layers = -1;

  std::cerr << "[CreateContext] Acquiring from ModelRegistry..." << std::endl;
  auto sharedModel = lloyal::ModelRegistry::acquire(fsPath, model_params);

  if (!sharedModel) {
    throw Napi::Error::New(env, "Failed to load model from " + fsPath);
  }

  std::cerr << "[CreateContext] Model loaded (refcount: " << sharedModel.use_count() << ")" << std::endl;

  // Create context
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = static_cast<uint32_t>(nCtx);
  ctx_params.n_batch = static_cast<uint32_t>(nBatch);
  ctx_params.n_ubatch = static_cast<uint32_t>(nBatch);
  ctx_params.n_threads = static_cast<uint32_t>(nThreads);
  ctx_params.n_seq_max = static_cast<uint32_t>(nSeqMax);
  ctx_params.type_k = typeK;
  ctx_params.type_v = typeV;
  ctx_params.kv_unified = true;  // Share KV across sequences (efficient for branching)

  // Apply embedding-specific params
  ctx_params.embeddings = embeddingsMode;
  ctx_params.pooling_type = static_cast<enum llama_pooling_type>(poolingType);

  std::cerr << "[CreateContext] Creating context (embeddings=" << embeddingsMode
            << ", pooling=" << poolingType << ")..." << std::endl;
  llama_context* ctx = llama_init_from_model(sharedModel.get(), ctx_params);

  if (!ctx) {
    throw Napi::Error::New(env, "Failed to create context");
  }

  std::cerr << "[CreateContext] Context created successfully" << std::endl;

  // Create SessionContext instance
  Napi::Function ctor = env.GetInstanceData<Napi::FunctionReference>()->Value();
  Napi::Object instance = ctor.New({});
  SessionContext* obj = SessionContext::Unwrap(instance);

  // Initialize
  obj->initializeContext(std::move(sharedModel), ctx, nBatch);

  std::cerr << "[CreateContext] SessionContext initialized" << std::endl;
  return instance;
}

// ===== BRANCH API IMPLEMENTATION =====

Napi::Value SessionContext::_branchCreate(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchCreate requires (position[, params[, nBatch[, grammar]]])");
  }

  auto position = static_cast<llama_pos>(info[0].As<Napi::Number>().Int32Value());

  // Extract sampling params from JS object (optional 2nd arg)
  LloyalSamplingParams params;
  if (info.Length() >= 2 && info[1].IsObject()) {
    params = adaptSamplingParamsFromJS(info[1].As<Napi::Object>());
  }

  // Per-branch nBatch override (optional 3rd arg), falls back to context default
  int32_t nBatch = _nBatch;
  if (info.Length() >= 3 && info[2].IsNumber()) {
    nBatch = info[2].As<Napi::Number>().Int32Value();
  }

  // Grammar string (optional 4th arg)
  const char* grammar_str = nullptr;
  std::string grammar_storage;
  if (info.Length() >= 4 && info[3].IsString()) {
    grammar_storage = info[3].As<Napi::String>().Utf8Value();
    grammar_str = grammar_storage.c_str();
  }

  // Create branch — seq_id allocated internally by tenancy
  auto handle = lloyal::branch::create(
    _context,
    _model.get(),
    _branchStore,
    position,
    params,
    nBatch,
    grammar_str
  );

  if (handle == lloyal::branch::INVALID_HANDLE) {
    throw Napi::Error::New(env, "Failed to create branch");
  }

  return Napi::Number::New(env, handle);
}

Napi::Value SessionContext::_branchFork(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchFork requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());

  auto newHandle = lloyal::branch::fork(handle, _branchStore);

  if (newHandle == lloyal::branch::INVALID_HANDLE) {
    throw Napi::Error::New(env, "Failed to fork branch");
  }

  return Napi::Number::New(env, newHandle);
}

// Bulk-decode tokens into a branch's KV cache and capture final logits.
// Wrapped by Branch.prefill() on the JS side.
Napi::Value SessionContext::_branchPrefill(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsArray()) {
    throw Napi::Error::New(env, "_branchPrefill requires (handle, tokens[])");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());

  Napi::Array jsTokens = info[1].As<Napi::Array>();
  std::vector<llama_token> tokens;
  tokens.reserve(jsTokens.Length());
  for (uint32_t i = 0; i < jsTokens.Length(); i++) {
    tokens.push_back(static_cast<llama_token>(jsTokens.Get(i).As<Napi::Number>().Int32Value()));
  }

  if (tokens.empty()) {
    auto deferred = Napi::Promise::Deferred::New(env);
    deferred.Resolve(env.Undefined());
    return deferred.Promise();
  }

  auto* worker = new BranchPrefillWorker(env, _branchStore, handle, std::move(tokens));
  worker->Queue();
  return worker->GetPromise();
}

Napi::Value SessionContext::_branchSample(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchSample requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  auto token = lloyal::branch::sample(handle, _branchStore);

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

  lloyal::branch::accept_token(handle, token, _branchStore);

  return env.Undefined();
}


Napi::Value SessionContext::_branchGetPosition(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchGetPosition requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  auto position = lloyal::branch::get_position(handle, _branchStore);

  return Napi::Number::New(env, position);
}

Napi::Value SessionContext::_branchGetPerplexity(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchGetPerplexity requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  auto ppl = lloyal::branch::get_perplexity(handle, _branchStore);

  return Napi::Number::New(env, ppl);
}

Napi::Value SessionContext::_branchGetLogits(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchGetLogits requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  const float* logits = lloyal::branch::get_logits(handle, _branchStore);

  if (!logits) {
    throw Napi::Error::New(env, "_branchGetLogits: no logits captured");
  }

  int n_vocab = lloyal::branch::get_n_vocab(handle, _branchStore);
  Napi::Float32Array result = Napi::Float32Array::New(env, n_vocab);
  std::memcpy(result.Data(), logits, n_vocab * sizeof(float));

  return result;
}

Napi::Value SessionContext::_branchPrune(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchPrune requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  lloyal::branch::prune(handle, _branchStore);

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

  // Only reseed stochastic chains (has_dist=true)
  // Reseeding greedy chains would corrupt them
  if (state->sampler_chain != 0 && _branchStore.sampler_has_dist(state->sampler_chain)) {
    llama_sampler* chain = _branchStore.get_sampler_chain(state->sampler_chain);
    if (chain) lloyal::sampler::reseed_chain(chain, seed);
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
  }, _branchStore);

  return env.Undefined();
}

Napi::Value SessionContext::_branchClearSteer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchClearSteer requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());

  lloyal::branch::clear_steer(handle, _branchStore);

  return env.Undefined();
}

Napi::Value SessionContext::_branchSetSamplerParams(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2) {
    throw Napi::Error::New(env, "_branchSetSamplerParams requires (handle, params)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());

  LloyalSamplingParams params;
  if (info[1].IsObject()) {
    params = adaptSamplingParamsFromJS(info[1].As<Napi::Object>());
  }

  lloyal::branch::set_sampler_params(handle, params, _branchStore);

  return env.Undefined();
}

Napi::Value SessionContext::_branchSetGrammar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2) {
    throw Napi::Error::New(env, "_branchSetGrammar requires (handle, grammarStr)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());

  std::string grammar_str = info[1].As<Napi::String>().Utf8Value();

  lloyal::branch::set_grammar(
    handle,
    _model.get(),
    grammar_str.empty() ? "" : grammar_str.c_str(),
    _branchStore
  );

  return env.Undefined();
}

// ===== BRANCH METRICS & LOGIT BIAS =====

Napi::Value SessionContext::_branchModelEntropy(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::TypeError::New(env, "_branchModelEntropy requires (handle[, base])");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());

  std::string baseStr = "nats";
  if (info.Length() >= 2 && info[1].IsString()) {
    baseStr = info[1].As<Napi::String>().Utf8Value();
  }

  auto* state = _branchStore.get(handle);
  if (!state) {
    throw Napi::Error::New(env, "_branchModelEntropy: invalid handle");
  }
  if (!state->has_logits) {
    throw Napi::Error::New(env, "_branchModelEntropy: no logits captured (call prefill or commit first)");
  }

  float entropy = lloyal::metrics::model_entropy(
    state->logits_snapshot.data(), state->n_vocab, parseBase(baseStr));

  return Napi::Number::New(env, static_cast<double>(entropy));
}

Napi::Value SessionContext::_branchModelSurprisal(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2) {
    throw Napi::TypeError::New(env, "_branchModelSurprisal requires (handle, token[, base])");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  auto token = static_cast<int32_t>(info[1].As<Napi::Number>().Int32Value());

  std::string baseStr = "nats";
  if (info.Length() >= 3 && info[2].IsString()) {
    baseStr = info[2].As<Napi::String>().Utf8Value();
  }

  auto* state = _branchStore.get(handle);
  if (!state) {
    throw Napi::Error::New(env, "_branchModelSurprisal: invalid handle");
  }
  if (!state->has_logits) {
    throw Napi::Error::New(env, "_branchModelSurprisal: no logits captured (call prefill or commit first)");
  }

  float surprisal = lloyal::metrics::model_surprisal(
    state->logits_snapshot.data(), state->n_vocab, token, parseBase(baseStr));

  return Napi::Number::New(env, static_cast<double>(surprisal));
}

Napi::Value SessionContext::_branchGetSamplingPerplexity(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::TypeError::New(env, "_branchGetSamplingPerplexity requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  float ppl = lloyal::branch::get_sampling_perplexity(handle, _branchStore);

  return Napi::Number::New(env, static_cast<double>(ppl));
}

Napi::Value SessionContext::_branchSetLogitBias(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2) {
    throw Napi::TypeError::New(env, "_branchSetLogitBias requires (handle, biases[])");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());

  if (!info[1].IsArray()) {
    throw Napi::TypeError::New(env, "_branchSetLogitBias: biases must be an array");
  }

  Napi::Array biasArray = info[1].As<Napi::Array>();
  uint32_t length = biasArray.Length();

  std::vector<llama_logit_bias> biases;
  biases.reserve(length);

  for (uint32_t i = 0; i < length; i++) {
    Napi::Value item = biasArray[i];
    if (!item.IsObject()) {
      throw Napi::Error::New(env, "_branchSetLogitBias: each bias must be {token, bias}");
    }
    Napi::Object obj = item.As<Napi::Object>();

    if (!obj.Has("token") || !obj.Has("bias")) {
      throw Napi::Error::New(env, "_branchSetLogitBias: each bias must have 'token' and 'bias' properties");
    }

    llama_logit_bias bias;
    bias.token = static_cast<llama_token>(obj.Get("token").As<Napi::Number>().Int32Value());
    bias.bias = obj.Get("bias").As<Napi::Number>().FloatValue();
    biases.push_back(bias);
  }

  lloyal::branch::set_logit_bias(handle, biases.data(), biases.size(), _branchStore);

  return env.Undefined();
}

Napi::Value SessionContext::_branchClearLogitBias(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::TypeError::New(env, "_branchClearLogitBias requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  lloyal::branch::clear_logit_bias(handle, _branchStore);

  return env.Undefined();
}

// ===== STORE API =====

Napi::Value SessionContext::_storeCommit(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2 || !info[0].IsArray() || !info[1].IsArray()) {
    throw Napi::Error::New(env, "_storeCommit requires (handles[], tokens[])");
  }

  Napi::Array jsHandles = info[0].As<Napi::Array>();
  Napi::Array jsTokens = info[1].As<Napi::Array>();
  uint32_t n = jsHandles.Length();

  if (jsTokens.Length() != n) {
    throw Napi::Error::New(env, "_storeCommit: handles and tokens must have same length");
  }

  if (n == 0) {
    auto deferred = Napi::Promise::Deferred::New(env);
    deferred.Resolve(env.Undefined());
    return deferred.Promise();
  }

  std::vector<lloyal::branch::DecodeEachItem> items(n);

  for (uint32_t i = 0; i < n; i++) {
    items[i].handle = static_cast<lloyal::branch::BranchHandle>(
      jsHandles.Get(i).As<Napi::Number>().Uint32Value());
    items[i].token = static_cast<llama_token>(
      jsTokens.Get(i).As<Napi::Number>().Int32Value());
  }

  auto* worker = new StoreCommitWorker(env, _branchStore, std::move(items));
  worker->Queue();
  return worker->GetPromise();
}

Napi::Value SessionContext::_storePrefill(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 2 || !info[0].IsArray() || !info[1].IsArray()) {
    throw Napi::Error::New(env, "_storePrefill requires (handles[], tokenArrays[][])");
  }

  Napi::Array jsHandles = info[0].As<Napi::Array>();
  Napi::Array jsTokenArrays = info[1].As<Napi::Array>();
  uint32_t n = jsHandles.Length();

  if (jsTokenArrays.Length() != n) {
    throw Napi::Error::New(env, "_storePrefill: handles and tokenArrays must have same length");
  }

  if (n == 0) {
    auto deferred = Napi::Promise::Deferred::New(env);
    deferred.Resolve(env.Undefined());
    return deferred.Promise();
  }

  std::vector<lloyal::branch::BranchHandle> handles(n);
  std::vector<std::vector<llama_token>> tokenStorage(n);

  for (uint32_t i = 0; i < n; i++) {
    handles[i] = static_cast<lloyal::branch::BranchHandle>(
      jsHandles.Get(i).As<Napi::Number>().Uint32Value());

    Napi::Array jsArr = jsTokenArrays.Get(i).As<Napi::Array>();
    uint32_t len = jsArr.Length();
    tokenStorage[i].resize(len);
    for (uint32_t j = 0; j < len; j++) {
      tokenStorage[i][j] = static_cast<llama_token>(
        jsArr.Get(j).As<Napi::Number>().Int32Value());
    }
  }

  auto* worker = new StorePrefillWorker(env, _branchStore, std::move(handles), std::move(tokenStorage));
  worker->Queue();
  return worker->GetPromise();
}

Napi::Value SessionContext::_branchPruneSubtree(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchPruneSubtree requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  lloyal::branch::pruneSubtree(handle, _branchStore);

  return env.Undefined();
}

Napi::Value SessionContext::_branchParent(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchParent requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  auto parent = _branchStore.parent(handle);

  return Napi::Number::New(env, parent);
}

Napi::Value SessionContext::_branchChildren(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchChildren requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  const auto& children = _branchStore.children(handle);

  Napi::Array result = Napi::Array::New(env, children.size());
  for (size_t i = 0; i < children.size(); i++) {
    result.Set(static_cast<uint32_t>(i), Napi::Number::New(env, children[i]));
  }

  return result;
}

Napi::Value SessionContext::_branchIsLeaf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchIsLeaf requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());

  return Napi::Boolean::New(env, _branchStore.isLeaf(handle));
}

Napi::Value SessionContext::_branchIsActive(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_branchIsActive requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());

  return Napi::Boolean::New(env, _branchStore.isActive(handle));
}

Napi::Value SessionContext::_storeRetainOnly(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  if (info.Length() < 1) {
    throw Napi::Error::New(env, "_storeRetainOnly requires (handle)");
  }

  auto handle = static_cast<lloyal::branch::BranchHandle>(info[0].As<Napi::Number>().Uint32Value());
  _branchStore.retainOnly(handle);

  return env.Undefined();
}

Napi::Value SessionContext::_storeAvailable(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ensureNotDisposed();

  return Napi::Number::New(env, static_cast<double>(_branchStore.available()));
}

} // namespace liblloyal_node
