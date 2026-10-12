#include "ChatParser.hpp"
#include "Addon.hpp"

#include <atomic>

namespace {
using Arena = std::shared_ptr<const common_peg_arena>;
std::atomic<size_t> g_live{0};
}

Napi::Function ChatParser::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor = DefineClass(env, "ChatParser", {});
  exports.Set("ChatParser", ctor);
  exports.Set("liveChatParsers", Napi::Function::New(env, [](const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), static_cast<double>(live()));
  }));
  return ctor;
}

// The constructor takes the arena through an External the factory makes on
// the spot; a construction from JS has none and is refused.
ChatParser::ChatParser(const Napi::CallbackInfo& info) : Napi::ObjectWrap<ChatParser>(info) {
  if (info.Length() != 1 || !info[0].IsExternal()) {
    throw Napi::TypeError::New(info.Env(), "ChatParser: a parser is made by formatChat, never constructed");
  }
  _arena = *info[0].As<Napi::External<Arena>>().Data();
  ++g_live;
}

ChatParser::~ChatParser() {
  --g_live;
}

Napi::Object ChatParser::New(Napi::Env env, Arena arena) {
  Napi::Function ctor = env.GetInstanceData<AddonData>()->parserCtor.Value();
  return ctor.New({ Napi::External<Arena>::New(env, &arena) });
}

Arena ChatParser::arenaOf(Napi::Value value) {
  if (!value.IsObject()) return nullptr;
  Napi::Object object = value.As<Napi::Object>();
  Napi::Function ctor = value.Env().GetInstanceData<AddonData>()->parserCtor.Value();
  if (!object.InstanceOf(ctor)) return nullptr;
  return Unwrap(object)->_arena;
}

size_t ChatParser::live() {
  return g_live.load();
}
