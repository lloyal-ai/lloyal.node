#pragma once

#include <napi.h>

namespace liblloyal_node {

class Util {
public:
  static void Init(Napi::Env env, Napi::Object exports);

private:
  static Napi::Value ParseMarkdown(const Napi::CallbackInfo& info);
};

} // namespace liblloyal_node
