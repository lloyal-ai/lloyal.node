#include <napi.h>
#include "SessionContext.hpp"
#include "Util.hpp"
#include <llama/llama.h>

/**
 * N-API Module Entry Point
 *
 * Exports:
 *   - createContext(options): Factory function to create SessionContext
 *   - SessionContext: Class for inference operations
 */

namespace liblloyal_node {

// Module initialization
Napi::Object Init(Napi::Env env, Napi::Object exports) {
  // Initialize SessionContext class and get constructor
  Napi::Object ctorObj = SessionContext::Init(env, exports);
  Napi::Function ctorFunc = ctorObj.Get("SessionContext").As<Napi::Function>();

  // Store constructor for CreateContext to use
  env.SetInstanceData(new Napi::FunctionReference(Napi::Persistent(ctorFunc)));

  // Export factory function
  exports.Set("createContext", Napi::Function::New(env, CreateContext));

  // Export utility functions (parseMarkdown, etc.)
  Util::Init(env, exports);

  return exports;
}

} // namespace liblloyal_node

// Wrapper function for module registration
Napi::Object InitModule(Napi::Env env, Napi::Object exports) {
  return liblloyal_node::Init(env, exports);
}

// Register the N-API module
NODE_API_MODULE(liblloyal_node, InitModule)
