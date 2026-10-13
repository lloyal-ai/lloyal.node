#include <napi.h>
#include "Addon.hpp"
#include "ChatParser.hpp"
#include "SessionContext.hpp"
#include "Util.hpp"
#include <llama/llama.h>

/**
 * N-API Module Entry Point
 *
 * Exports:
 *   - createContext(options): Factory function to create SessionContext
 *   - SessionContext: Class for inference operations
 *   - ChatParser: the parser handle formatChat returns; liveChatParsers(): how many are alive
 */

namespace liblloyal_node {

// Module initialization
Napi::Object Init(Napi::Env env, Napi::Object exports) {
  // The classes, and their constructors kept for the factories that call them.
  Napi::Function contextCtor = SessionContext::Init(env, exports).Get("SessionContext").As<Napi::Function>();
  Napi::Function parserCtor = ChatParser::Init(env, exports);
  env.SetInstanceData(new AddonData{ Napi::Persistent(contextCtor), Napi::Persistent(parserCtor) });

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
