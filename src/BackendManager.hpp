#pragma once

#include <llama/llama.h>
#include "log.h"
#include <mutex>

namespace liblloyal_node {

/**
 * RAII guard to manage the global llama.cpp backend lifecycle
 *
 * Thread-safe singleton using std::call_once.
 * Ensures llama_backend_init() is called once on first use and
 * llama_backend_free() is called on program termination.
 *
 * Pattern matches LlamaBackendManager from nitro-llama.
 */
class BackendManager {
public:
  /**
   * Ensure the global llama.cpp backend is initialized
   * Safe to call multiple times from multiple threads
   */
  static void ensureInitialized() {
    std::call_once(init_flag_, [] {
      instance_ = new BackendManager();
    });
  }

private:
  /**
   * Private constructor - initializes backend and logging
   * Called exactly once by ensureInitialized()
   *
   * BACKEND_DL flavor: resolveBackends() MUST run before
   * llama_backend_init() — init auto-triggers ggml's backend discovery
   * over the WRONG search paths (the node executable's dir + cwd) whenever
   * the registry is empty. Loading from the addon's own directory first
   * leaves the registry populated, so init performs no discovery of its
   * own. std::call_once bakes the module set once per process, which is
   * what keeps ModelRegistry's (path, n_gpu_layers, use_mmap) cache key
   * sufficient — backends are immutable for the process lifetime.
   */
  BackendManager() {
#ifdef LLOYAL_BACKEND_DL
    resolveBackends();
#endif
    llama_backend_init();
    common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
    llama_log_set(common_log_default_callback, nullptr);
  }

#ifdef LLOYAL_BACKEND_DL
  /**
   * dlopen + score every backend module sitting BESIDE this addon binary
   * (dladdr self-location). The pack ships modules next to lloyal.node, so
   * "selection = which addon you require" — zero JS→native plumbing.
   * Implemented in BackendManager.cpp.
   */
  static void resolveBackends();
#endif

  /**
   * Destructor cleans up backend
   * Called automatically on program termination
   */
  ~BackendManager() {
    llama_backend_free();
  }

  // Delete copy/move
  BackendManager(const BackendManager&) = delete;
  BackendManager& operator=(const BackendManager&) = delete;

  // Singleton state
  static std::once_flag init_flag_;
  static BackendManager* instance_;
};

} // namespace liblloyal_node
