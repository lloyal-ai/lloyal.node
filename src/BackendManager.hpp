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
   */
  BackendManager() {
    llama_backend_init();
    common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
    llama_log_set(common_log_default_callback, nullptr);
  }

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
