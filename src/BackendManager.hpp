#pragma once

#include <llama/llama.h>
#include <mutex>
#include <iostream>

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
    std::cout << "[BackendManager] Initializing llama.cpp backend..." << std::endl;

    // Initialize llama backend (matches Nitro's LlamaBackendManager exactly)
    llama_backend_init();
    std::cout << "[BackendManager] llama_backend_init() called" << std::endl;

    // Match Nitro: Enable logging callback
    llama_log_set([](ggml_log_level level, const char* text, void* user_data) {
      const char* level_str = "";
      switch (level) {
        case GGML_LOG_LEVEL_ERROR: level_str = "ERROR"; break;
        case GGML_LOG_LEVEL_WARN:  level_str = "WARN"; break;
        case GGML_LOG_LEVEL_INFO:  level_str = "INFO"; break;
        case GGML_LOG_LEVEL_DEBUG: level_str = "DEBUG"; break;
        case GGML_LOG_LEVEL_NONE: level_str = "NONE"; break;
        case GGML_LOG_LEVEL_CONT: level_str = "CONT"; break;
      }
      std::cerr << "[llama.cpp " << level_str << "] " << text << std::flush;
    }, nullptr);

    std::cout << "[BackendManager] llama.cpp logging configured" << std::endl;
  }

  /**
   * Destructor cleans up backend
   * Called automatically on program termination
   */
  ~BackendManager() {
    llama_backend_free();
    std::cout << "[~BackendManager] llama_backend_free() called" << std::endl;
  }

  // Delete copy/move
  BackendManager(const BackendManager&) = delete;
  BackendManager& operator=(const BackendManager&) = delete;

  // Singleton state
  static std::once_flag init_flag_;
  static BackendManager* instance_;
};

} // namespace liblloyal_node
