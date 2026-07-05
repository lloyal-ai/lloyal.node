#include "BackendManager.hpp"

#ifdef LLOYAL_BACKEND_DL
#include <ggml-backend.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#endif

namespace liblloyal_node {

// Static member definitions
std::once_flag BackendManager::init_flag_;
BackendManager* BackendManager::instance_ = nullptr;

#ifdef LLOYAL_BACKEND_DL
namespace {
// File-local anchor whose address dladdr resolves to THIS shared object
// (the addon .node file) — not a member function, whose pointer-to-void
// cast is UB-adjacent on some toolchains.
void backend_dir_anchor() {}
} // namespace

void BackendManager::resolveBackends() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&backend_dir_anchor), &info) == 0 ||
      info.dli_fname == nullptr) {
    // Loud, not silent: a DL build that can't find its own directory would
    // otherwise let llama_backend_init discover from exe-dir/cwd — the
    // wrong-search-path landmine this function exists to defuse.
    std::fprintf(stderr,
                 "[lloyal.node] FATAL: BACKEND_DL flavor could not resolve "
                 "its own module directory (dladdr failed)\n");
    std::abort();
  }
  const std::filesystem::path dir =
      std::filesystem::path(info.dli_fname).parent_path();
  std::fprintf(stderr, "[lloyal.node] Loading ggml backends from %s\n",
               dir.c_str());
  ggml_backend_load_all_from_path(dir.c_str());
}
#endif

} // namespace liblloyal_node
