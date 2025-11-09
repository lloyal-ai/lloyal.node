#include "BackendManager.hpp"

namespace liblloyal_node {

// Static member definitions
std::once_flag BackendManager::init_flag_;
BackendManager* BackendManager::instance_ = nullptr;

} // namespace liblloyal_node
