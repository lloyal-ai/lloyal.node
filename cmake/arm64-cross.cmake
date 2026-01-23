# CMake toolchain file for Windows ARM64 cross-compilation
# Used by CI to build ARM64 binaries from x64 Windows runners

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR ARM64)

# Use clang-cl for cross-compilation (MSVC-compatible)
set(CMAKE_C_COMPILER clang-cl)
set(CMAKE_CXX_COMPILER clang-cl)

# Target ARM64 architecture
set(CMAKE_C_FLAGS_INIT "/arch:ARM64EC")
set(CMAKE_CXX_FLAGS_INIT "/arch:ARM64EC")

# Search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
