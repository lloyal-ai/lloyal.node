cmake_minimum_required(VERSION 3.18)
project(lloyal_node)

# =============================================================================
# lloyal.node - N-API bindings for liblloyal + llama.cpp
# =============================================================================
#
# This CMakeLists.txt builds the native Node.js addon using cmake-js.
# It uses add_subdirectory() to build llama.cpp and liblloyal as part of
# a single CMake invocation, ensuring GPU library paths propagate correctly.
#
# Usage:
#   npx cmake-js compile                    # CPU build
#   npx cmake-js compile --CDGGML_CUDA=ON   # CUDA build
#   npx cmake-js compile --CDGGML_VULKAN=ON # Vulkan build
#   npx cmake-js compile --CDGGML_METAL=ON  # Metal build (macOS)
#

# =============================================================================
# C++ Standard
# =============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# cmake-js Integration
# =============================================================================

# cmake-js provides these variables:
#   CMAKE_JS_INC  - Node.js include directories
#   CMAKE_JS_SRC  - Additional source files (if any)
#   CMAKE_JS_LIB  - Node.js library to link against

add_definitions(-DNAPI_VERSION=8)

# Find node-addon-api include directory
# node-addon-api provides napi.h which wraps the raw N-API
execute_process(
    COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
message(STATUS "lloyal.node: node-addon-api include: ${NODE_ADDON_API_DIR}")

# =============================================================================
# Dependencies (Git Submodules)
# =============================================================================

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")
set(LIBLLOYAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/liblloyal")

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "lloyal.node: llama.cpp submodule not found. Run 'git submodule update --init --recursive'")
endif()

if(NOT EXISTS "${LIBLLOYAL_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "lloyal.node: liblloyal submodule not found. Run 'git submodule update --init --recursive'")
endif()

message(STATUS "lloyal.node: Using llama.cpp from ${LLAMA_CPP_DIR}")
message(STATUS "lloyal.node: Using liblloyal from ${LIBLLOYAL_DIR}")

# Build llama.cpp (creates llama, ggml targets with GPU support if enabled)
# GPU backends are controlled via cmake-js flags:
#   --CDGGML_CUDA=ON    - NVIDIA CUDA
#   --CDGGML_VULKAN=ON  - AMD/Intel Vulkan
#   --CDGGML_METAL=ON   - Apple Metal

# Windows: Force static library build to avoid DLL initialization issues and CRT
# heap mismatches that cause silent crashes. This matches the llama.node reference.
if(WIN32)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
endif()

# Enable llama.cpp common library (normally only built for standalone)
# This provides chat template processing, batch utilities, JSON schema to grammar, etc.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common library" FORCE)

# Set RPATH globally BEFORE adding llama.cpp subdirectory
# This ensures ALL shared libraries (llama, ggml, ggml-*, gpu backends, etc.)
# inherit these settings automatically - no need to hardcode target names
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path" CACHE STRING "RPATH for macOS" FORCE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE CACHE BOOL "Use install RPATH during build" FORCE)
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN" CACHE STRING "RPATH for Linux" FORCE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE CACHE BOOL "Use install RPATH during build" FORCE)
endif()

add_subdirectory(${LLAMA_CPP_DIR} llama.cpp)

# MSVC C++20 char8_t workaround: llama.cpp uses u8"" literals which MSVC treats
# as char8_t[] in C++20, but then passes to functions expecting const char*.
# Apply /Zc:char8_t- ONLY to llama.cpp targets, not globally (would break nlohmann/json).
if(MSVC)
    # Apply to all llama.cpp library targets
    if(TARGET llama)
        target_compile_options(llama PRIVATE /Zc:char8_t-)
    endif()
    if(TARGET ggml)
        target_compile_options(ggml PRIVATE /Zc:char8_t-)
    endif()
    if(TARGET ggml-base)
        target_compile_options(ggml-base PRIVATE /Zc:char8_t-)
    endif()
    if(TARGET ggml-cpu)
        target_compile_options(ggml-cpu PRIVATE /Zc:char8_t-)
    endif()
endif()

# Build liblloyal (INTERFACE library, links llama transitively)
# This also sets up the llama/llama.h include structure automatically
add_subdirectory(${LIBLLOYAL_DIR} liblloyal)

# =============================================================================
# Addon Sources
# =============================================================================

set(ADDON_SOURCES
    src/binding.cpp
    src/BackendManager.cpp
    src/SessionContext.cpp
)

# =============================================================================
# Create N-API Addon Module
# =============================================================================

add_library(${PROJECT_NAME} MODULE ${ADDON_SOURCES} ${CMAKE_JS_SRC})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_JS_INC}
    ${NODE_ADDON_API_DIR}
    src
)

# Link libraries
# liblloyal::liblloyal transitively brings:
#   - llama (and ggml, and GPU deps like CUDA::cudart)
#   - The llama/llama.h include structure we created
# common is llama.cpp's common library (chat templates, batch utils, etc.)
target_link_libraries(${PROJECT_NAME} PRIVATE
    liblloyal::liblloyal
    common
    ${CMAKE_JS_LIB}
)

# =============================================================================
# Node Addon Properties
# =============================================================================

set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
    OUTPUT_NAME "lloyal"
)

# =============================================================================
# Platform-Specific Configuration
# =============================================================================

if(APPLE)
    # macOS: Allow undefined symbols (N-API) to be resolved at runtime by Node.js
    # cmake-js sets CMAKE_SHARED_LINKER_FLAGS, but MODULE libraries need this in
    # CMAKE_MODULE_LINKER_FLAGS or directly on the target
    target_link_options(${PROJECT_NAME} PRIVATE -undefined dynamic_lookup)
    
    # macOS frameworks
    target_link_libraries(${PROJECT_NAME} PRIVATE
        "-framework Foundation"
        "-framework Accelerate"
    )
    
    # Metal framework when GPU enabled
    if(GGML_METAL)
        target_link_libraries(${PROJECT_NAME} PRIVATE "-framework Metal")
    endif()
    
    # RPATH for finding libllama.dylib at runtime
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    
elseif(UNIX)
    # Linux RPATH
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    
elseif(MSVC)
    # Windows: ensure .node (MODULE) and DLLs (RUNTIME) are in correct directory
    # Disable per-config subdirectories to avoid build/Release/Release nesting
    set_target_properties(${PROJECT_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Release
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Release
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
    )
endif()

# =============================================================================
# Post-Build: Copy Shared Libraries
# =============================================================================
# Copy llama.cpp shared libraries to the same directory as the .node file.
# This is required because RPATH is set to @loader_path / $ORIGIN (same directory).
#
# llama.cpp sets SOVERSION on all shared targets, producing versioned filenames:
#   macOS: libllama.0.0.X.dylib (real), libllama.0.dylib (soname), libllama.dylib (linker)
#   Linux: libllama.so.0.0.X (real), libllama.so.0 (soname), libllama.so (linker)
# The binary's load command references the SONAME (e.g., @rpath/libllama.0.dylib),
# so all three variants must be present for runtime loading to succeed.
#
# Note: On Windows, BUILD_SHARED_LIBS=OFF (static linking), so this section is skipped.

if(BUILD_SHARED_LIBS)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:llama>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_SONAME_FILE:llama>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_LINKER_FILE:llama>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMENT "Copying libllama to output directory"
    )

    # Copy ggml shared libraries
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:ggml>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_SONAME_FILE:ggml>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_LINKER_FILE:ggml>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMENT "Copying libggml to output directory"
    )

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:ggml-base>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_SONAME_FILE:ggml-base>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_LINKER_FILE:ggml-base>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMENT "Copying libggml-base to output directory"
    )

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:ggml-cpu>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_SONAME_FILE:ggml-cpu>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_LINKER_FILE:ggml-cpu>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMENT "Copying libggml-cpu to output directory"
    )

    # Copy GPU backend libraries if built
    if(TARGET ggml-metal)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:ggml-metal>
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_SONAME_FILE:ggml-metal>
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_LINKER_FILE:ggml-metal>
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMENT "Copying libggml-metal to output directory"
        )
    endif()

    if(TARGET ggml-cuda)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:ggml-cuda>
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_SONAME_FILE:ggml-cuda>
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_LINKER_FILE:ggml-cuda>
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMENT "Copying libggml-cuda to output directory"
        )
    endif()

    if(TARGET ggml-vulkan)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:ggml-vulkan>
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_SONAME_FILE:ggml-vulkan>
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_LINKER_FILE:ggml-vulkan>
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMENT "Copying libggml-vulkan to output directory"
        )
    endif()

    if(TARGET ggml-blas)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:ggml-blas>
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_SONAME_FILE:ggml-blas>
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_LINKER_FILE:ggml-blas>
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMENT "Copying libggml-blas to output directory"
        )
    endif()
endif()

# =============================================================================
# Status Messages
# =============================================================================

message(STATUS "lloyal.node configured:")
message(STATUS "  llama.cpp: ${LLAMA_CPP_DIR}")
message(STATUS "  liblloyal: ${LIBLLOYAL_DIR}")
message(STATUS "  C++ Standard: C++20")
if(GGML_CUDA)
    message(STATUS "  GPU Backend: CUDA")
elseif(GGML_VULKAN)
    message(STATUS "  GPU Backend: Vulkan")
elseif(GGML_METAL)
    message(STATUS "  GPU Backend: Metal")
else()
    message(STATUS "  GPU Backend: None (CPU only)")
endif()
