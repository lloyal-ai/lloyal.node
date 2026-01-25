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
# Vendor Dependencies
# =============================================================================

# Determine whether to use submodules (development) or vendor/ (npm install)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/CMakeLists.txt")
    set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp")
    set(LIBLLOYAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/liblloyal")
    message(STATUS "lloyal.node: Using vendored dependencies")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/CMakeLists.txt")
    set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")
    set(LIBLLOYAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/liblloyal")
    message(STATUS "lloyal.node: Using submodule dependencies")
else()
    message(FATAL_ERROR "lloyal.node: No llama.cpp found. Run 'git submodule update --init' or 'npm run update-vendors'")
endif()

# Build llama.cpp (creates llama, ggml targets with GPU support if enabled)
# GPU backends are controlled via cmake-js flags:
#   --CDGGML_CUDA=ON    - NVIDIA CUDA
#   --CDGGML_VULKAN=ON  - AMD/Intel Vulkan
#   --CDGGML_METAL=ON   - Apple Metal
add_subdirectory(${LLAMA_CPP_DIR} llama.cpp)

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

add_library(${PROJECT_NAME} SHARED ${ADDON_SOURCES} ${CMAKE_JS_SRC})

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
target_link_libraries(${PROJECT_NAME} PRIVATE
    liblloyal::liblloyal
    ${CMAKE_JS_LIB}
)

# =============================================================================
# Node Addon Properties
# =============================================================================

set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
    OUTPUT_NAME "lloyal"
    CXX_STANDARD 20
)

# =============================================================================
# Platform-Specific Configuration
# =============================================================================

if(APPLE)
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
    # Windows: ensure DLLs are in same directory as .node
    set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Release
    )
endif()

# =============================================================================
# Post-Build: Copy Shared Libraries
# =============================================================================
# Copy llama.cpp shared libraries to the same directory as the .node file
# This is required because RPATH is set to @loader_path (same directory)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:llama>
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying libllama to output directory"
)

# Copy ggml shared libraries
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:ggml>
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying libggml to output directory"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:ggml-base>
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying libggml-base to output directory"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:ggml-cpu>
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying libggml-cpu to output directory"
)

# Copy GPU backend libraries if built
if(TARGET ggml-metal)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:ggml-metal>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMENT "Copying libggml-metal to output directory"
    )
endif()

if(TARGET ggml-cuda)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:ggml-cuda>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMENT "Copying libggml-cuda to output directory"
    )
endif()

if(TARGET ggml-vulkan)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:ggml-vulkan>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMENT "Copying libggml-vulkan to output directory"
    )
endif()

if(TARGET ggml-blas)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:ggml-blas>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMENT "Copying libggml-blas to output directory"
    )
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
