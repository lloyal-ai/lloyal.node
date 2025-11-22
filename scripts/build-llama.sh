#!/usr/bin/env bash
#
# Build llama.cpp for the current platform
# Called automatically by npm prepare
#
# Respects LLOYAL_GPU environment variable:
#   cuda   - Enable CUDA (Linux/Windows)
#   vulkan - Enable Vulkan (Linux/Windows)
#   metal  - Enable Metal (macOS)
#   cpu    - Force CPU only
#   (unset)- Auto-detect (Metal on macOS local, CPU on Linux/Windows)

set -e

# Resolve llama.cpp source directory (vendor/ or submodule)
if [ -d "vendor/llama.cpp" ]; then
    LLAMA_DIR="vendor/llama.cpp"
    echo "Using vendored llama.cpp from vendor/"
elif [ -d "llama.cpp" ]; then
    LLAMA_DIR="llama.cpp"
    echo "Using llama.cpp submodule"
else
    echo "Error: llama.cpp not found in vendor/ or as submodule"
    echo "Run: npm run update-vendors"
    exit 1
fi

# --- GPU Selection Logic ---
GPU_BACKEND="cpu"
if [ -n "$LLOYAL_GPU" ]; then
    GPU_BACKEND=$(echo "$LLOYAL_GPU" | tr '[:upper:]' '[:lower:]')
    echo "Explicit GPU backend requested: $GPU_BACKEND"
else
    # Default behavior
    if [[ "$OSTYPE" == "darwin"* ]] && [ -z "$CI" ]; then
        GPU_BACKEND="metal"
        echo "Local macOS detected, defaulting to Metal"
    else
        echo "No backend specified, defaulting to CPU"
    fi
fi

# --- Platform Validation ---
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ "$GPU_BACKEND" == "cuda" ]] || [[ "$GPU_BACKEND" == "vulkan" ]]; then
        echo "Error: $GPU_BACKEND is not supported on macOS"
        exit 1
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [[ "$GPU_BACKEND" == "metal" ]]; then
        echo "Warning: Metal only supported on macOS, falling back to CPU"
        GPU_BACKEND="cpu"
    fi
fi

CMAKE_ARGS=""
# Common CMake Args
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release"
CMAKE_ARGS="$CMAKE_ARGS -DLLAMA_BUILD_EXAMPLES=OFF"
CMAKE_ARGS="$CMAKE_ARGS -DLLAMA_BUILD_TOOLS=OFF"
CMAKE_ARGS="$CMAKE_ARGS -DLLAMA_BUILD_TESTS=OFF"
CMAKE_ARGS="$CMAKE_ARGS -DLLAMA_BUILD_SERVER=OFF"
CMAKE_ARGS="$CMAKE_ARGS -DLLAMA_BUILD_COMMON=OFF"
CMAKE_ARGS="$CMAKE_ARGS -DLLAMA_CURL=OFF"

# Apply Backend Flags
METAL_ENABLED="OFF"
case "$GPU_BACKEND" in
  cuda)
    echo "Enabling CUDA..."
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON"
    ;;
  vulkan)
    echo "Enabling Vulkan..."
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_VULKAN=ON"
    ;;
  metal)
    echo "Enabling Metal..."
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_METAL=ON"
    METAL_ENABLED="ON"
    ;;
  cpu)
    echo "Building for CPU only..."
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF"
    ;;
  *)
    echo "Warning: Unknown backend '$GPU_BACKEND', defaulting to CPU"
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF"
    ;;
esac

# --- Platform Specific Builds ---

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Building llama.cpp for Linux..."
    BUILD_DIR="$LLAMA_DIR/build-linux"

    # Linux Strategy: Build static libs with PIC, then combine into single .so
    # This avoids ODR violations and ensures a single artifact for N-API

    cmake -B "$BUILD_DIR" -S "$LLAMA_DIR" \
        $CMAKE_ARGS \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON

    cmake --build "$BUILD_DIR" --config Release -j$(nproc)

    # Combine into single shared library
    echo "Combining static libraries into libllama.so..."

    # Create MRI script for ar
    MRI_SCRIPT="$BUILD_DIR/libllama.mri"
    echo "create $BUILD_DIR/libllama-combined.a" > "$MRI_SCRIPT"
    echo "addlib $BUILD_DIR/src/libllama.a" >> "$MRI_SCRIPT"
    echo "addlib $BUILD_DIR/ggml/src/libggml.a" >> "$MRI_SCRIPT"
    echo "addlib $BUILD_DIR/ggml/src/libggml-base.a" >> "$MRI_SCRIPT"
    echo "addlib $BUILD_DIR/ggml/src/libggml-cpu.a" >> "$MRI_SCRIPT"

    if [ "$GPU_BACKEND" == "cuda" ]; then
        if [ -f "$BUILD_DIR/ggml/src/ggml-cuda/libggml-cuda.a" ]; then
             echo "addlib $BUILD_DIR/ggml/src/ggml-cuda/libggml-cuda.a" >> "$MRI_SCRIPT"
        fi
    fi

    if [ "$GPU_BACKEND" == "vulkan" ]; then
        if [ -f "$BUILD_DIR/ggml/src/ggml-vulkan/libggml-vulkan.a" ]; then
             echo "addlib $BUILD_DIR/ggml/src/ggml-vulkan/libggml-vulkan.a" >> "$MRI_SCRIPT"
        fi
    fi

    # Always check for blas
    if [ -f "$BUILD_DIR/ggml/src/ggml-blas/libggml-blas.a" ]; then
        echo "addlib $BUILD_DIR/ggml/src/ggml-blas/libggml-blas.a" >> "$MRI_SCRIPT"
    fi

    echo "save" >> "$MRI_SCRIPT"
    echo "end" >> "$MRI_SCRIPT"
    ar -M < "$MRI_SCRIPT"

    # Link into shared object with all required libraries
    EXTRA_LIBS=""
    if [ "$GPU_BACKEND" == "cuda" ]; then
        # CUDA runtime libraries (dynamically linked at runtime)
        EXTRA_LIBS="-lcudart -lcublas -lcublasLt"
        echo "Adding CUDA runtime libraries to link command"
    elif [ "$GPU_BACKEND" == "vulkan" ]; then
        # Vulkan loader (dynamically linked at runtime)
        EXTRA_LIBS="-lvulkan"
        echo "Adding Vulkan libraries to link command"
    fi

    g++ -shared -fPIC -fopenmp \
        -Wl,--whole-archive "$BUILD_DIR/libllama-combined.a" -Wl,--no-whole-archive \
        -o "$BUILD_DIR/libllama.so" \
        -lpthread -ldl -lm -lstdc++ $EXTRA_LIBS

    echo "✓ Created libllama.so"

    # Copy .so to build/Release/ for node-gyp
    RELEASE_DIR="build/Release"
    mkdir -p "$RELEASE_DIR"
    cp "$BUILD_DIR/libllama.so" "$RELEASE_DIR/libllama.so"
    echo "✓ Copied libllama.so to $RELEASE_DIR/"

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Building llama.cpp for macOS..."
    BUILD_DIR="$LLAMA_DIR/build-apple"

    # Add BLAS for Accelerate framework support
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_BLAS=ON"

    cmake -B "$BUILD_DIR" -S "$LLAMA_DIR" \
        $CMAKE_ARGS \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"

    cmake --build "$BUILD_DIR" --config Release -j$(sysctl -n hw.ncpu)

    echo "Combining static libraries into single shared library..."

    # List of libs to combine
    LIBS_TO_COMBINE=(
        "$BUILD_DIR/src/libllama.a"
        "$BUILD_DIR/ggml/src/libggml.a"
        "$BUILD_DIR/ggml/src/libggml-base.a"
        "$BUILD_DIR/ggml/src/libggml-cpu.a"
        "$BUILD_DIR/ggml/src/ggml-blas/libggml-blas.a"
    )

    if [ "$METAL_ENABLED" = "ON" ]; then
        LIBS_TO_COMBINE+=("$BUILD_DIR/ggml/src/ggml-metal/libggml-metal.a")
    fi

    libtool -static -o "$BUILD_DIR/libllama-combined.a" "${LIBS_TO_COMBINE[@]}"

    # Create single shared library with proper install_name
    FRAMEWORKS="-framework Foundation -framework Accelerate"
    if [ "$METAL_ENABLED" = "ON" ]; then
        FRAMEWORKS="$FRAMEWORKS -framework Metal"
    fi

    clang++ -dynamiclib \
        -arch arm64 -arch x86_64 \
        -Wl,-force_load,"$BUILD_DIR/libllama-combined.a" \
        $FRAMEWORKS \
        -install_name "@rpath/libllama.dylib" \
        -o "$BUILD_DIR/libllama.dylib"

    echo "✓ Combined shared library created: $BUILD_DIR/libllama.dylib"

    # Copy .dylib to build/Release/
    RELEASE_DIR="build/Release"
    mkdir -p "$RELEASE_DIR"
    cp "$BUILD_DIR/libllama.dylib" "$RELEASE_DIR/libllama.dylib"
    echo "✓ Copied libllama.dylib to $RELEASE_DIR/"

elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]] || [[ "$OSTYPE" == "win32"* ]]; then
    echo "Building llama.cpp for Windows..."
    BUILD_DIR="$LLAMA_DIR/build-windows"

    CMAKE_ARGS="$CMAKE_ARGS -DBUILD_SHARED_LIBS=ON"

    cmake -B "$BUILD_DIR" -S "$LLAMA_DIR" $CMAKE_ARGS
    cmake --build "$BUILD_DIR" --config Release

    # Copy DLLs and import library
    RELEASE_DIR="build/Release"
    mkdir -p "$RELEASE_DIR"

    cp "$BUILD_DIR/bin/Release/"*.dll "$RELEASE_DIR/"
    echo "✓ Copied DLLs to $RELEASE_DIR/"

    # Locate and copy llama.lib
    if [ -f "$BUILD_DIR/src/Release/llama.lib" ]; then
        cp "$BUILD_DIR/src/Release/llama.lib" "$RELEASE_DIR/llama.lib"
    elif [ -f "$BUILD_DIR/lib/Release/llama.lib" ]; then
        cp "$BUILD_DIR/lib/Release/llama.lib" "$RELEASE_DIR/llama.lib"
    elif [ -f "$BUILD_DIR/bin/Release/llama.lib" ]; then
        cp "$BUILD_DIR/bin/Release/llama.lib" "$RELEASE_DIR/llama.lib"
    else
        echo "Error: llama.lib not found in build output"
        exit 1
    fi
    echo "✓ Copied llama.lib to $RELEASE_DIR/"

else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

echo "✓ llama.cpp build complete"
