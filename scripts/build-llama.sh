#!/usr/bin/env bash
#
# Build llama.cpp for the current platform
# Called automatically by npm prepare

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

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Building llama.cpp for Linux..."
    if [ ! -f "$LLAMA_DIR/build-linux/libllama.so" ]; then
        ./scripts/build-linux-shared.sh "$LLAMA_DIR"
    fi

    # Copy .so to build/Release/ for node-gyp (needed for dynamic linking)
    RELEASE_DIR="build/Release"
    mkdir -p "$RELEASE_DIR"
    cp "$LLAMA_DIR/build-linux/libllama.so" "$RELEASE_DIR/libllama.so"
    echo "✓ Copied libllama.so to $RELEASE_DIR/"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Building llama.cpp for macOS..."
    if [ ! -f "$LLAMA_DIR/build-apple/libllama.dylib" ]; then
        cd "$LLAMA_DIR"
        BUILD_DIR="build-apple"

        # Detect CI environment and disable Metal (virtualized macOS has no real GPU)
        if [ -n "$CI" ]; then
            echo "CI detected - building without Metal (CPU only)"
            METAL_ENABLED=OFF
            METAL_LIBS=""
        else
            echo "Local build - enabling Metal for GPU acceleration"
            METAL_ENABLED=ON
            METAL_LIBS="$BUILD_DIR/ggml/src/ggml-metal/libggml-metal.a"
        fi

        echo "Building static libraries with CMake..."
        cmake -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_SHARED_LIBS=OFF \
            -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_TOOLS=OFF \
            -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_BUILD_SERVER=OFF \
            -DLLAMA_BUILD_COMMON=OFF \
            -DLLAMA_CURL=OFF \
            -DGGML_METAL=$METAL_ENABLED \
            -DGGML_BLAS=ON \
            -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
            -S .

        cmake --build "$BUILD_DIR" --config Release -j$(sysctl -n hw.ncpu)

        echo "Combining static libraries into single shared library..."
        # Combine all static libs like build-xcframework.sh does
        if [ "$METAL_ENABLED" = "ON" ]; then
            libtool -static -o "$BUILD_DIR/libllama-combined.a" \
                "$BUILD_DIR/src/libllama.a" \
                "$BUILD_DIR/ggml/src/libggml.a" \
                "$BUILD_DIR/ggml/src/libggml-base.a" \
                "$BUILD_DIR/ggml/src/libggml-cpu.a" \
                "$BUILD_DIR/ggml/src/ggml-metal/libggml-metal.a" \
                "$BUILD_DIR/ggml/src/ggml-blas/libggml-blas.a"
        else
            libtool -static -o "$BUILD_DIR/libllama-combined.a" \
                "$BUILD_DIR/src/libllama.a" \
                "$BUILD_DIR/ggml/src/libggml.a" \
                "$BUILD_DIR/ggml/src/libggml-base.a" \
                "$BUILD_DIR/ggml/src/libggml-cpu.a" \
                "$BUILD_DIR/ggml/src/ggml-blas/libggml-blas.a"
        fi

        # Create single shared library with proper install_name
        if [ "$METAL_ENABLED" = "ON" ]; then
            clang++ -dynamiclib \
                -arch arm64 -arch x86_64 \
                -Wl,-force_load,"$BUILD_DIR/libllama-combined.a" \
                -framework Foundation -framework Metal -framework Accelerate \
                -install_name "@rpath/libllama.dylib" \
                -o "$BUILD_DIR/libllama.dylib"
        else
            clang++ -dynamiclib \
                -arch arm64 -arch x86_64 \
                -Wl,-force_load,"$BUILD_DIR/libllama-combined.a" \
                -framework Foundation -framework Accelerate \
                -install_name "@rpath/libllama.dylib" \
                -o "$BUILD_DIR/libllama.dylib"
        fi

        echo "✓ Combined shared library created: $BUILD_DIR/libllama.dylib"
        cd -
    fi

    # Copy .dylib to build/Release/ for node-gyp (needed for dynamic linking)
    RELEASE_DIR="build/Release"
    mkdir -p "$RELEASE_DIR"
    cp "$LLAMA_DIR/build-apple/libllama.dylib" "$RELEASE_DIR/libllama.dylib"
    echo "✓ Copied libllama.dylib to $RELEASE_DIR/"
elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]] || [[ "$OSTYPE" == "win32"* ]]; then
    echo "Building llama.cpp for Windows..."
    if [ ! -f "$LLAMA_DIR/build-windows/llama.dll" ]; then
        cd "$LLAMA_DIR"
        BUILD_DIR="build-windows"

        echo "Building with CMake for Windows..."
        cmake -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_SHARED_LIBS=ON \
            -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_TOOLS=OFF \
            -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_BUILD_SERVER=OFF \
            -DLLAMA_BUILD_COMMON=OFF \
            -DLLAMA_CURL=OFF \
            -S .

        cmake --build "$BUILD_DIR" --config Release

        echo "✓ Shared library created: $BUILD_DIR/bin/Release/llama.dll"
        cd -
    fi

    # Copy DLL and import library to build/Release/ for node-gyp
    RELEASE_DIR="build/Release"
    mkdir -p "$RELEASE_DIR"
    cp "$LLAMA_DIR/build-windows/bin/Release/llama.dll" "$RELEASE_DIR/llama.dll"
    echo "✓ Copied llama.dll to $RELEASE_DIR/"

    # Copy import library (.lib) - needed for linking
    # CMake may place it in lib/, bin/, or src/ depending on configuration
    if [ -f "$LLAMA_DIR/build-windows/src/Release/llama.lib" ]; then
        cp "$LLAMA_DIR/build-windows/src/Release/llama.lib" "$RELEASE_DIR/llama.lib"
        echo "✓ Copied llama.lib (from src/) to $RELEASE_DIR/"
    elif [ -f "$LLAMA_DIR/build-windows/lib/Release/llama.lib" ]; then
        cp "$LLAMA_DIR/build-windows/lib/Release/llama.lib" "$RELEASE_DIR/llama.lib"
        echo "✓ Copied llama.lib (from lib/) to $RELEASE_DIR/"
    elif [ -f "$LLAMA_DIR/build-windows/bin/Release/llama.lib" ]; then
        cp "$LLAMA_DIR/build-windows/bin/Release/llama.lib" "$RELEASE_DIR/llama.lib"
        echo "✓ Copied llama.lib (from bin/) to $RELEASE_DIR/"
    else
        echo "Error: llama.lib not found in $LLAMA_DIR/build-windows/{src,lib,bin}/Release/"
        exit 1
    fi
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

echo "✓ llama.cpp build complete"
