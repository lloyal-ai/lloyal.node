#!/usr/bin/env bash
#
# Copy libllama shared library to build/Release/ for node-gyp linking
# Called between node-gyp configure and node-gyp build
#
# This script is needed because node-gyp clean removes build/Release/,
# so we can't copy the .dylib before clean. We copy it after configure
# creates the directory structure.

set -e

# Resolve llama.cpp source directory (vendor/ or submodule)
if [ -d "vendor/llama.cpp" ]; then
    LLAMA_DIR="vendor/llama.cpp"
elif [ -d "llama.cpp" ]; then
    LLAMA_DIR="llama.cpp"
else
    echo "Error: llama.cpp not found"
    exit 1
fi

# Create build/Release if it doesn't exist (should exist after configure)
mkdir -p build/Release

# Copy appropriate library for platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ -f "$LLAMA_DIR/build-apple/libllama.dylib" ]; then
        cp "$LLAMA_DIR/build-apple/libllama.dylib" build/Release/
        echo "✓ Copied libllama.dylib to build/Release/"
    else
        echo "Error: libllama.dylib not found at $LLAMA_DIR/build-apple/"
        echo "Run: bash scripts/build-llama.sh"
        exit 1
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ -f "$LLAMA_DIR/build-linux/libllama.so" ]; then
        cp "$LLAMA_DIR/build-linux/libllama.so" build/Release/
        echo "✓ Copied libllama.so to build/Release/"
    else
        echo "Error: libllama.so not found at $LLAMA_DIR/build-linux/"
        echo "Run: bash scripts/build-llama.sh"
        exit 1
    fi
elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]] || [[ "$OSTYPE" == "win32"* ]]; then
    if [ -f "$LLAMA_DIR/build-windows/bin/Release/llama.dll" ]; then
        # Copy all DLLs (llama.dll depends on ggml*.dll)
        cp "$LLAMA_DIR/build-windows/bin/Release/"*.dll build/Release/
        echo "✓ Copied DLLs to build/Release/:"
        ls -1 build/Release/*.dll | sed 's/.*\//  - /'

        # Copy import library (.lib) - needed for linking
        # CMake may place it in lib/, bin/, or src/ depending on configuration
        if [ -f "$LLAMA_DIR/build-windows/src/Release/llama.lib" ]; then
            cp "$LLAMA_DIR/build-windows/src/Release/llama.lib" build/Release/
            echo "✓ Copied llama.lib (from src/) to build/Release/"
        elif [ -f "$LLAMA_DIR/build-windows/lib/Release/llama.lib" ]; then
            cp "$LLAMA_DIR/build-windows/lib/Release/llama.lib" build/Release/
            echo "✓ Copied llama.lib (from lib/) to build/Release/"
        elif [ -f "$LLAMA_DIR/build-windows/bin/Release/llama.lib" ]; then
            cp "$LLAMA_DIR/build-windows/bin/Release/llama.lib" build/Release/
            echo "✓ Copied llama.lib (from bin/) to build/Release/"
        else
            echo "Error: llama.lib not found in $LLAMA_DIR/build-windows/{src,lib,bin}/Release/"
            echo "Run: bash scripts/build-llama.sh"
            exit 1
        fi
    else
        echo "Error: llama.dll not found at $LLAMA_DIR/build-windows/bin/Release/"
        echo "Run: bash scripts/build-llama.sh"
        exit 1
    fi
else
    echo "Error: Unsupported platform: $OSTYPE"
    exit 1
fi
