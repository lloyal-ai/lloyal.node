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
else
    echo "Error: Unsupported platform: $OSTYPE"
    exit 1
fi
