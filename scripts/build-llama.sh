#!/usr/bin/env bash
#
# Build llama.cpp for the current platform
# Called automatically by npm prepare

set -e

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Building llama.cpp for Linux..."
    if [ ! -f "llama.cpp/build-linux/libllama.so" ]; then
        ./scripts/build-linux-shared.sh
    else
        echo "✓ llama.cpp already built (llama.cpp/build-linux/libllama.so exists)"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Building llama.cpp for macOS..."
    if [ ! -d "llama.cpp/build-apple/llama.xcframework" ]; then
        cd llama.cpp
        ./build-xcframework.sh
        cd ..
    else
        echo "✓ llama.cpp already built (llama.cpp/build-apple/llama.xcframework exists)"
    fi
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

echo "✓ llama.cpp build complete"
