#!/usr/bin/env bash
#
# Build llama.cpp as a single shared library (.so) for Linux
# This mirrors the XCFramework approach to avoid ODR violations in N-API
#
# Run from liblloyal-node root: ./scripts/build-linux-shared.sh

set -e

cd llama.cpp

BUILD_DIR="build-linux"
COMMON_C_FLAGS="-O3 -DNDEBUG"
COMMON_CXX_FLAGS="-O3 -DNDEBUG -std=c++20"

echo "=== Building llama.cpp for Linux (single shared library) ==="
echo ""

# Clean previous build
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning previous build..."
    rm -rf "$BUILD_DIR"
fi

# Build static libraries with CMake
echo "Building static libraries with CMake..."
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_CURL=OFF \
    -DCMAKE_C_FLAGS="${COMMON_C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -S .

cmake --build "$BUILD_DIR" --config Release -j$(nproc)

echo ""
echo "Static libraries built successfully:"
ls -lh "$BUILD_DIR"/src/libllama.a "$BUILD_DIR"/ggml/src/libggml*.a

# Combine all static libraries into one
echo ""
echo "Combining static libraries..."
COMBINED_LIB="$BUILD_DIR/libllama-combined.a"

# Use GNU ar MRI script to merge archives
# This is equivalent to macOS 'libtool -static'
cat > "$BUILD_DIR/merge.mri" << EOF
CREATE $COMBINED_LIB
ADDLIB $BUILD_DIR/src/libllama.a
ADDLIB $BUILD_DIR/ggml/src/libggml-base.a
ADDLIB $BUILD_DIR/ggml/src/libggml-cpu.a
SAVE
END
EOF

ar -M < "$BUILD_DIR/merge.mri"
rm "$BUILD_DIR/merge.mri"

echo "✓ Combined library: $COMBINED_LIB ($(du -h $COMBINED_LIB | cut -f1))"

# Create shared library from combined archive
echo ""
echo "Creating shared library..."
SHARED_LIB="$BUILD_DIR/libllama.so"

# Use --whole-archive to include ALL symbols (equivalent to macOS -force_load)
# This ensures liblloyal's inline static members are properly merged
g++ -shared -fPIC \
    -Wl,--whole-archive "$COMBINED_LIB" -Wl,--no-whole-archive \
    -o "$SHARED_LIB" \
    -lpthread -ldl -lm -lstdc++

echo "✓ Shared library: $SHARED_LIB ($(du -h $SHARED_LIB | cut -f1))"

# Verify shared library
echo ""
echo "Verifying shared library..."
if ldd "$SHARED_LIB" > /dev/null 2>&1; then
    echo "✓ Shared library dependencies:"
    ldd "$SHARED_LIB" | grep -E "(pthread|stdc)"
else
    echo "⚠️  ldd check failed (may be normal on some systems)"
fi

if nm -D "$SHARED_LIB" | grep -q "llama_model_load"; then
    echo "✓ Symbol verification passed (llama_model_load found)"
else
    echo "❌ Symbol verification failed!"
    exit 1
fi

echo ""
echo "=== Build complete! ==="
echo ""
echo "Output: $SHARED_LIB"
echo ""
echo "Next steps:"
echo "  1. Update binding.gyp to link against $SHARED_LIB"
echo "  2. Run: npm run build"
echo ""
