#!/bin/bash
set -e

echo "=== GPU Test Environment ==="
# NOTE: nvcc is N/A because we use cuda:runtime image (no compiler needed - binary already built)
echo "CUDA:   $(nvcc --version 2>/dev/null | grep release || echo 'runtime-only')"
echo "GPU:    $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "Vulkan ICD files:"
ls -la /usr/share/vulkan/icd.d/ 2>/dev/null || echo "  (none found)"
echo ""
echo "Vulkan devices:"
vulkaninfo --summary 2>&1 | head -20 || echo "  (vulkaninfo failed)"
echo ""

# GPU backend to test (passed as arg or env)
GPU_BACKEND="${1:-${LLOYAL_GPU:-cuda}}"

# Validate GPU backend to avoid injection and invalid values
case "${GPU_BACKEND}" in
  cuda|vulkan)
    ;;
  *)
    echo "Error: Invalid GPU backend '${GPU_BACKEND}'. Allowed values are: cuda, vulkan." >&2
    exit 1
    ;;
esac

PACKAGE_NAME="linux-x64-${GPU_BACKEND}"

echo "Testing backend: ${GPU_BACKEND}"
echo "Installing package: ${PACKAGE_NAME}"

# Install the pre-built package
cd /app
if [ -d "./packages/package-${PACKAGE_NAME}" ]; then
  npm install "./packages/package-${PACKAGE_NAME}"
else
  echo "Error: Package directory not found: ./packages/package-${PACKAGE_NAME}"
  ls -la ./packages/
  exit 1
fi

echo ""
echo "=== Downloading Test Models ==="
./scripts/download-test-models.sh

echo ""
echo "=== Verifying Backend ==="
node -e "
const { loadBinary } = require('./lib');
process.env.LLOYAL_GPU = '${GPU_BACKEND}';
process.env.LLOYAL_NO_FALLBACK = '1';
try {
  const addon = loadBinary();
  console.log('✓ Package loaded successfully');
  console.log('  Exports:', Object.keys(addon));
} catch (e) {
  console.error('Failed to load binary:', e);
  process.exit(1);
}
"

echo ""
echo "=== Running Integration Tests ==="
LLOYAL_GPU="${GPU_BACKEND}" \
LLOYAL_NO_FALLBACK=1 \
node test/integration.js

echo ""
echo "=== Running Examples ==="
LLOYAL_GPU="${GPU_BACKEND}" \
LLOYAL_NO_FALLBACK=1 \
node test/examples.js

echo ""
echo "=== ✅ GPU Tests Passed ==="
