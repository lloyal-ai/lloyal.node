#!/bin/bash
set -e

echo "=== GPU Test Environment ==="
echo "CUDA:   $(nvcc --version 2>/dev/null | grep release || echo 'runtime-only')"
echo "GPU:    $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'N/A')"
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
./scripts/download-test-models.sh --all

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
echo "=== Running Model Matrix ==="

# Read model list from matrix.json
MODELS=$(jq -c '.models[]' test/matrix.json)

# Per-model results tracking
TOTAL=0
PASS=0
FAIL=0
declare -a RESULTS=()

# Don't exit on per-model failure — track results and report at end
set +e

while IFS= read -r model; do
  name=$(echo "$model" | jq -r '.name')
  file=$(echo "$model" | jq -r '.file')

  echo ""
  echo "══════════════════════════════════════"
  echo "MODEL: $name ($file)"
  echo "══════════════════════════════════════"

  TOTAL=$((TOTAL + 1))

  LLOYAL_GPU="${GPU_BACKEND}" \
  LLOYAL_NO_FALLBACK=1 \
  MODEL_PATH="models/$file" \
  node test/integration.js

  if [ $? -eq 0 ]; then
    RESULTS+=("✅ $name")
    PASS=$((PASS + 1))
  else
    RESULTS+=("❌ $name")
    FAIL=$((FAIL + 1))
  fi
done <<< "$MODELS"

set -e

echo ""
echo "=== Running Examples (default model) ==="
LLOYAL_GPU="${GPU_BACKEND}" \
LLOYAL_NO_FALLBACK=1 \
node test/examples.js

# Final summary table
echo ""
echo "══════════════════════════════════════"
echo "MODEL MATRIX RESULTS"
echo "══════════════════════════════════════"
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo ""
echo "Total: $PASS passed, $FAIL failed out of $TOTAL models"

if [ $FAIL -eq 0 ]; then
  echo ""
  echo "=== ✅ GPU Tests Passed ==="
  exit 0
else
  echo ""
  echo "=== ❌ GPU Tests Failed ==="
  exit 1
fi
