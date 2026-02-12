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

# Common env for all test runs
export LLOYAL_GPU="${GPU_BACKEND}"
export LLOYAL_NO_FALLBACK=1
export LLAMA_CTX_SIZE=4096

echo ""
echo "=== Running Model Matrix (nCtx=${LLAMA_CTX_SIZE}) ==="

# Read model list from matrix.json
MODELS=$(jq -c '.models[]' test/matrix.json)

# Per-model results tracking
TOTAL=0
PASS=0
FAIL=0
declare -a RESULTS=()
declare -a FAIL_DETAILS=()

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
  MODEL_LOG=$(mktemp)
  MODEL_FAILED=false

  # --- Integration tests ---
  echo "── Integration Tests ──"
  LLAMA_TEST_MODEL="models/$file" \
  node test/integration.js 2>&1 | tee "$MODEL_LOG"
  INT_EXIT=${PIPESTATUS[0]}

  if [ $INT_EXIT -ne 0 ]; then
    MODEL_FAILED=true
  fi

  # --- Example tests ---
  echo ""
  echo "── Example Tests ──"
  LLAMA_TEST_MODEL="models/$file" \
  node test/examples.js 2>&1 | tee -a "$MODEL_LOG"
  EX_EXIT=${PIPESTATUS[0]}

  if [ $EX_EXIT -ne 0 ]; then
    MODEL_FAILED=true
  fi

  # Per-model summary
  if [ "$MODEL_FAILED" = false ]; then
    RESULTS+=("✅ $name")
    PASS=$((PASS + 1))
  else
    RESULTS+=("❌ $name")
    FAIL=$((FAIL + 1))
    # Extract failure lines for the final summary
    FAILURES=$(grep -E '\[FAIL\]|❌ FAILED|Assertion failed|Fatal error' "$MODEL_LOG" | head -10)
    FAIL_DETAILS+=("── $name ──"$'\n'"$FAILURES")
  fi

  rm -f "$MODEL_LOG"
done <<< "$MODELS"

set -e

# Final summary table
echo ""
echo "══════════════════════════════════════"
echo "MODEL MATRIX RESULTS"
echo "══════════════════════════════════════"
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo ""
echo "Total: $PASS passed, $FAIL failed out of $TOTAL models"

if [ $FAIL -gt 0 ] && [ ${#FAIL_DETAILS[@]} -gt 0 ]; then
  echo ""
  echo "══════════════════════════════════════"
  echo "FAILURE DETAILS"
  echo "══════════════════════════════════════"
  for d in "${FAIL_DETAILS[@]}"; do
    echo "$d"
    echo ""
  done
fi

if [ $FAIL -eq 0 ]; then
  echo ""
  echo "=== ✅ GPU Tests Passed ==="
  exit 0
else
  echo ""
  echo "=== ❌ GPU Tests Failed ==="
  exit 1
fi
