#!/bin/bash
set -e

# Download test models from test/matrix.json
# Usage:
#   ./scripts/download-test-models.sh          # Download default models only
#   ./scripts/download-test-models.sh --all    # Download all models in matrix

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MATRIX_FILE="$PROJECT_ROOT/test/matrix.json"
MODELS_DIR="$PROJECT_ROOT/models"

# Check for --all flag
DOWNLOAD_ALL=false
if [ "$1" = "--all" ]; then
  DOWNLOAD_ALL=true
fi

# Ensure jq is available
if ! command -v jq &> /dev/null; then
  echo "Error: jq is required but not installed."
  echo "Install with: brew install jq (macOS) or apt-get install jq (Linux)"
  exit 1
fi

# Ensure matrix.json exists
if [ ! -f "$MATRIX_FILE" ]; then
  echo "Error: $MATRIX_FILE not found"
  exit 1
fi

mkdir -p "$MODELS_DIR"

download_model() {
  local name="$1"
  local file="$2"
  local url="$3"
  local dest="$MODELS_DIR/$file"

  if [ -f "$dest" ]; then
    echo "  ✓ $name already exists"
  else
    echo "  → Downloading $name..."
    curl -L -o "$dest" "$url"
    echo "  ✓ Downloaded $name"
  fi
}

echo "=========================================="
echo "Downloading test models from matrix.json"
echo "=========================================="
echo ""

# Download chat models
echo "Chat Models:"
if [ "$DOWNLOAD_ALL" = true ]; then
  # Download all models
  jq -c '.models[]' "$MATRIX_FILE" | while read -r model; do
    name=$(echo "$model" | jq -r '.name')
    file=$(echo "$model" | jq -r '.file')
    url=$(echo "$model" | jq -r '.url')
    download_model "$name" "$file" "$url"
  done
else
  # Download only default models
  jq -c '.models[] | select(.default == true)' "$MATRIX_FILE" | while read -r model; do
    name=$(echo "$model" | jq -r '.name')
    file=$(echo "$model" | jq -r '.file')
    url=$(echo "$model" | jq -r '.url')
    download_model "$name" "$file" "$url"
  done
fi
echo ""

# Download embedding models
echo "Embedding Models:"
if [ "$DOWNLOAD_ALL" = true ]; then
  jq -c '.embeddings[]' "$MATRIX_FILE" | while read -r model; do
    name=$(echo "$model" | jq -r '.name')
    file=$(echo "$model" | jq -r '.file')
    url=$(echo "$model" | jq -r '.url')
    download_model "$name" "$file" "$url"
  done
else
  jq -c '.embeddings[] | select(.default == true)' "$MATRIX_FILE" | while read -r model; do
    name=$(echo "$model" | jq -r '.name')
    file=$(echo "$model" | jq -r '.file')
    url=$(echo "$model" | jq -r '.url')
    download_model "$name" "$file" "$url"
  done
fi
echo ""

# Download sidecar models
echo "Sidecar Models:"
jq -c '.sidecars[]' "$MATRIX_FILE" | while read -r model; do
  name=$(echo "$model" | jq -r '.name')
  file=$(echo "$model" | jq -r '.file')
  url=$(echo "$model" | jq -r '.url')
  download_model "$name" "$file" "$url"
done
echo ""

echo "=========================================="
echo "✅ All test models ready"
echo "=========================================="
