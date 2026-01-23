#!/bin/bash
set -e

echo "Downloading test models..."

mkdir -p models
cd models

# SmolLM2 1.7B Instruct Q4_K_M (1.0GB)
if [ ! -f "SmolLM2-1.7B-Instruct-Q4_K_M.gguf" ]; then
  echo "  → Downloading SmolLM2-1.7B-Instruct-Q4_K_M.gguf..."
  curl -L -o "SmolLM2-1.7B-Instruct-Q4_K_M.gguf" \
    "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf"
  echo "  ✓ Downloaded SmolLM2"
else
  echo "  ✓ SmolLM2 already exists"
fi

# Nomic Embed Text v1.5 Q4_K_M (80MB)
if [ ! -f "nomic-embed-text-v1.5.Q4_K_M.gguf" ]; then
  echo "  → Downloading nomic-embed-text-v1.5.Q4_K_M.gguf..."
  curl -L -o "nomic-embed-text-v1.5.Q4_K_M.gguf" \
    "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf"
  echo "  ✓ Downloaded nomic-embed-text"
else
  echo "  ✓ nomic-embed-text already exists"
fi

echo ""
echo "✅ All test models ready"
