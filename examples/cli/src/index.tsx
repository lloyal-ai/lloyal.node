#!/usr/bin/env node
import React from 'react';
import { render } from 'ink';
import { App } from './components/App.js';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Default model path (relative to package root)
const DEFAULT_MODEL_PATH = path.join(__dirname, '../../../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf');

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  let modelPath = DEFAULT_MODEL_PATH;
  let contextSize = 8192;
  let threads = 1;

  for (const arg of args) {
    if (arg.startsWith('--model=')) {
      modelPath = arg.substring('--model='.length);
    } else if (arg.startsWith('--context=')) {
      contextSize = parseInt(arg.substring('--context='.length), 10);
    } else if (arg.startsWith('--threads=')) {
      threads = parseInt(arg.substring('--threads='.length), 10);
    } else if (!arg.startsWith('--')) {
      // Support legacy format (positional argument)
      modelPath = arg;
    }
  }

  return { modelPath, contextSize, threads };
}

const { modelPath, contextSize, threads } = parseArgs();

// Render the app
render(<App modelPath={modelPath} contextSize={contextSize} threads={threads} />);
