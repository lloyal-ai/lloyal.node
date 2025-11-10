#!/usr/bin/env node
import React from 'react';
import { render } from 'ink';
import { Chat } from './components/Chat.js';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Default model path (relative to package root)
const DEFAULT_MODEL_PATH = path.join(__dirname, '../../../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf');

// Parse command line arguments
const args = process.argv.slice(2);
const modelPath = args[0] || DEFAULT_MODEL_PATH;

// Render the app
render(<Chat modelPath={modelPath} />);
