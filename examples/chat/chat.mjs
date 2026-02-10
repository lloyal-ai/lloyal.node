#!/usr/bin/env node
/**
 * Simple chat example using lloyal.node
 *
 * Usage:
 *   node chat.mjs /path/to/model.gguf
 *   node chat.mjs  # uses default model path
 *
 * This example demonstrates:
 * - Sync generator for token production (sample, check stop, convert to text)
 * - Async commit via decode() to update KV cache
 * - Clear separation: sync produce, async commit
 */

import * as readline from "node:readline";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { createContext } from "../../lib/index.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_MODEL = path.resolve(
  __dirname,
  "../../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
);

/**
 * Sync generator - produces tokens until stop token.
 * All operations are synchronous: sample, isStopToken, tokenToText.
 */
function* produceTokens(ctx, params) {
  while (true) {
    const tokenId = ctx.sample(params);
    if (ctx.isStopToken(tokenId)) return;
    const text = ctx.tokenToText(tokenId);
    yield { text, tokenId };
  }
}

async function main() {
  const modelPath = process.argv[2] || DEFAULT_MODEL;

  console.log(`Loading model: ${modelPath}`);
  console.log("This may take a moment...\n");

  const ctx = await createContext({
    modelPath,
    contextSize: 2048,
    threads: 4,
  });

  console.log("Model loaded! Type your message and press Enter.");
  console.log("Commands: /clear to reset, /quit to exit\n");

  const messages = [];
  let position = 0;
  let lastPrompt = "";

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const askUser = () => rl.question("> ", handleInput);

  async function handleInput(input) {
    const trimmed = input.trim();

    if (trimmed === "/quit" || trimmed === "/exit") {
      console.log("Goodbye!");
      ctx.dispose();
      rl.close();
      return;
    }

    if (trimmed === "/clear") {
      await ctx.kvCacheClear();
      messages.length = 0;
      position = 0;
      lastPrompt = "";
      console.clear();
      console.log("Conversation cleared.\n");
      askUser();
      return;
    }

    if (!trimmed) {
      askUser();
      return;
    }

    messages.push({ role: "user", content: trimmed });

    // Format with chat template
    const { prompt: fullPrompt } = await ctx.formatChat(
      JSON.stringify(messages),
    );

    // Prompt diffing - only tokenize new content
    const newContent = fullPrompt.startsWith(lastPrompt)
      ? fullPrompt.slice(lastPrompt.length)
      : fullPrompt;

    const tokens = await ctx.tokenize(newContent);
    await ctx.decode(tokens, position);
    position += tokens.length;

    // Generate: sync produce, async commit
    process.stdout.write("< ");
    let response = "";

    for (const { text, tokenId } of produceTokens(ctx, {
      temperature: 0.7,
      topK: 40,
      topP: 0.9,
    })) {
      process.stdout.write(text);
      response += text;

      await ctx.decode([tokenId], position); // async commit to KV
      position += 1;
    }

    console.log("\n");

    messages.push({ role: "assistant", content: response.trim() });
    lastPrompt = fullPrompt + response;

    askUser();
  }

  askUser();
}

main().catch((err) => {
  console.error("Error:", err.message);
  process.exit(1);
});
