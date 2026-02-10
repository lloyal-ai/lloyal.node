#!/usr/bin/env node
/**
 * Simple chat example using lloyal.node
 *
 * Usage:
 *   node chat.mjs /path/to/model.gguf
 *   node chat.mjs  # uses default model path
 *
 * This example demonstrates:
 * - Branch API for token generation (produce/commit two-phase)
 * - Warm multi-turn continuation via getWarmTurnTokens() + branch.prefill()
 * - Cold/warm routing: formatChat() on first turn, probe-based prefill on subsequent turns
 */

import * as readline from "node:readline";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { createContext, Branch } from "../../lib/index.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_MODEL = path.resolve(
  __dirname,
  "../../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
);

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
  let branch = null;
  const warm = ctx.getWarmTurnTokens();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const askUser = () => rl.question("> ", handleInput);

  async function handleInput(input) {
    const trimmed = input.trim();

    if (trimmed === "/quit" || trimmed === "/exit") {
      console.log("Goodbye!");
      if (branch) branch.prune();
      ctx.dispose();
      rl.close();
      return;
    }

    if (trimmed === "/clear") {
      if (branch) branch.prune();
      branch = null;
      await ctx.kvCacheClear();
      messages.length = 0;
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

    if (!branch) {
      // === COLD (position === 0): full format → tokenize with BOS → decode ===
      const { prompt } = await ctx.formatChat(JSON.stringify(messages));
      const tokens = await ctx.tokenize(prompt);
      await ctx.decode(tokens, 0, 0);
      branch = Branch.create(ctx, 0, tokens.length, {
        temperature: 0.7,
        topK: 40,
        topP: 0.9,
      });
      branch.captureLogits();
    } else {
      // === WARM (position > 0): probe-based prefill — no formatChat(), no BOS ===
      const contentToks = await ctx.tokenize(trimmed, false);
      branch.prefill([
        ...warm.turnSeparator,
        ...warm.userPrefix,
        ...contentToks,
        ...warm.userToAssistant,
      ]);
    }

    // Generate: produce inspects, commit advances
    process.stdout.write("< ");
    let response = "";

    while (true) {
      const { token, text, isStop } = branch.produce();
      if (isStop) break;
      process.stdout.write(text);
      response += text;
      branch.commit(token);
    }

    console.log("\n");

    messages.push({ role: "assistant", content: response.trim() });

    askUser();
  }

  askUser();
}

main().catch((err) => {
  console.error("Error:", err.message);
  process.exit(1);
});
