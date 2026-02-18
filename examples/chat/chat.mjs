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
 * - Warm multi-turn continuation via formatChat([newMsg]) + getTurnSeparator()
 * - Cold/warm routing: full format on first turn, format-only-new on subsequent turns
 * - parseChatOutput() for correct reasoning_content handling on thinking models
 */

import * as readline from "node:readline";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { createContext, Branch } from "../../lib/index.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_MODEL = path.resolve(
  __dirname,
  "../../models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
);

async function main() {
  const modelPath = process.argv[2] || DEFAULT_MODEL;

  console.log(`Loading model: ${modelPath}`);
  console.log("This may take a moment...\n");

  const nCtx = parseInt(process.env.LLAMA_CTX_SIZE || '2048', 10);
  const ctx = await createContext({
    modelPath,
    nCtx,
    threads: 4,
  });

  console.log("Model loaded! Type your message and press Enter.");
  console.log("Commands: /clear to reset, /quit to exit\n");

  const messages = [];
  let branch = null;
  let fmt = null;
  const sep = ctx.getTurnSeparator();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const askUser = () => rl.question("> ", handleInput);

  async function handleInput(input) {
    const trimmed = input.trim();

    if (trimmed === "/quit" || trimmed === "/exit") {
      console.log("Goodbye!");
      if (branch) await branch.prune();
      ctx.dispose();
      rl.close();
      return;
    }

    if (trimmed === "/clear") {
      if (branch) await branch.prune();
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
      fmt = await ctx.formatChat(JSON.stringify(messages));
      const tokens = await ctx.tokenize(fmt.prompt);
      await ctx.decode(tokens, 0, 0);
      branch = Branch.create(ctx, tokens.length, {
        temperature: 0.7,
        topK: 40,
        topP: 0.9,
      });
      branch.captureLogits();
    } else {
      // === WARM (position > 0): format only the new message ===
      fmt = await ctx.formatChat(
        JSON.stringify([{ role: "system", content: "" }, { role: "user", content: trimmed }]),
      );
      const delta = await ctx.tokenize(fmt.prompt, false);
      await branch.prefill([...sep, ...delta]);
    }

    process.stdout.write("< ");
    let rawOutput = "";

    for await (const { text } of branch) {
      process.stdout.write(text);
      rawOutput += text;
    }

    console.log("\n");

    // Parse output: separates reasoning from content for thinking models
    const parsed = ctx.parseChatOutput(rawOutput, fmt.format, {
      reasoningFormat: fmt.reasoningFormat,
      thinkingForcedOpen: fmt.thinkingForcedOpen,
      parser: fmt.parser,
    });

    const msg = { role: "assistant", content: parsed.content };
    if (parsed.reasoningContent) {
      msg.reasoning_content = parsed.reasoningContent;
    }
    messages.push(msg);

    askUser();
  }

  askUser();
}

main().catch((err) => {
  console.error("Error:", err.message);
  process.exit(1);
});
