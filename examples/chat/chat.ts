#!/usr/bin/env node
/**
 * Simple chat example using lloyal.node
 *
 * Usage:
 *   npx tsx chat.ts /path/to/model.gguf
 *   npx tsx chat.ts  # uses default model path
 *
 * This example demonstrates:
 * - Branch API for token generation (produce/commit two-phase)
 * - Warm multi-turn continuation via formatChat([newMsg]) + getTurnSeparator()
 * - Cold/warm routing: full format on first turn, format-only-new on subsequent turns
 * - parseChatOutput() for correct reasoning_content handling on thinking models
 */

import * as readline from "node:readline";
import * as path from "node:path";
import { createContext, Branch } from "../../dist/index.js";
import type { SessionContext, FormattedChatResult } from "../../dist/index.js";

const DEFAULT_MODEL = path.resolve(
  __dirname,
  "../../models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
);

async function main(): Promise<void> {
  const modelPath = process.argv[2] || DEFAULT_MODEL;

  console.log(`Loading model: ${modelPath}`);
  console.log("This may take a moment...\n");

  const nCtx = parseInt(process.env.LLAMA_CTX_SIZE || '2048', 10);
  const ctx: SessionContext = await createContext({
    modelPath,
    nCtx,
    threads: 4,
  });

  console.log("Model loaded! Type your message and press Enter.");
  console.log("Commands: /clear to reset, /quit to exit\n");

  const messages: Array<{role: string; content: string; reasoning_content?: string}> = [];
  let branch: InstanceType<typeof Branch> | null = null;
  let fmt: FormattedChatResult | null = null;
  const sep: number[] = ctx.getTurnSeparator();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const askUser = (): void => { rl.question("> ", handleInput); };

  async function handleInput(input: string): Promise<void> {
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
      // === COLD (position === 0): full format → tokenize with BOS → prefill ===
      fmt = await ctx.formatChat(JSON.stringify(messages));
      const tokens = await ctx.tokenize(fmt.prompt);
      branch = Branch.create(ctx, 0, {
        temperature: 0.7,
        topK: 40,
        topP: 0.9,
      });
      await branch.prefill(tokens);
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
    const parsed = ctx.parseChatOutput(rawOutput, fmt!.format, {
      reasoningFormat: fmt!.reasoningFormat,
      thinkingForcedOpen: fmt!.thinkingForcedOpen,
      parser: fmt!.parser,
    });

    const msg: {role: string; content: string; reasoning_content?: string} = { role: "assistant", content: parsed.content };
    if (parsed.reasoningContent) {
      msg.reasoning_content = parsed.reasoningContent;
    }
    messages.push(msg);

    askUser();
  }

  askUser();
}

main().catch((err: unknown) => {
  console.error("Error:", (err as Error).message);
  process.exit(1);
});
