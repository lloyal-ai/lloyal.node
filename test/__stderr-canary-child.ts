/**
 * Child process for test/stderr-canary.ts — runs REAL inference with a
 * canary-bearing prompt so the parent can inspect this process's stderr.
 * Native writes go straight to fd 2 and bypass any in-process JS hook,
 * which is why the canary must run in a child rather than in-process.
 *
 * Prints generated text and a completion marker to STDOUT only.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { Branch, loadBinary } from '../dist/index.js';
import type { NativeBinding } from '../dist/index.js';

const CANARY = process.env.STDERR_CANARY;
if (!CANARY) {
  console.error('STDERR_CANARY env missing'); // eslint-disable-line no-console
  process.exit(2);
}

const MODEL_PATH: string = process.env.LLAMA_TEST_MODEL
  ? path.resolve(process.env.LLAMA_TEST_MODEL)
  : path.join(__dirname, '../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf');

async function main(): Promise<void> {
  if (!fs.existsSync(MODEL_PATH)) {
    process.stdout.write(`MODEL_MISSING ${MODEL_PATH}\n`);
    process.exit(3);
  }

  let addon: NativeBinding;
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    addon = require('../build/Release/lloyal.node') as NativeBinding;
  } catch {
    addon = loadBinary();
  }

  const ctx = await addon.createContext({ modelPath: MODEL_PATH, nCtx: 512 });
  try {
    // The canary rides the PROMPT — through tokenize, prefill (decode),
    // and sampling. If any layer echoes prompt content to stderr, the
    // parent's assertion catches it.
    const prompt = `Repeat this code verbatim: ${CANARY}`;
    const tokens = await ctx.tokenize(prompt);
    const branch = Branch.create(ctx, 0, { temperature: 0 });
    await branch.prefill(tokens);
    let generated = '';
    for await (const { text } of branch) {
      generated += text;
      if (generated.length > 64) break; // a few tokens traverse the hot path; that's enough
    }
    await branch.prune();
    process.stdout.write(`GENERATED: ${generated}\n`);
    process.stdout.write('CANARY_RUN_COMPLETE\n');
  } finally {
    ctx.dispose();
  }
  process.exit(0);
}

main().catch((err) => {
  // Failure detail on STDOUT — this process's stderr is the surface under
  // test and must stay attributable to the native layer alone.
  process.stdout.write(`CHILD_ERROR: ${(err as Error).stack ?? String(err)}\n`);
  process.exit(1);
});
