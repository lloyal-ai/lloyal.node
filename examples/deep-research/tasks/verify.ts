import * as fs from 'node:fs';
import * as path from 'node:path';
import { Branch, BranchStore } from '../../../dist/index.js';
import type { SessionContext } from '../../../dist/index.js';

const VERIFY_PROMPT = fs.readFileSync(path.resolve(__dirname, 'verify.md'), 'utf8');

export interface Attempt {
  output: string;
  tokenCount: number;
  ppl: number;
}

export interface VerifyResult {
  attempts: Attempt[];
  bestOutput: string;
  bestBranch: InstanceType<typeof Branch>;
  totalTokens: number;
  steps: number;
  prefixLength: number;
}

export async function verify(ctx: SessionContext, store: BranchStore, opts: {
  findings: string;
  query: string;
  count: number;
}): Promise<VerifyResult> {
  const userContent = VERIFY_PROMPT
    .replace('{{findings}}', opts.findings)
    .replace('{{query}}', opts.query);

  const messages = [
    { role: 'system', content: 'Synthesize the research findings into a coherent, concise summary.' },
    { role: 'user', content: userContent },
  ];
  const { prompt } = await ctx.formatChat(JSON.stringify(messages));
  const synthTokens = await ctx.tokenize(prompt);

  const synthRoot = Branch.create(ctx, 0, { temperature: 0.7 });
  await synthRoot.prefill(synthTokens);

  // Fork N branches with reseeded samplers for stochastic divergence
  const live: { branch: InstanceType<typeof Branch>; output: string; done: boolean; tokenCount: number; ppl: number }[] = [];
  for (let i = 0; i < opts.count; i++) {
    const branch = await synthRoot.fork();
    branch.reseedSampler(2000 + i);
    live.push({ branch, output: '', done: false, tokenCount: 0, ppl: Infinity });
  }

  // BranchStore batched decode — produceSync/commit loop
  let steps = 0;
  for (;;) {
    const entries: [InstanceType<typeof Branch>, number][] = [];
    for (const a of live) {
      if (a.done) continue;
      const { token, text, isStop } = a.branch.produceSync();
      if (isStop) {
        const p = a.branch.perplexity;
        a.ppl = Number.isFinite(p) ? p : Infinity;
        a.done = true;
        continue;
      }
      entries.push([a.branch, token]);
      a.output += text;
      a.tokenCount++;
    }
    if (entries.length === 0) break;
    await store.commit(entries);
    steps++;
  }

  // Pick lowest perplexity (most coherent)
  const bestIdx = live.reduce((bi, a, i) => a.ppl <= live[bi].ppl ? i : bi, 0);

  // Prune non-best attempts; synthRoot stays alive (bestBranch is its child)
  // — caller's retainOnly will clean up synthRoot when promoting bestBranch
  for (let i = 0; i < live.length; i++) {
    if (i !== bestIdx) await live[i].branch.prune();
  }

  const totalTokens = live.reduce((s, a) => s + a.tokenCount, 0);

  return {
    attempts: live.map((a) => ({ output: a.output, tokenCount: a.tokenCount, ppl: a.ppl })),
    bestOutput: live[bestIdx].output,
    bestBranch: live[bestIdx].branch,
    totalTokens,
    steps,
    prefixLength: synthTokens.length,
  };
}
