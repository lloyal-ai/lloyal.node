import * as fs from 'node:fs';
import * as path from 'node:path';
import { Branch } from '../../../dist/index.js';
import type { SessionContext } from '../../../dist/index.js';

const EVAL_PROMPT = fs.readFileSync(path.resolve(__dirname, 'eval.md'), 'utf8');

export async function evaluate(ctx: SessionContext, opts: {
  attempts: { output: string }[];
}): Promise<{ converged: boolean | null; tokenCount: number }> {
  const responsesText = opts.attempts
    .map((a, i) => `Response ${i + 1}: ${a.output.trim()}`)
    .join('\n\n');

  const userContent = EVAL_PROMPT.replace('{{responses}}', responsesText);

  const messages = [
    {
      role: 'system',
      content: 'You are a consistency checker. Compare the responses and determine if they convey the same core meaning. Output JSON only.',
    },
    { role: 'user', content: userContent },
  ];

  const evalSchema = {
    type: 'object',
    properties: { converged: { type: 'boolean' } },
    required: ['converged'],
  };
  const grammar = await ctx.jsonSchemaToGrammar(JSON.stringify(evalSchema));

  const { prompt } = await ctx.formatChat(JSON.stringify(messages));
  const tokens = await ctx.tokenize(prompt);

  const branch = Branch.create(ctx, 0, { temperature: 0 }, undefined, grammar);
  await branch.prefill(tokens);

  let output = '';
  let tokenCount = 0;
  for await (const { text } of branch) {
    output += text;
    tokenCount++;
  }
  await branch.prune();

  let converged: boolean | null;
  try {
    converged = JSON.parse(output).converged;
  } catch {
    converged = null;
  }

  return { converged, tokenCount };
}
