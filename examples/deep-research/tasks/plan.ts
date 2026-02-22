import * as fs from 'node:fs';
import * as path from 'node:path';
import { Branch } from '../../../dist/index.js';
import type { SessionContext } from '../../../dist/index.js';

const PLAN_PROMPT = fs.readFileSync(path.resolve(__dirname, 'plan.md'), 'utf8');

export async function plan(ctx: SessionContext, opts: {
  query: string;
  agentCount: number;
  parent?: InstanceType<typeof Branch>;
}): Promise<{ questions: string[]; tokenCount: number }> {
  const schema = {
    type: 'object',
    properties: {
      questions: {
        type: 'array',
        items: { type: 'string' },
        minItems: 2,
        maxItems: opts.agentCount,
      },
    },
    required: ['questions'],
  };
  const grammar = await ctx.jsonSchemaToGrammar(JSON.stringify(schema));

  const userContent = PLAN_PROMPT
    .replace('{{count}}', String(opts.agentCount))
    .replace('{{query}}', opts.query);

  const messages = [
    { role: 'system', content: 'You break research queries into sub-questions. Output JSON only.' },
    { role: 'user', content: userContent },
  ];
  const { prompt } = await ctx.formatChat(JSON.stringify(messages));

  let lead: InstanceType<typeof Branch>;
  if (opts.parent) {
    // Warm: fork from trunk — planner inherits conversation KV
    lead = await opts.parent.fork();
    lead.setGrammar(grammar);
    const sep = ctx.getTurnSeparator();
    const delta = await ctx.tokenize(prompt, false);
    await lead.prefill([...sep, ...delta]);
  } else {
    // Cold: fresh branch at position 0
    const tokens = await ctx.tokenize(prompt);
    lead = Branch.create(ctx, 0, { temperature: 0.3 }, undefined, grammar);
    await lead.prefill(tokens);
  }

  let output = '';
  let tokenCount = 0;
  for await (const { text } of lead) {
    output += text;
    tokenCount++;
  }
  await lead.prune();

  let questions: string[];
  try {
    questions = JSON.parse(output).questions.slice(0, opts.agentCount);
    if (!questions.length) throw new Error('empty questions');
  } catch {
    questions = Array.from({ length: opts.agentCount }, (_, i) => `${opts.query} (aspect ${i + 1})`);
  }

  return { questions, tokenCount };
}
