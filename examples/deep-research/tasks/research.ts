import * as fs from 'node:fs';
import * as path from 'node:path';
import { Branch, BranchStore, forkAgent, runAgents } from '../../../dist/index.js';
import type { SessionContext, AgentState } from '../../../dist/index.js';
import type { ExecuteToolFn } from '../tools/types.js';

const DEFAULT_SYSTEM_PROMPT = fs.readFileSync(path.resolve(__dirname, 'research.md'), 'utf8');

export { DEFAULT_SYSTEM_PROMPT as RESEARCH_SYSTEM_PROMPT };

export interface AgentResult {
  findings: string | null;
  toolCallCount: number;
  tokenCount: number;
}

export interface ResearchResult {
  agents: AgentResult[];
  totalTokens: number;
  totalToolCalls: number;
  steps: number;
  counters: Record<string, number>;
  sharedPrefixLength: number;
}

export async function research(ctx: SessionContext, store: BranchStore, opts: {
  questions: string[];
  parent?: InstanceType<typeof Branch>;
  seed?: number;
  systemPrompt?: string;
  toolsJson: string;
  executeTool: ExecuteToolFn;
  maxTurns?: number;
  onToolCall?: (agentIndex: number, toolName: string, args: string) => void;
  onToolResult?: (agentIndex: number, toolName: string, resultStr: string) => void;
}): Promise<ResearchResult> {
  const systemPrompt = opts.systemPrompt ?? DEFAULT_SYSTEM_PROMPT;

  let agents: AgentState[];
  let sharedPrefixLength: number;
  let root: InstanceType<typeof Branch> | null;

  if (opts.parent) {
    // Warm: fork from conversation trunk — each agent inherits full KV,
    // gets a fresh system prompt + question injected as suffix.
    // Diversity via reseeded sampler, not divergent content.
    agents = await Promise.all(
      opts.questions.map((q, i) =>
        forkAgent(opts.parent!, {
          systemPrompt,
          content: q,
          tools: opts.toolsJson,
          seed: opts.seed != null ? opts.seed + i : undefined,
        }, ctx)
      )
    );
    sharedPrefixLength = 0;
    root = null;
  } else {
    // Cold: shared-prefix optimization — one root with system prompt,
    // fork N agents with divergent user-question suffixes.
    const sharedMessages = [{ role: 'system', content: systemPrompt }];
    const sharedFmt = await ctx.formatChat(
      JSON.stringify(sharedMessages),
      { tools: opts.toolsJson, addGenerationPrompt: false },
    );
    const sharedTokens = await ctx.tokenize(sharedFmt.prompt);

    root = Branch.create(ctx, 0, { temperature: 0.5 });
    await root.prefill(sharedTokens);

    agents = [];
    for (const q of opts.questions) {
      const branch = await root.fork();
      const fullMessages = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: q },
      ];
      const fmt = await ctx.formatChat(JSON.stringify(fullMessages), { tools: opts.toolsJson });
      const fullTokens = await ctx.tokenize(fmt.prompt);
      const suffixTokens = fullTokens.slice(sharedTokens.length);

      agents.push({
        branch,
        suffixTokens,
        fmt: {
          format: fmt.format,
          reasoningFormat: fmt.reasoningFormat,
          thinkingForcedOpen: fmt.thinkingForcedOpen,
          parser: fmt.parser,
        },
        rawOutput: '',
        done: false,
        tokenCount: 0,
        toolCallCount: 0,
        turns: 0,
        findings: null,
      });
    }
    sharedPrefixLength = sharedTokens.length;
  }

  // Common path: batch prefill + agentic loop + prune
  await store.prefill(agents.map((a) => [a.branch, a.suffixTokens]));

  const result = await runAgents(agents, {
    store, ctx,
    executeTool: opts.executeTool,
    maxTurns: opts.maxTurns ?? 6,
    onToolCall: opts.onToolCall,
    onToolResult: opts.onToolResult,
  });

  for (const a of agents) await a.branch.prune();
  if (root) await root.prune();

  return {
    agents: agents.map((a) => ({
      findings: a.findings,
      toolCallCount: a.toolCallCount,
      tokenCount: a.tokenCount,
    })),
    ...result,
    sharedPrefixLength,
  };
}
