import { createContext, Branch } from '../../dist/index.js';
import type { Chunk } from './resources/types.js';
import type { Reranker, ScoredChunk } from './tools/types.js';

const RERANK_PREFIX =
  '<|im_start|>system\n' +
  'Judge whether the Document meets the requirements based on the Query ' +
  'and the Instruct provided. Note that the answer can only be "yes" or "no".' +
  '<|im_end|>\n<|im_start|>user\n' +
  '<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n\n' +
  '<Query>: ';
const RERANK_MID = '\n\n<Document>: ';
const RERANK_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n';

export async function createReranker(
  modelPath: string,
  opts?: { nSeqMax?: number },
): Promise<Reranker> {
  const ctx = await createContext({
    modelPath,
    nCtx: 16384,
    nSeqMax: opts?.nSeqMax ?? 3,
  });

  const [yesId] = await ctx.tokenize('yes', false);
  const [noId] = await ctx.tokenize('no', false);
  const prefixTokens = await ctx.tokenize(RERANK_PREFIX, true);
  const midTokens = await ctx.tokenize(RERANK_MID, false);
  const suffixTokens = await ctx.tokenize(RERANK_SUFFIX, false);

  function rerankScore(logits: Float32Array): number {
    const max = Math.max(logits[yesId], logits[noId]);
    const yesExp = Math.exp(logits[yesId] - max);
    const noExp = Math.exp(logits[noId] - max);
    return yesExp / (yesExp + noExp);
  }

  // Serialize access — concurrent Branch.prefill on the same llama_context
  // races llama_decode. BranchStore serializes via batched decode, but
  // individual Branch.prefill calls on the reranker bypass that.
  let lock = Promise.resolve();

  return {
    async score(query: string, chunks: Chunk[]): Promise<ScoredChunk[]> {
      const prev = lock;
      let release!: () => void;
      lock = new Promise<void>((r) => { release = r; });
      await prev;
      try {
        const queryTokens = await ctx.tokenize(query, false);
        const budget = 16384 - prefixTokens.length - queryTokens.length
                     - midTokens.length - suffixTokens.length;
        const scored: ScoredChunk[] = [];
        for (const chunk of chunks) {
          const docTokens = chunk.tokens.length > budget
            ? chunk.tokens.slice(0, budget) : chunk.tokens;
          const tokens = [
            ...prefixTokens, ...queryTokens,
            ...midTokens, ...docTokens,
            ...suffixTokens,
          ];
          const branch = Branch.create(ctx, 0, { temperature: 0 });
          await branch.prefill(tokens);
          const score = rerankScore(branch.getLogits());
          await branch.prune();
          scored.push({
            file: chunk.resource, heading: chunk.heading,
            score: Math.round(score * 1000) / 1000,
            startLine: chunk.startLine, endLine: chunk.endLine,
          });
        }
        return scored.sort((a, b) => b.score - a.score).slice(0, 5);
      } finally {
        release();
      }
    },

    async tokenizeChunks(chunks: Chunk[]): Promise<void> {
      for (const chunk of chunks) {
        chunk.tokens = await ctx.tokenize(chunk.text, false);
      }
    },

    dispose(): void {
      ctx.dispose();
    },
  };
}
