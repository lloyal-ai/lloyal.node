import { createContext, Branch } from '../../dist/index.js';
import type { Chunk } from './resources/types.js';
import type { Reranker, ScoredChunk } from './tools/types.js';

const SYSTEM_PROMPT =
  'Judge whether the Document meets the requirements based on the Query ' +
  'and the Instruct provided. Note that the answer can only be "yes" or "no".';

const USER_PREFIX =
  '<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n\n' +
  '<Query>: ';

export async function createReranker(
  modelPath: string,
  opts?: { nSeqMax?: number; nCtx?: number },
): Promise<Reranker> {
  const nCtx = opts?.nCtx ?? 16384;
  const ctx = await createContext({
    modelPath,
    nCtx,
    nSeqMax: opts?.nSeqMax ?? 3,
  });

  const [yesId] = await ctx.tokenize('yes', false);
  const [noId] = await ctx.tokenize('no', false);

  // Probe the chat template once to extract prefix/mid/suffix, then
  // pre-tokenize segments. Per-chunk scoring concatenates token arrays
  // synchronously — no per-chunk formatChat calls needed.
  const SENTINEL_Q = '\x00QUERY\x00';
  const SENTINEL_D = '\x00DOC\x00';
  const probe = await ctx.formatChat(JSON.stringify([
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: `${USER_PREFIX}${SENTINEL_Q}\n\n<Document>: ${SENTINEL_D}` },
  ]), { addGenerationPrompt: true, enableThinking: false });

  const p = probe.prompt;
  const qi = p.indexOf(SENTINEL_Q);
  const di = p.indexOf(SENTINEL_D);
  const prefixTokens = await ctx.tokenize(p.slice(0, qi), true);
  const midTokens = await ctx.tokenize(p.slice(qi + SENTINEL_Q.length, di), false);
  const suffixTokens = await ctx.tokenize(p.slice(di + SENTINEL_D.length), false);

  function rerankScore(logits: Float32Array): number {
    const max = Math.max(logits[yesId], logits[noId]);
    const yesExp = Math.exp(logits[yesId] - max);
    const noExp = Math.exp(logits[noId] - max);
    return yesExp / (yesExp + noExp);
  }

  // Serialize access — concurrent Branch.prefill on the same llama_context
  // races llama_decode.
  let lock = Promise.resolve();

  return {
    async score(query: string, chunks: Chunk[]): Promise<ScoredChunk[]> {
      const prev = lock;
      let release!: () => void;
      lock = new Promise<void>((r) => { release = r; });
      await prev;
      try {
        const queryTokens = await ctx.tokenize(query, false);
        const budget = nCtx - prefixTokens.length - queryTokens.length
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
