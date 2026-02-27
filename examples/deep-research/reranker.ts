import { Rerank } from '../../dist/index.js';
import type { Chunk } from './resources/types.js';
import type { Reranker, ScoredResult } from './tools/types.js';

export async function createReranker(
  modelPath: string,
  opts?: { nSeqMax?: number; nCtx?: number },
): Promise<Reranker> {
  const rerank = await Rerank.create({ modelPath, ...opts });

  return {
    score(query: string, chunks: Chunk[]): AsyncIterable<ScoredResult> {
      const inner = rerank.score(query, chunks.map(c => c.tokens), 5);
      return {
        [Symbol.asyncIterator](): AsyncIterator<ScoredResult> {
          const it = inner[Symbol.asyncIterator]();
          return {
            async next(): Promise<IteratorResult<ScoredResult>> {
              const { value, done } = await it.next();
              if (done) return { value: undefined as unknown as ScoredResult, done: true };
              return {
                value: {
                  filled: value.filled,
                  total: value.total,
                  results: value.results.map(r => ({
                    file: chunks[r.index].resource,
                    heading: chunks[r.index].heading,
                    score: r.score,
                    startLine: chunks[r.index].startLine,
                    endLine: chunks[r.index].endLine,
                  })),
                },
                done: false,
              };
            },
          };
        },
      };
    },

    async tokenizeChunks(chunks: Chunk[]): Promise<void> {
      for (const chunk of chunks) {
        chunk.tokens = await rerank.tokenize(chunk.text);
      }
    },

    dispose() { rerank.dispose(); },
  };
}
