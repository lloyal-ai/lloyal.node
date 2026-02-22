import type { Chunk } from '../resources/types.js';
import type { Reranker, Tool } from './types.js';

export function createSearchTool(chunks: Chunk[], reranker: Reranker): Tool {
  return {
    name: 'search',
    schema: {
      type: 'function',
      function: {
        name: 'search',
        description: 'Search the knowledge base. Returns sections ranked by relevance with line ranges for read_file.',
        parameters: {
          type: 'object',
          properties: { query: { type: 'string', description: 'Search query' } },
          required: ['query'],
        },
      },
    },
    async execute(args) {
      return reranker.score((args.query as string) || '', chunks);
    },
  };
}
