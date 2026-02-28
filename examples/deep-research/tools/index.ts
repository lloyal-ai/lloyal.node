import { createToolkit } from '../../../dist/agents/index.js';
import type { Toolkit } from '../../../dist/agents/index.js';
import type { Resource, Chunk } from '../resources/types.js';
import type { Reranker } from './types.js';
import { SearchTool } from './search.js';
import { ReadFileTool } from './read-file.js';
import { GrepTool } from './grep.js';

export function createTools(opts: {
  resources: Resource[];
  chunks: Chunk[];
  reranker: Reranker;
}): Toolkit {
  return createToolkit([
    new SearchTool(opts.chunks, opts.reranker),
    new ReadFileTool(opts.resources),
    new GrepTool(opts.resources),
  ]);
}
