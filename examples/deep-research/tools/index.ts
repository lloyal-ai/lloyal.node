import { createToolkit } from '../../../dist/agents';
import type { Toolkit } from '../../../dist/agents';
import type { Resource, Chunk } from '../resources/types';
import type { Reranker } from './types';
import { SearchTool } from './search';
import { ReadFileTool } from './read-file';
import { GrepTool } from './grep';

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
