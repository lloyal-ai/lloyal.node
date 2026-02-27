import type { Resource, Chunk } from '../resources/types.js';
import type { Reranker, Tool, ExecuteToolFn } from './types.js';
import { createSearchTool } from './search.js';
import { createReadFileTool } from './read-file.js';
import { createGrepTool } from './grep.js';
import { createReportTool } from './report.js';

export function createTools(opts: {
  resources: Resource[];
  chunks: Chunk[];
  reranker: Reranker;
}): { tools: Tool[]; toolsJson: string; executeTool: ExecuteToolFn } {
  const tools = [
    createSearchTool(opts.chunks, opts.reranker),
    createReadFileTool(opts.resources),
    createGrepTool(opts.resources),
    createReportTool(),
  ];

  const toolsJson = JSON.stringify(tools.map((t) => t.schema));
  const toolMap = new Map(tools.map((t) => [t.name, t]));

  const executeTool: ExecuteToolFn = async (name, args, context?) => {
    const tool = toolMap.get(name);
    if (!tool) return { error: `Unknown tool: ${name}` };
    return tool.execute(args, context);
  };

  return { tools, toolsJson, executeTool };
}
