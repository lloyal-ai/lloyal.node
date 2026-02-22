import type { Resource } from '../resources/types.js';
import type { Tool } from './types.js';

export function createReadFileTool(resources: Resource[]): Tool {
  return {
    name: 'read_file',
    schema: {
      type: 'function',
      function: {
        name: 'read_file',
        description: 'Read content from a file at specific line ranges. Use startLine/endLine from search results.',
        parameters: {
          type: 'object',
          properties: {
            filename: {
              type: 'string',
              description: 'Filename from search results',
              enum: resources.map((r) => r.name),
            },
            startLine: { type: 'number', description: 'Start line (1-indexed, from search results)' },
            endLine: { type: 'number', description: 'End line (1-indexed, from search results)' },
          },
          required: ['filename'],
        },
      },
    },
    async execute(args) {
      const filename = (args.filename as string) || (args.path as string) || '';
      const file = resources.find((r) => r.name === filename);
      if (!file) {
        return { error: `File not found: ${filename}. Available: ${resources.map((r) => r.name).join(', ')}` };
      }
      const lines = file.content.split('\n');
      const s = Math.max(0, ((args.startLine as number) ?? 1) - 1);
      const e = Math.min(lines.length, (args.endLine as number) ?? Math.min(100, lines.length));
      return { file: file.name, content: lines.slice(s, e).join('\n') };
    },
  };
}
