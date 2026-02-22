import type { Tool } from './types.js';

export function createReportTool(): Tool {
  return {
    name: 'report',
    schema: {
      type: 'function',
      function: {
        name: 'report',
        description: 'Submit your final research findings. Call this when you have gathered enough information to answer the question.',
        parameters: {
          type: 'object',
          properties: { findings: { type: 'string', description: 'Your research findings and answer' } },
          required: ['findings'],
        },
      },
    },
    async execute() {
      return { acknowledged: true };
    },
  };
}
