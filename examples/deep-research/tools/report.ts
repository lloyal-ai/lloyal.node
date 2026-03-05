import { Tool } from '../../../dist/agents';
import type { JsonSchema } from '../../../dist/agents';

export class ReportTool extends Tool<{ findings: string }> {
  readonly name = 'report';
  readonly description = 'Submit your final research findings. Call this when you have gathered enough information to answer the question.';
  readonly parameters: JsonSchema = {
    type: 'object',
    properties: { findings: { type: 'string', description: 'Your research findings and answer' } },
    required: ['findings'],
  };

  async execute(): Promise<unknown> { return {}; }
}
