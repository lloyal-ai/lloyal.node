import type { Chunk } from '../resources/types.js';

export interface ScoredChunk {
  file: string;
  heading: string;
  score: number;
  startLine: number;
  endLine: number;
}

export interface Reranker {
  score(query: string, chunks: Chunk[]): Promise<ScoredChunk[]>;
  tokenizeChunks(chunks: Chunk[]): Promise<void>;
  dispose(): void;
}

export interface Tool {
  name: string;
  schema: object;
  execute: (args: Record<string, unknown>) => Promise<unknown>;
}

export type ExecuteToolFn = (name: string, args: Record<string, unknown>) => Promise<unknown>;
