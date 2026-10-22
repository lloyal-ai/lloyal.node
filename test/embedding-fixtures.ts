import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { loadLocalAddon } from './local-addon';
import type { AudioLimits, MultimodalInput, SessionContext } from '@lloyal-labs/sdk';

/** The addon's embedding surface, ahead of the SDK release that declares it. */
export type EmbeddingContext = SessionContext & {
  encodeMultimodal(prompt: string, inputs: MultimodalInput[], limits?: AudioLimits): Promise<void>;
};

export const MARKER = '<__media__>';
const fixtures = resolve(__dirname, '../liblloyal/tests/fixtures');
const models = resolve(__dirname, '../models/embeddinggemma-2');
export const modelPath = process.env.LLAMA_EMBED_MM_MODEL ?? resolve(models, 'embeddinggemma-2-Q8_0.gguf');
export const mmprojPath = process.env.LLAMA_EMBED_MM_MMPROJ ?? resolve(models, 'mmproj-BF16.gguf');

export const fixture = (name: string) => readFileSync(resolve(fixtures, name));

/** The frames llama.cpp's video helper samples from the committed clip, at their sample times; `from` starts a window. */
export const clip = (from = 0): MultimodalInput => ({
  kind: 'video',
  frames: Array.from({ length: 16 - from }, (_, i) => fixture(`video/frame-${String(from + i).padStart(2, '0')}.png`)),
  timestampsMs: Array.from({ length: 16 - from }, (_, i) => (from + i) * 250),
  timestampIntervalMs: 5000,
});

/** An EmbeddingGemma 2 context: one dispatch holds `nBatch` entries, at the projector's image budget unless one is pinned. */
export const embeddingContext = async (nBatch = 8192, imageTokens?: number) => await loadLocalAddon().createContext({
  modelPath, mmprojPath, nCtx: nBatch, nBatch, nThreads: 4, embeddings: true, poolingType: 1,
  ...(imageTokens !== undefined ? { imageMinTokens: imageTokens, imageMaxTokens: imageTokens } : {}),
}) as EmbeddingContext;

/** A context that lives for `use` alone, freed before this resolves: each holds a model and projector in GPU memory. */
export async function withContext<T>(made: Promise<SessionContext>, use: (context: EmbeddingContext) => Promise<T> | T): Promise<T> {
  const context = await made as EmbeddingContext;
  try {
    return await use(context);
  } finally {
    await context.dispose();
  }
}

/** The dot product of two vectors; for unit vectors, their cosine. */
export const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => Array.from(a).reduce((sum, x, i) => sum + x * b[i], 0);

/** One input's vector: the context cleared, the prompt and its media encoded, the pooled embedding read. */
export async function embed(context: EmbeddingContext, prompt: string, inputs: MultimodalInput[] = [], limits?: AudioLimits): Promise<Float32Array> {
  await context.kvCacheClear();
  if (inputs.length === 0) await context.encode(await context.tokenize(prompt, true));
  else await context.encodeMultimodal(prompt, inputs, limits);
  return context.getEmbeddings(true);
}

/** `vectorOf` over `items` one after another: an embed clears, encodes and reads its context, so two must not interleave. */
export async function oneAtATime<T>(items: readonly T[], vectorOf: (item: T) => Promise<Float32Array>): Promise<Float32Array[]> {
  const vectors: Float32Array[] = [];
  for (const item of items) vectors.push(await vectorOf(item));
  return vectors;
}
