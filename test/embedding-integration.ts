import { strict as assert } from 'node:assert';
import { after, before, test } from 'node:test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { loadLocalAddon } from './local-addon';
import type { SessionContext } from '@lloyal-labs/sdk';

interface ReferenceEntry { name: string; kind: string; text?: string; vector: number[] }

const fixtures = resolve(__dirname, '../liblloyal/tests/fixtures');
const models = resolve(__dirname, '../models/embeddinggemma-2');
const reference: { dimension: number; entries: ReferenceEntry[] } =
  JSON.parse(readFileSync(resolve(fixtures, 'embedding/embeddinggemma2-reference.json'), 'utf8'));
const queries = reference.entries.filter(entry => entry.kind === 'text');
let ctx: SessionContext;

before(async () => {
  // The reference was read at these sizes, so every input fits one dispatch.
  ctx = await loadLocalAddon().createContext({
    modelPath: process.env.LLAMA_EMBED_MM_MODEL ?? resolve(models, 'embeddinggemma-2-Q8_0.gguf'),
    mmprojPath: process.env.LLAMA_EMBED_MM_MMPROJ ?? resolve(models, 'mmproj-BF16.gguf'),
    nCtx: 8192, nBatch: 8192, nThreads: 4,
    embeddings: true, poolingType: 1,
  });
});

after(() => ctx?.dispose());

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  Array.from(a).reduce((sum, value, i) => sum + value * b[i], 0);
const cosine = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  dot(a, b) / Math.sqrt(dot(a, a) * dot(b, b));

async function embedText(text: string): Promise<Float32Array> {
  await ctx.kvCacheClear();
  await ctx.encode(await ctx.tokenize(text, true));
  return ctx.getEmbeddings(true);
}

test('the projector carries a vision and an audio encoder', () => {
  assert.equal(ctx.supportsVision(), true);
  assert.equal(ctx.supportsAudio(), true);
});

test('the vector is as wide as the model outputs', () => {
  assert.equal(ctx.getEmbeddingDimension(), reference.dimension);
});

test('a query embeds where llama.cpp puts it', async () => {
  for (const query of queries) {
    const vector = await embedText(query.text!);
    assert.equal(vector.length, reference.dimension, query.name);
    assert.ok(cosine(vector, query.vector) >= 0.999, `${query.name}: ${cosine(vector, query.vector)}`);
  }
});
