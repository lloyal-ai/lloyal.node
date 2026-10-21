import { strict as assert } from 'node:assert';
import { after, before, test } from 'node:test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { loadLocalAddon } from './local-addon';
import type { AudioLimits, MultimodalInput, SessionContext } from '@lloyal-labs/sdk';

/** The addon's embedding surface, ahead of the SDK release that declares it. */
type EmbeddingContext = SessionContext & {
  encodeMultimodal(prompt: string, inputs: MultimodalInput[], limits?: AudioLimits): Promise<void>;
};

interface ReferenceEntry {
  name: string;
  kind: 'text' | 'image' | 'video' | 'text+image';
  text?: string;
  image?: string;
  text_before?: string;
  text_after?: string;
  finds?: string;
  vector: number[];
}

const MARKER = '<__media__>';
const fixtures = resolve(__dirname, '../liblloyal/tests/fixtures');
const models = resolve(__dirname, '../models/embeddinggemma-2');
const fixture = (name: string) => readFileSync(resolve(fixtures, name));
const reference: { dimension: number; entries: ReferenceEntry[] } =
  JSON.parse(fixture('embedding/embeddinggemma2-reference.json').toString('utf8'));
const queries = reference.entries.filter(entry => entry.kind === 'text');
const media = reference.entries.filter(entry => entry.kind !== 'text');
const searchable = media.filter(entry => entry.kind !== 'text+image');

const image = (name: string): MultimodalInput => ({ kind: 'image', bytes: fixture(name) });
/** The frames llama.cpp's video helper samples from the clip, at their sample times. */
const clip: MultimodalInput = {
  kind: 'video',
  frames: Array.from({ length: 16 }, (_, i) => fixture(`video/frame-${String(i).padStart(2, '0')}.png`)),
  timestampsMs: Array.from({ length: 16 }, (_, i) => i * 250),
  timestampIntervalMs: 5000,
};

/** Each media entry of the reference as a prompt and its inputs. */
const inputs: Record<Exclude<ReferenceEntry['kind'], 'text'>, (entry: ReferenceEntry) => [string, MultimodalInput[]]> = {
  image: entry => [MARKER, [image(entry.image!)]],
  'text+image': entry => [entry.text_before! + MARKER + entry.text_after!, [image(entry.image!)]],
  video: () => [MARKER, [clip]],
};

let ctx: EmbeddingContext;
const contexts: SessionContext[] = [];

/** An EmbeddingGemma 2 context whose one dispatch holds `nBatch` entries. */
async function embeddingContext(nBatch: number): Promise<EmbeddingContext> {
  const context = await loadLocalAddon().createContext({
    modelPath: process.env.LLAMA_EMBED_MM_MODEL ?? resolve(models, 'embeddinggemma-2-Q8_0.gguf'),
    mmprojPath: process.env.LLAMA_EMBED_MM_MMPROJ ?? resolve(models, 'mmproj-BF16.gguf'),
    nCtx: nBatch, nBatch, nThreads: 4,
    embeddings: true, poolingType: 1,
  }) as EmbeddingContext;
  contexts.push(context);
  return context;
}

// The reference was read at this size, so every input fits one dispatch.
before(async () => { ctx = await embeddingContext(8192); });
after(() => contexts.forEach(context => context.dispose()));

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  Array.from(a).reduce((sum, value, i) => sum + value * b[i], 0);
const cosine = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  dot(a, b) / Math.sqrt(dot(a, a) * dot(b, b));

async function embed(context: EmbeddingContext, entry: ReferenceEntry): Promise<Float32Array> {
  await context.kvCacheClear();
  if (entry.kind === 'text') await context.encode(await context.tokenize(entry.text!, true));
  else await context.encodeMultimodal(...inputs[entry.kind](entry));
  return context.getEmbeddings(true);
}

/** Each entry embedded once on the full-size context: on a CPU runner the largest take minutes. */
const embedded = new Map<string, Promise<Float32Array>>();
const vectorOf = (entry: ReferenceEntry) => {
  if (!embedded.has(entry.name)) embedded.set(entry.name, embed(ctx, entry));
  return embedded.get(entry.name)!;
};

function assertMatches(vector: Float32Array, entry: ReferenceEntry): void {
  assert.equal(vector.length, reference.dimension, entry.name);
  const similarity = cosine(vector, entry.vector);
  assert.ok(similarity >= 0.999, `${entry.name}: ${similarity}`);
}

test('the projector carries a vision and an audio encoder', () => {
  assert.equal(ctx.supportsVision(), true);
  assert.equal(ctx.supportsAudio(), true);
});

test('the vector is as wide as the model outputs', () => {
  assert.equal(ctx.getEmbeddingDimension(), reference.dimension);
});

test('a query embeds where llama.cpp puts it', async () => {
  for (const query of queries) assertMatches(await vectorOf(query), query);
});

test('an image, text around an image, and a clip embed where llama.cpp puts them', async () => {
  for (const entry of media) assertMatches(await vectorOf(entry), entry);
});

test('each query finds its own image or clip', async () => {
  for (const query of queries) {
    const vector = await vectorOf(query);
    const scored = await Promise.all(searchable.map(async entry => ({ entry, score: dot(vector, await vectorOf(entry)) })));
    const best = scored.reduce((a, b) => (b.score > a.score ? b : a));
    assert.equal(best.entry.name, query.finds, query.name);
  }
});

test('the measure is exactly what one dispatch admits, and a refusal leaves the context as it was', async () => {
  const cat = media.find(entry => entry.kind === 'image' && entry.image === 'cat.jpg')!;
  const [prompt, catInputs] = inputs.image(cat);
  const cells = await ctx._cellsMultimodal([], prompt, catInputs);

  const short = await embeddingContext(cells - 1);
  await assert.rejects(short.encodeMultimodal(prompt, catInputs), /must go in one dispatch/);
  assertMatches(await embed(short, queries[0]), queries[0]);

  assertMatches(await embed(await embeddingContext(cells), cat), cat);
});
