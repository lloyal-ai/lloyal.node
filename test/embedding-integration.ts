import { strict as assert } from 'node:assert';
import { after, before, test } from 'node:test';
import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { loadLocalAddon, withContext } from './local-addon';
import { clip, dot, embed, embeddingContext, fixture, MARKER, mmprojPath, modelPath, oneAtATime } from './embedding-fixtures';
import type { EmbeddingContext } from './embedding-fixtures';
import type { AudioLimits, MultimodalInput } from '@lloyal-labs/sdk';

/** One part of an input, in order: text as written, or a fixture of one kind of media. */
type Part = { text: string } | { image: string } | { video: string } | { audio: string };

interface ReferenceEntry {
  name: string;
  kind: 'text' | 'image' | 'video' | 'audio' | 'mixed';
  text?: string;
  image?: string;
  video?: string;
  audio?: string;
  parts?: Part[];
  finds?: string;
  image_tokens?: number;
  vector: number[];
}

const reference: { dimension: number; entries: ReferenceEntry[] } =
  JSON.parse(fixture('embedding/embeddinggemma2-reference.json').toString('utf8'));
const named = (name: string) => reference.entries.find(entry => entry.name === name)!;

/** On a CPU runner the projector takes minutes for each image and each frame: CI's macOS build has Metal off, since
 *  the paravirtual GPU there computes wrong. `LLOYAL_EMBED_LEAN=1` checks the text, the smallest image budget and the
 *  recording against the reference, and leaves search, which reads every image and the clip, to the runners that
 *  check the whole reference. */
const lean = process.env.LLOYAL_EMBED_LEAN === '1';
const smallestBudget = Math.min(...reference.entries.map(entry => entry.image_tokens ?? Infinity));
const checked = lean
  ? reference.entries.filter(entry => entry.kind === 'text' || entry.kind === 'audio' || entry.image_tokens === smallestBudget)
  : reference.entries;
const searchSkipped = lean && 'search reads every image and the clip: checked where the whole reference is';
const queries = reference.entries.filter(entry => entry.finds !== undefined);

/** Room for every committed recording; the cases below that refuse a recording set their own limits. */
const ROOM: AudioLimits = { maxBytes: 64 << 20, maxSamples: 16_000 * 600 };

/** Each kind of media part, and the input it becomes. */
const media: Record<string, (name: string) => MultimodalInput> = {
  image: name => ({ kind: 'image', bytes: fixture(name) }),
  audio: name => ({ kind: 'audio', bytes: fixture(name) }),
  video: name => {
    assert.equal(name, 'video/red-square-moves.mp4', 'a video part is the committed clip');
    return clip();
  },
};

/** Every lone image, clip and recording at the projector's own budget: what a query searches. */
const searchable = reference.entries.filter(entry => entry.kind in media && entry.image_tokens === undefined);

/** The parts an entry names, in order: a mixed entry's own, or its one image, clip or recording. */
const partsOf = (entry: ReferenceEntry): Part[] =>
  entry.parts ?? [{ [entry.kind]: entry[entry.kind as 'image' | 'video' | 'audio'] } as Part];

/** Ordered parts as the addon takes them: a prompt with a marker per media part, and those inputs in order. */
function inputOf(parts: Part[]): [string, MultimodalInput[]] {
  const prompt = parts.map(part => ('text' in part ? part.text : MARKER)).join('');
  const inputs = parts.flatMap(part => Object.entries(part).filter(([kind]) => kind !== 'text').map(([kind, name]) => media[kind](name)));
  return [prompt, inputs];
}

// The one context shared across cases. The reference was read at this size, so every input fits one dispatch.
let ctx: EmbeddingContext;
before(async () => { ctx = await embeddingContext(); });
after(async () => { await ctx?.dispose(); });

const cosine = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  dot(a, b) / Math.sqrt(dot(a, a) * dot(b, b));

/** An entry's vector: its text alone, or its parts with room for every committed recording. */
const embedEntry = (context: EmbeddingContext, entry: ReferenceEntry) =>
  entry.kind === 'text' ? embed(context, entry.text!) : embed(context, ...inputOf(partsOf(entry)), ROOM);

/** Each entry embedded once, on its own image budget's context: on a CPU runner the largest take minutes. */
const embedded = new Map<string, Promise<Float32Array>>();
const vectorOf = (entry: ReferenceEntry) => {
  if (!embedded.has(entry.name)) {
    embedded.set(entry.name, entry.image_tokens === undefined
      ? embedEntry(ctx, entry)
      : withContext(embeddingContext(8192, entry.image_tokens), context => embedEntry(context, entry)));
  }
  return embedded.get(entry.name)!;
};

function assertMatches(vector: Float32Array, entry: ReferenceEntry): void {
  assert.equal(vector.length, reference.dimension, entry.name);
  const similarity = cosine(vector, entry.vector);
  assert.ok(similarity >= 0.999, `${entry.name}: ${similarity}`);
}

const vectorsOf = (entries: ReferenceEntry[]) => oneAtATime(entries, vectorOf);

/** The searchable input a vector is closest to. */
async function nearest(vector: Float32Array): Promise<string> {
  const scores = (await vectorsOf(searchable)).map(candidate => dot(vector, candidate));
  return searchable[scores.indexOf(Math.max(...scores))].name;
}

test('the projector carries a vision and an audio encoder', () => {
  assert.equal(ctx.supportsVision(), true);
  assert.equal(ctx.supportsAudio(), true);
});

test('the vector is as wide as the model outputs', () => {
  assert.equal(ctx.getEmbeddingDimension(), reference.dimension);
});

test('every input the reference names embeds where llama.cpp puts it: text in every task framing and language, images at every budget, the clip, the recording and mixed inputs', async () => {
  for (const entry of checked) assertMatches(await vectorOf(entry), entry);
});

test('each query finds its own image, clip or recording, in any language', { skip: searchSkipped }, async () => {
  for (const query of queries) assert.equal(await nearest(await vectorOf(query)), query.finds, query.name);
});

test('a window from the middle of the clip, at its own times, is what the clip\'s query finds over every still', { skip: searchSkipped }, async () => {
  const window = await embed(ctx, MARKER, [clip(8)]);
  const query = await vectorOf(queries.find(entry => entry.finds === 'video/red-square-moves.mp4')!);
  const stills = await vectorsOf(searchable.filter(entry => entry.kind !== 'video'));
  assert.ok(stills.every(still => dot(query, still) < dot(query, window)));
});

test('a mixed input costs its parts and the model\'s framing', async () => {
  const parts = partsOf(named('text, image, video and audio'));
  const framing = (await ctx.tokenize('', true)).length;
  const alone: number[] = [];
  for (const part of parts) alone.push(await ctx._cellsMultimodal([], ...inputOf([part]), ROOM));
  const whole = await ctx._cellsMultimodal([], ...inputOf(parts), ROOM);
  // Each part measured alone carries the model's framing; the whole carries it once.
  assert.equal(whole, alone.reduce((sum, cells) => sum + cells - framing, 0) + framing);
});

test('the image budget sets what an image costs', async () => {
  const pinned = reference.entries.filter(entry => entry.image_tokens !== undefined)
    .sort((a, b) => a.image_tokens! - b.image_tokens!);
  assert.ok(pinned.length >= 2);
  const cost = (entry: ReferenceEntry, context: EmbeddingContext) =>
    context._cellsMultimodal([], ...inputOf(partsOf(entry)), ROOM);
  const pinnedCost = (entry: ReferenceEntry) =>
    withContext(embeddingContext(8192, entry.image_tokens), context => cost(entry, context));
  const [smallest, largest] = [pinned[0], pinned[pinned.length - 1]];
  const ownBudget = await cost(smallest, ctx);
  assert.ok(await pinnedCost(smallest) < ownBudget);
  assert.ok(ownBudget < await pinnedCost(largest));
});

test('the measure is exactly what one dispatch admits, and a refusal leaves the context as it was', async () => {
  const cat = named('cat.jpg');
  const [prompt, catInputs] = inputOf(partsOf(cat));
  const cells = await ctx._cellsMultimodal([], prompt, catInputs);

  await withContext(embeddingContext(cells - 1), async short => {
    await assert.rejects(short.encodeMultimodal(prompt, catInputs), /must go in one dispatch/);
    assertMatches(await embedEntry(short, queries[0]), queries[0]);
  });
  assertMatches(await withContext(embeddingContext(cells), exact => embedEntry(exact, cat)), cat);
});

test('a recording over its limits is refused, and the context embeds as it did', async () => {
  const recording = named('asr-counting.wav');
  const [prompt, inputs] = inputOf(partsOf(recording));
  await assert.rejects(ctx.encodeMultimodal(prompt, inputs, { maxBytes: 16, maxSamples: ROOM.maxSamples }), /byte/i);
  await assert.rejects(ctx.encodeMultimodal(prompt, inputs), /limit|budget/i);
  assertMatches(await embedEntry(ctx, recording), recording);
});

test('a prompt whose markers do not match its inputs is refused, and the context embeds as it did', async () => {
  const cat = named('cat.jpg');
  const [, inputs] = inputOf(partsOf(cat));
  await assert.rejects(ctx.encodeMultimodal(MARKER + MARKER, inputs), /marker/i);
  assertMatches(await embedEntry(ctx, cat), cat);
});

test('encodeMultimodal is refused on a context without a projector, or one not made for embeddings', async () => {
  const addon = loadLocalAddon();
  const [prompt, inputs] = inputOf(partsOf(named('cat.jpg')));
  for (const options of [
    { modelPath, nCtx: 512, nBatch: 512, nThreads: 4, embeddings: true, poolingType: 1 },
    { modelPath, mmprojPath, nCtx: 512, nBatch: 512, nThreads: 4 },
  ]) {
    await withContext(addon.createContext(options) as Promise<EmbeddingContext>, context => {
      assert.throws(() => context.encodeMultimodal(prompt, inputs), /embedding context with a projector/);
    });
  }
});

test('a getter that disposes the context while a call reads its input lets the call finish, then frees the context', async () => {
  const image = fixture('cat.jpg');
  /** `target` with `key` read through a getter that disposes `context` before it answers `value`. */
  const disposingAt = <T extends object>(context: EmbeddingContext, target: T, key: string, value: unknown): T =>
    Object.defineProperty(target, key, { get() { void context.dispose(); return value; }, enumerable: true });
  const reads: Record<string, (context: EmbeddingContext) => Promise<unknown>> = {
    'an input read by encodeMultimodal': context =>
      context.encodeMultimodal(MARKER, [disposingAt(context, { bytes: image }, 'kind', 'image') as MultimodalInput]),
    'an input read by _cellsMultimodal': context =>
      context._cellsMultimodal([], MARKER, [disposingAt(context, { bytes: image }, 'kind', 'image') as MultimodalInput]),
    'audio limits read by encodeMultimodal': context =>
      context.encodeMultimodal(MARKER, [{ kind: 'image', bytes: image }],
        disposingAt(context, { maxSamples: ROOM.maxSamples }, 'maxBytes', ROOM.maxBytes) as AudioLimits),
  };
  for (const [read, call] of Object.entries(reads)) {
    const context = await embeddingContext(512);
    const order: string[] = [];
    // The getter's dispose() and this one return the same Promise, resolved once the context is freed.
    await Promise.all([
      call(context).then(() => order.push('call resolved')),
      context.dispose().then(() => order.push('freed')),
    ]);
    assert.deepEqual(order, ['call resolved', 'freed'], read);
    assert.throws(() => context.getEmbeddingDimension(), /disposed/, read);
  }
});

test('a context disposed while an encode runs frees its memory only once the encode settles', async () => {
  // In a child process: before the fix this can crash outright, which only a separate process survives to report.
  const child = spawn(process.execPath, ['--expose-gc', '--import', 'tsx', resolve(__dirname, '__dispose-in-flight-child.ts')], {
    env: process.env, stdio: ['ignore', 'pipe', 'pipe'],
  });
  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (d: Buffer) => (stdout += d.toString()));
  child.stderr.on('data', (d: Buffer) => (stderr += d.toString()));
  const [code, signal] = await new Promise<[number | null, NodeJS.Signals | null]>((done, fail) => {
    child.on('close', (c, s) => done([c, s]));
    child.on('error', fail);
  });
  assert.equal(signal, null, `child killed by ${signal}\n${stderr.slice(-2000)}`);
  assert.equal(code, 0, `child exited ${code}\n${stdout}\n${stderr.slice(-2000)}`);
  const report = JSON.parse(stdout.trim().split('\n').pop()!);
  assert.deepEqual(report.disposed, {
    returnsPromise: true,
    samePromise: true,
    refused: true,
    order: ['encode resolved', 'freed'],
  }, stderr.split('\n').filter(line => /fail|error|alloc|memory/i.test(line)).slice(-20).join('\n'));
  assert.equal(report.collected, 'resolved');
});
