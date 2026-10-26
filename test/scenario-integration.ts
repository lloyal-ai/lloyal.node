/**
 * Scenarios: EmbeddingGemma 2 on long, real footage with known ground truth.
 *
 * Each video was ingested by the media pipeline (frames at one a second, the
 * soundtrack in windows, a transcript) and is read here by the digest of its
 * root, every blob verified. The ground truth beside each scenario in
 * `test/scenarios/` was confirmed by a person who watched the footage.
 *
 * EmbeddingGemma 2 finds; where a scenario goes on to read what it found,
 * Qwen3.5-4B with its vision projector reads it, loaded beside it as a harness
 * loads them. Each model is named explicitly.
 *
 * Hundreds of encodes per scenario: run on a GPU with `npm run test:scenarios`.
 */
import { strict as assert } from 'node:assert';
import { after, before, describe, test } from 'node:test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { dot, embed, embeddingContext, MARKER, oneAtATime } from './embedding-fixtures';
import { loadLocalAddon, withContext } from './local-addon';
import { visionModel } from './vision-fixtures';
import type { EmbeddingContext } from './embedding-fixtures';
import { openPack, overlaps } from './scenario-pack';
import type { Pack, Span } from './scenario-pack';
import { Branch, buildUserDeltaMultimodal } from '@lloyal-labs/sdk';
import type { AudioLimits, MultimodalInput, SessionContext } from '@lloyal-labs/sdk';

const truth = (name: string) => JSON.parse(readFileSync(resolve(__dirname, 'scenarios', `${name}.json`), 'utf8'));
const search = (query: string) => `task: search result | query: ${query}`;
const passage = (text: string) => `title: none | text: ${text}`;

/** A plate asked for exactly, with a way to say it cannot be read: asked without one, the model invents a plate. */
const askPlate = (vehicle: string) =>
  `What is the number plate of ${vehicle}? If you cannot read every character clearly, answer only: unreadable. Otherwise answer with the plate text only.`;

/** What the vision model answers about one image: greedy, without thinking, on a branch pruned afterwards. */
async function answer(context: SessionContext, image: Buffer, question: string): Promise<string> {
  const delta = buildUserDeltaMultimodal(context, question, [{ kind: 'image', bytes: image }], { enableThinking: false });
  const branch = Branch.create(context, 0, { temperature: 0 });
  try {
    await branch.prefillMultimodal(delta.prompt, delta.bitmaps, delta.sep);
    let text = '';
    for (let produced = 0; produced < 48; produced++) {
      const next = branch.produceSync();
      if (next.isStop) break;
      text += next.text;
      await branch.commit(next.token);
    }
    return text.trim();
  } finally {
    branch.pruneSync();
  }
}

/** The most a soundtrack window holds: what the transcription service admits, 16 MiB and 60 s, at the encoder's 16 kHz. */
const WINDOW_LIMITS: AudioLimits = { maxBytes: 16 << 20, maxSamples: 16_000 * 60 + 1 };

/** The widths EmbeddingGemma 2 is trained to be cut to (Matryoshka). */
const WIDTHS = [768, 256, 128];

/** A vector cut to its first `width` dimensions and made unit length again. */
function truncated(vector: ArrayLike<number>, width: number): number[] {
  const head = Array.from(vector).slice(0, width);
  const norm = Math.sqrt(dot(head, head));
  return head.map(x => x / norm);
}

/** Positions in `vectors` (all of them, or those `among`), best match for `query` first, compared at `width`. */
function ranked(query: ArrayLike<number>, vectors: ArrayLike<number>[], width: number, among = vectors.map((_, i) => i)): number[] {
  const q = truncated(query, width);
  const score = new Map(among.map(i => [i, dot(q, truncated(vectors[i], width))]));
  return [...among].sort((a, b) => score.get(b)! - score.get(a)!);
}

/** Reciprocal rank fusion: each ranking gives an item 1 / (k + its rank), so rankings of different senses combine without comparing their scores. */
function fused(rankings: string[][], k = 60): string[] {
  const score = new Map<string, number>();
  for (const ranking of rankings) ranking.forEach((item, i) => score.set(item, (score.get(item) ?? 0) + 1 / (k + i + 1)));
  return [...score.keys()].sort((a, b) => score.get(b)! - score.get(a)!);
}

// One context for every scenario.
let ctx: EmbeddingContext;
before(async () => { ctx = await embeddingContext(); });
after(async () => { await ctx?.dispose(); });

const embedText = (text: string) => embed(ctx, text);

/** Each item's vector, its one input behind one marker, one after another. */
const embedAll = <T>(items: T[], input: (item: T) => MultimodalInput) =>
  oneAtATime(items, item => embed(ctx, MARKER, [input(item)], WINDOW_LIMITS));

const still = (frame: Pack['frames'][number]): MultimodalInput => ({ kind: 'image', bytes: frame.bytes });

/** The video in consecutive windows of `size` frames, each at its own times. */
function windowsOf(pack: Pack, size: number): (Span & { input: MultimodalInput })[] {
  return Array.from({ length: Math.floor(pack.frames.length / size) }, (_, w) => {
    const frames = pack.frames.slice(w * size, (w + 1) * size);
    return {
      startMs: frames[0].tMs,
      endMs: frames[frames.length - 1].tMs + 1,
      input: { kind: 'video', frames: frames.map(f => f.bytes), timestampsMs: frames.map(f => f.tMs), timestampIntervalMs: pack.timestampIntervalMs },
    };
  });
}

describe('find every cat in a ten-minute pet compilation', () => {
  const scenario = truth('cats-and-dogs');
  const clips: (Span & { shows: 'cat' | 'dog' | 'unclear' })[] = scenario.clips;
  let frames: Pack['frames'];
  let vectors: Float32Array[];
  const queries = new Map<string, Float32Array>();
  before(async () => {
    frames = openPack(scenario.root).frames;
    vectors = await embedAll(frames, still);
    for (const animal of ['cat', 'dog']) queries.set(animal, await embedText(search(`a ${animal}`)));
  });

  const clipOf = (frame: number) => clips.findIndex(c => c.startMs <= frames[frame].tMs && frames[frame].tMs < c.endMs);
  /** Labelled clips in the order their best frames come in. */
  const clipsIn = (frameOrder: number[]) => [...new Set(frameOrder.map(clipOf))].filter(c => clips[c]?.shows !== undefined && clips[c].shows !== 'unclear');
  const clipsOf = (animal: string) => clips.filter(c => c.shows === animal).length;

  test('the clips a query for an animal ranks first all show it', () => {
    for (const [animal, top] of [['cat', 20], ['dog', 10]] as const) {
      const first = clipsIn(ranked(queries.get(animal)!, vectors, 768)).slice(0, top);
      assert.deepEqual(first.filter(c => clips[c].shows !== animal), [], `${animal}: the first ${top} clips`);
    }
  });

  test('nearly every clip of the animal comes before any other', () => {
    for (const [animal, least] of [['cat', 0.9], ['dog', 0.8]] as const) {
      const found = clipsIn(ranked(queries.get(animal)!, vectors, 768)).slice(0, clipsOf(animal));
      const precision = found.filter(c => clips[c].shows === animal).length / clipsOf(animal);
      assert.ok(precision >= least, `${animal}: ${precision} of the first ${clipsOf(animal)} clips`);
    }
  });

  test('a 128-wide scan of every frame, rescored at 768 on the better half, finds the clips the full scan finds', () => {
    for (const animal of ['cat', 'dog']) {
      const query = queries.get(animal)!;
      const n = clipsOf(animal);
      const full = clipsIn(ranked(query, vectors, 768)).slice(0, n);
      const shortlist = ranked(query, vectors, 128).slice(0, Math.ceil(vectors.length / 2));
      const rescored = new Set(clipsIn(ranked(query, vectors, 768, shortlist)).slice(0, n));
      const kept = full.filter(c => rescored.has(c)).length / n;
      assert.ok(kept >= 0.95, `${animal}: ${kept} of the full scan's clips`);
    }
  });
});

describe('find a car and read its plate in half an hour of dash-cam footage', () => {
  const scenario = truth('dash-cam-crashes');
  const target: Span & { query: string; vehicle: string; plateQuery: string; plate: string; legibleMs: number; illegibleMs: number } = scenario.target;
  let pack: Pack;
  let query: Float32Array;
  before(async () => {
    pack = openPack(scenario.root);
    query = await embedText(search(target.query));
  });

  test('a description of the scene finds the moment among every frame, at every width', async () => {
    const vectors = await embedAll(pack.frames, still);
    for (const width of WIDTHS) {
      const first = ranked(query, vectors, width).slice(0, 3);
      assert.ok(first.some(i => overlaps({ startMs: pack.frames[i].tMs, endMs: pack.frames[i].tMs + 1 }, target)), `at ${width}`);
    }
  });

  test('a description of the scene finds the moment among eight-second windows, at every width', async () => {
    const windows = windowsOf(pack, 8);
    const vectors = await embedAll(windows, w => w.input);
    for (const width of WIDTHS) {
      const first = ranked(query, vectors, width).slice(0, 3);
      assert.ok(first.some(i => overlaps(windows[i], target)), `at ${width}`);
    }
  });

  test('within the moment, a query for the plate finds the frame it can be read in', async () => {
    const moment = pack.frames.filter(f => overlaps({ startMs: f.tMs, endMs: f.tMs + 1 }, target));
    const [first] = ranked(await embedText(search(target.plateQuery)), await embedAll(moment, still), 768);
    assert.equal(moment[first].tMs, target.legibleMs);
  });

  test('the plate is read exactly where it is legible, and not guessed where it is too far to read', async () => {
    const frameAt = (tMs: number) => pack.frames.find(f => f.tMs === tMs)!.bytes;
    const read = await withContext(
      loadLocalAddon().createContext({ ...visionModel, nCtx: 4096, nBatch: 512, nSeqMax: 1, nThreads: 4 }) as Promise<SessionContext>,
      async vision => [
        await answer(vision, frameAt(target.legibleMs), askPlate(target.vehicle)),
        await answer(vision, frameAt(target.illegibleMs), askPlate(target.vehicle)),
      ],
    );
    assert.deepEqual(read, [target.plate, 'unreadable']);
  });
});

describe('a question in English finds Hindi speech by its sound, among every soundtrack window', () => {
  const scenario = truth('hindi-movie-dialog');
  const moments: (Span & { query: string })[] = scenario.moments;
  /** Every soundtrack searched: the Hindi clip's six windows among the other videos' English music, speech and lyrics. */
  const searched = ['hindi-movie-dialog', 'cats-and-dogs', 'dash-cam-crashes'];
  type Window = Span & { video: string };
  const windows: Window[] = [];
  const heard: Float32Array[] = [];
  let transcript: Pack['transcript'];
  const said: Float32Array[] = [];
  const questions = new Map<string, Float32Array>();
  before(async () => {
    transcript = openPack(scenario.root).transcript;
    for (const video of searched) {
      const pack = openPack(truth(video).root);
      windows.push(...pack.soundtrack.map(w => ({ video, startMs: w.startMs, endMs: w.endMs })));
      heard.push(...await embedAll(pack.soundtrack, w => ({ kind: 'audio', bytes: w.bytes })));
    }
    for (const segment of transcript) said.push(await embedText(passage(segment.text)));
    for (const moment of moments) questions.set(moment.query, await embedText(search(moment.query)));
  });

  const key = (w: Window) => `${w.video} ${w.startMs}-${w.endMs}`;
  const isMoment = (w: Window, moment: Span) => w.video === 'hindi-movie-dialog' && overlaps(w, moment);

  test('each question finds its moment among the first three by sound alone, with no transcript', () => {
    for (const moment of moments) {
      const first = ranked(questions.get(moment.query)!, heard, 768).slice(0, 3).map(i => windows[i]);
      assert.ok(first.some(w => isMoment(w, moment)), `${moment.query}: ${first.map(key).join(', ')} of ${windows.length} windows`);
    }
  });

  test('fusing that with the Hindi transcript keeps each moment first', () => {
    const segments: Window[] = transcript.map(s => ({ video: 'hindi-movie-dialog', startMs: s.startMs, endMs: s.endMs }));
    const byKey = new Map([...windows, ...segments].map(w => [key(w), w]));
    for (const moment of moments) {
      const question = questions.get(moment.query)!;
      const [best] = fused([ranked(question, heard, 768).map(i => key(windows[i])), ranked(question, said, 768).map(i => key(segments[i]))]);
      assert.ok(isMoment(byKey.get(best)!, moment), `${moment.query}: ${best}`);
    }
  });
});
