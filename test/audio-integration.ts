import { strict as assert } from 'node:assert';
import { after, before, test } from 'node:test';
import { readFileSync, realpathSync } from 'node:fs';
import { resolve } from 'node:path';
import { Branch, BranchStore, loadBinary } from '../dist/index.js';
import type { SessionContext, AudioLimits, MultimodalInput } from '@lloyal-labs/sdk';

const fixture = (name: string) => readFileSync(resolve(__dirname, '../liblloyal/tests/fixtures', name));
const recording = fixture('asr-counting.wav');
const limits: AudioLimits = { maxBytes: 400_000, maxSamples: 200_000 };
const audio = (bytes: Uint8Array = recording): MultimodalInput => ({ kind: 'audio', bytes });
let ctx: SessionContext;
let prompt: string;

before(async () => {
  assert.equal(process.env.LLOYAL_LOCAL, '1', 'addon tests must exercise the local native build');
  const addon = loadBinary();
  const binary = realpathSync(require.resolve('../build/Release/lloyal.node'));
  assert.equal(addon, require(binary), 'loader resolves this checkout');
  console.log(`Native addon: ${binary}`);
  console.log(`SDK: ${realpathSync(require.resolve('@lloyal-labs/sdk'))}`);
  ctx = await addon.createContext({
    modelPath: process.env.LLAMA_ASR_MODEL ?? resolve(__dirname, '../models/audio/qwen3-asr/Qwen3-ASR-0.6B-Q8_0.gguf'),
    mmprojPath: process.env.LLAMA_ASR_MMPROJ ?? resolve(__dirname, '../models/audio/qwen3-asr/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf'),
    nCtx: 4096, nBatch: 128, nSeqMax: 8, nThreads: 4,
  });
  ({ prompt } = await ctx.formatChat(JSON.stringify([
    { role: 'user', content: [{ type: 'media_marker', text: '<__media__>' }] },
  ]), { enableThinking: false }));
});

after(() => {
  if (!ctx) return;
  ctx.dispose();
  assert.throws(() => ctx.supportsAudio(), /disposed/i);
  ctx.dispose();
});

const root = () => Branch.create(ctx, 0, { temperature: 0 });
const cells = () => ctx._storeKvPressure().cellsUsed;
const prefill = (branch: Branch, inputs: MultimodalInput[], budget?: AudioLimits) =>
  ctx._storePrefillMultimodal([branch.handle], [[]], [prompt], [inputs], [budget]);
const measure = (inputs: MultimodalInput[], budget?: AudioLimits) =>
  ctx._cellsMultimodal([], prompt, inputs, budget);

function grounded(text: string): void {
  assert.match(text, /one, two, three, four, five/i);
  assert.match(text, /the meeting is on thursday/i);
  assert.match(text, /nine thirty|9:30/i);
}

async function transcripts(branches: Branch[]): Promise<string[]> {
  const store = new BranchStore(ctx);
  const outputs = branches.map(branch => ({ branch, tokens: [] as number[], done: false }));
  for (let step = 0; step < 96; step++) {
    const entries: Array<[Branch, number]> = [];
    for (const output of outputs.filter(output => !output.done)) {
      const { token, isStop } = await output.branch.produce();
      output.done = isStop;
      if (isStop) continue;
      output.tokens.push(token);
      entries.push([output.branch, token]);
    }
    if (entries.length === 0) break;
    await store.commit(entries);
  }
  assert.ok(outputs.every(output => output.done), 'all transcripts reach a stop token');
  return Promise.all(outputs.map(output => ctx.detokenize(output.tokens)));
}

test('audio capability and sample rate come from the loaded projector', () => {
  assert.equal(ctx.supportsAudio(), true);
  assert.equal(ctx.supportsVision(), false);
  assert.equal(ctx.audioSampleRate(), 16000);
});

test('audio prefill owns byte-offset views and forks inherit the projected parent', async () => {
  const parent = root();
  const children: Branch[] = [];
  try {
    const expected = await measure([audio()], limits);
    assert.ok(expected > 0);
    assert.equal(cells(), 0, 'measurement does not mutate KV');
    const storage = new Uint8Array(recording.length + 16);
    const view = storage.subarray(8, 8 + recording.length);
    view.set(recording);
    const pending = prefill(parent, [audio(view)], limits);
    storage.fill(0); // Native work must own its copy before the JS call returns.
    const [result] = await pending;
    assert.equal(result.error, undefined);
    assert.equal(result.tokensDecoded, expected);
    assert.equal(result.positionAdvance, expected);
    assert.equal(parent.position, expected);
    assert.equal(cells(), expected);
    for (let i = 0; i < 4; i++) children.push(await parent.fork());
    assert.equal(cells(), expected, 'forks add no copy of the projected prefix');
    (await transcripts(children)).forEach(grounded);
    const suffixCells = children.reduce((sum, child) => sum + child.position - expected, 0);
    assert.equal(cells(), expected + suffixCells);
  } finally {
    for (const child of children) await child.prune();
    await parent.prune();
  }
  assert.equal(cells(), 0, 'pruning reclaims prefix and suffixes');
});

test('measurement owns a Buffer slice through worker completion', async () => {
  const expected = await measure([audio()], limits);
  const storage = Buffer.alloc(recording.length + 32);
  recording.copy(storage, 16);
  const pending = measure([audio(storage.subarray(16, 16 + recording.length))], limits);
  storage.fill(0);
  assert.equal(await pending, expected);
  assert.equal(cells(), 0);
});

test('audio admission errors leave branches untouched and valid cohort entries succeed', async () => {
  const cases = [
    { name: 'missing limits', inputs: [audio()], budget: undefined, error: /limit|budget/i },
    { name: 'byte budget', inputs: [audio()], budget: { ...limits, maxBytes: recording.length - 1 }, error: /byte/i },
    { name: 'sample budget', inputs: [audio()], budget: { ...limits, maxSamples: 10 }, error: /sample/i },
    { name: 'invalid WAV', inputs: [audio(Buffer.from('not a recording'))], budget: limits, error: /wav|riff|audio/i },
    { name: 'audio labelled image', inputs: [{ kind: 'image', bytes: recording } as MultimodalInput], budget: limits, error: /audio|vision|image/i },
    { name: 'legacy bytes are images', inputs: [recording], budget: limits, error: /audio|vision|image/i },
    { name: 'marker mismatch', inputs: [], budget: limits, error: /marker/i },
  ];
  const branches = cases.map(() => root());
  const valid = root();
  try {
    const expected = await measure([audio()], limits);
    const results = await ctx._storePrefillMultimodal(
      [...branches, valid].map(branch => branch.handle),
      [...cases, null].map(() => []), [...cases, null].map(() => prompt),
      [...cases.map(entry => entry.inputs), [audio()]],
      [...cases.map(entry => entry.budget), limits],
    );
    cases.forEach((entry, index) => {
      assert.match(results[index].error ?? '', entry.error, entry.name);
      assert.equal(results[index].tokensDecoded, 0, entry.name);
      assert.equal(results[index].rc, undefined, entry.name);
      assert.equal(branches[index].position, 0, entry.name);
    });
    assert.equal(results.at(-1)?.error, undefined);
    assert.equal(cells(), expected, 'only the valid entry was admitted');
    grounded((await transcripts([valid]))[0]);
  } finally {
    for (const branch of [...branches, valid]) await branch.prune();
  }
  assert.equal(cells(), 0);
});

test('invalid limits and unsupported typed arrays are rejected before inference', async () => {
  const budgets = [0, -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1];
  for (const maxBytes of budgets) {
    await assert.rejects(async () => measure([audio()], { ...limits, maxBytes }), /limit|budget|integer|positive|byte/i);
  }
  for (const maxSamples of budgets) {
    await assert.rejects(async () => measure([audio()], { ...limits, maxSamples }), /limit|budget|integer|positive|sample/i);
  }
  await assert.rejects(async () => measure([audio(new Int32Array(10) as unknown as Uint8Array)], limits), /Uint8Array|Buffer/i);
  await assert.rejects(async () => measure([{ kind: 'video', bytes: recording } as unknown as MultimodalInput], limits), /kind/i);
  assert.equal(cells(), 0);
});

test('a partial audio decode reports rc and partial, then pruning reclaims landed rows', async () => {
  const branch = root();
  try {
    const dot = (await ctx.tokenize('.', false))[0];
    await branch.prefill(Array.from({ length: 4096 - 32 }, () => dot));
    const [result] = await prefill(branch, [audio(fixture('asr-repeated.wav'))], limits);
    assert.match(result.error ?? '', /decode|capacity|space/i);
    assert.equal(result.rc, 1);
    assert.equal(result.partial, true);
  } finally {
    await branch.prune();
  }
  assert.equal(cells(), 0);
  const valid = root();
  try {
    assert.equal((await prefill(valid, [audio()], limits))[0].error, undefined);
    grounded((await transcripts([valid]))[0]);
  } finally {
    await valid.prune();
  }
});

test('audio byte budgets apply across recordings in one prompt', async () => {
  const branch = root();
  try {
    const [result] = await ctx._storePrefillMultimodal(
      [branch.handle], [[]], [`${prompt} <__media__>`], [[audio(), audio()]],
      [{ ...limits, maxBytes: recording.length }],
    );
    assert.match(result.error ?? '', /byte/i);
    assert.equal(branch.position, 0);
    assert.equal(cells(), 0);
  } finally {
    await branch.prune();
  }
});
