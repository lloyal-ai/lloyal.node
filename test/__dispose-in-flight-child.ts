/**
 * Child process for the dispose case in test/embedding-integration.ts: frees
 * an embedding context while an encode is running on the native pool, and
 * lets a second one be collected mid-encode. A failure here can be a crash,
 * which only a separate process survives to report, so this prints what it
 * saw as one JSON line on stdout and the parent judges it.
 *
 * Run with --expose-gc.
 */

import { clip, embeddingContext, fixture, MARKER } from './embedding-fixtures';
import type { MultimodalInput } from '@lloyal-labs/sdk';

/** The largest committed input: the infographic and the clip, over three thousand cells of encoder work. */
const prompt = MARKER + MARKER;
const inputs = (): MultimodalInput[] => [
  { kind: 'image', bytes: fixture('embedding/Global-X-Digital-Evolution-Series-Aug-25-Part-2.jpg') },
  clip(),
];

const outcome = (work: Promise<unknown>) => work.then(() => 'resolved', (e: unknown) => `rejected: ${String(e)}`);

/** dispose() while the encode is on the pool: what each call gave, and the order things settled in. */
async function disposedMidEncode() {
  const ctx = await embeddingContext();
  // Each settlement is recorded at the same depth, so the order is the order they settled in.
  const order: string[] = [];
  const encoding = ctx.encodeMultimodal(prompt, inputs());
  void outcome(encoding).then((settled) => order.push(`encode ${settled}`));
  const disposing: unknown = ctx.dispose();
  const again: unknown = ctx.dispose();
  const freed = disposing instanceof Promise;
  if (freed) void outcome(disposing).then(() => order.push('freed'));
  const refused = (() => {
    try { ctx.getEmbeddingDimension(); return false; } catch (e) { return /disposed/i.test(String(e)); }
  })();
  await outcome(encoding);
  if (freed) await disposing;
  return { returnsPromise: freed, samePromise: disposing === again, refused, order };
}

/** The last reference dropped while the encode is on the pool, and the collector run until it settles. */
async function collectedMidEncode() {
  const encoding = outcome((await embeddingContext()).encodeMultimodal(prompt, inputs()));
  let settled: string | undefined;
  void encoding.then((s) => { settled = s; });
  while (settled === undefined) {
    global.gc!();
    await new Promise((r) => setTimeout(r, 1));
  }
  return settled;
}

async function main() {
  const disposed = await disposedMidEncode();
  const collected = await collectedMidEncode();
  process.stdout.write(`${JSON.stringify({ disposed, collected })}\n`);
}

main().catch((e) => {
  process.stdout.write(`${JSON.stringify({ error: String(e) })}\n`);
  process.exit(1);
});
