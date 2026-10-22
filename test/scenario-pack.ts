import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

/**
 * A scenario pack: a long video ingested by the media pipeline, read from its
 * OCI image layout by the digest of its root manifest. Every blob read is
 * checked against its digest, so the root pins the whole pack.
 *
 * The types below are the media store's video manifest as this suite relies on
 * it; a change to that schema fails here, in one place.
 */

const VIDEO_CONFIG_TYPE = 'application/vnd.lloyal.video.v1+json';
const TRANSCRIPT_MEDIA_TYPE = 'application/vnd.lloyal.transcript.v1+json';
const DIGEST = /^sha256:[0-9a-f]{64}$/;

/** Where the packs are: a local OCI layout, built by hdk's `scripts/scenario-pack.ts` or pulled by digest. */
export const packsDir = process.env.LLOYAL_SCENARIO_PACKS ?? resolve(__dirname, '../models/scenarios');

interface Descriptor { mediaType: string; digest: string; size: number }

interface Timeline {
  width: number;
  height: number;
  durationMs: number;
  frames: { tMs: number; frame: number }[];
  sampling: { timestampIntervalMs: number };
  soundtrack?: { startMs: number; endMs: number; audio: number }[];
}

export interface Span { startMs: number; endMs: number }

export interface Pack {
  durationMs: number;
  /** The interval the frames' time marks were laid out at. */
  timestampIntervalMs: number;
  /** Frames in time order, as the pipeline sampled them. */
  frames: { tMs: number; bytes: Buffer }[];
  /** The soundtrack's windows in time order, as PCM WAV. */
  soundtrack: (Span & { bytes: Buffer })[];
  /** What the transcription service heard, window by window; empty when nothing was. */
  transcript: (Span & { text: string; language?: string })[];
}

/** One blob of the layout, refused unless its bytes hash to its name. */
function blob(dir: string, digest: string): Buffer {
  if (!DIGEST.test(digest)) throw new Error(`not a sha256 digest: ${digest}`);
  const file = join(dir, 'blobs', 'sha256', digest.slice('sha256:'.length));
  if (!existsSync(file)) {
    throw new Error(`${digest} is not in ${dir}: build the pack with hdk's scripts/scenario-pack.ts, or pull it by digest`);
  }
  const bytes = readFileSync(file);
  const actual = `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
  if (actual !== digest) throw new Error(`${digest} has drifted: its bytes hash to ${actual}`);
  return bytes;
}

const json = <T>(dir: string, digest: string): T => JSON.parse(blob(dir, digest).toString('utf8'));

/** The video a root manifest names, every blob verified. */
export function openPack(root: string, dir = packsDir): Pack {
  const manifest = json<{ config: Descriptor; layers: Descriptor[] }>(dir, root);
  if (manifest.config?.mediaType !== VIDEO_CONFIG_TYPE) {
    throw new Error(`${root} is not a video: its config is ${manifest.config?.mediaType}`);
  }
  const timeline = json<Timeline>(dir, manifest.config.digest);
  const layer = (index: number) => blob(dir, manifest.layers[index].digest);
  const heard = manifest.layers.find(l => l.mediaType === TRANSCRIPT_MEDIA_TYPE);
  return {
    durationMs: timeline.durationMs,
    timestampIntervalMs: timeline.sampling.timestampIntervalMs,
    frames: timeline.frames.map(f => ({ tMs: f.tMs, bytes: layer(f.frame) })),
    soundtrack: (timeline.soundtrack ?? []).map(w => ({ startMs: w.startMs, endMs: w.endMs, bytes: layer(w.audio) })),
    transcript: heard ? json<{ segments: Pack['transcript'] }>(dir, heard.digest).segments : [],
  };
}

/** Whether a moment or span overlaps another. */
export const overlaps = (a: Span, b: Span) => a.startMs < b.endMs && b.startMs < a.endMs;
