import { strict as assert } from 'node:assert';
import { realpathSync } from 'node:fs';
import { loadBinary } from '../dist/index.js';

/**
 * The addon this checkout built, proven to be the one loaded: a suite that
 * resolved an installed binary would pass without exercising this code.
 */
export function loadLocalAddon(): ReturnType<typeof loadBinary> {
  assert.equal(process.env.LLOYAL_LOCAL, '1', 'addon tests must exercise the local native build');
  const addon = loadBinary();
  const binary = realpathSync(require.resolve('../build/Release/lloyal.node'));
  assert.equal(addon, require(binary), 'loader resolves this checkout');
  console.log(`Native addon: ${binary}`);
  console.log(`SDK: ${realpathSync(require.resolve('@lloyal-labs/sdk'))}`);
  return addon;
}

/** A context that lives for `use` alone, freed before this resolves: each holds a model, and any projector, in GPU memory. */
export async function withContext<C extends { dispose(): unknown }, T>(made: Promise<C>, use: (context: C) => Promise<T> | T): Promise<T> {
  const context = await made;
  try {
    return await use(context);
  } finally {
    await context.dispose();
  }
}
