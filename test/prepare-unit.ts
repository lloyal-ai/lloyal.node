/**
 * `npm install` runs `prepare`, which installs the commit hooks. An image copies the manifest alone and installs it
 * with `--omit=dev`, outside a git checkout: there `prepare` must succeed and install nothing, or the image's install
 * fails. So the guard is the manifest's own, with nothing else to copy.
 * tsx + node:assert (the repo's test convention; run via `npm run test:unit`).
 */
import { strict as assert } from 'node:assert';
import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const root = join(__dirname, '..');
const image = mkdtempSync(join(tmpdir(), 'lloyal-prepare-'));
try {
  // What an image copies before its install: the manifest alone, with no dev dependencies and no .git.
  copyFileSync(join(root, 'package.json'), join(image, 'package.json'));
  const run = spawnSync('npm', ['run', 'prepare'], { cwd: image, encoding: 'utf8', shell: process.platform === 'win32' });
  assert.equal(run.status, 0, `prepare failed without dev dependencies:\n${run.stdout}${run.stderr}`);
  assert.equal(existsSync(join(image, 'lefthook.yml')), false);
  console.log('✓ prepare succeeds, and installs nothing, without dev dependencies or a git checkout');
} finally {
  rmSync(image, { recursive: true, force: true });
}
