/**
 * `npm install` runs `prepare`, which installs the commit hooks. An image installs the package with
 * `--omit=dev` and no git checkout: there `prepare` must succeed and install nothing, or the image's install fails.
 * tsx + node:assert (the repo's test convention; run via `npm run test:unit`).
 */
import { strict as assert } from 'node:assert';
import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const root = join(__dirname, '..');
const image = mkdtempSync(join(tmpdir(), 'lloyal-prepare-'));
try {
  // What an image copies before its install: the manifest and the scripts, with no dev dependencies and no .git.
  mkdirSync(join(image, 'scripts'));
  copyFileSync(join(root, 'package.json'), join(image, 'package.json'));
  copyFileSync(join(root, 'scripts', 'install-hooks.js'), join(image, 'scripts', 'install-hooks.js'));
  const run = spawnSync('npm', ['run', 'prepare'], { cwd: image, encoding: 'utf8', shell: process.platform === 'win32' });
  assert.equal(run.status, 0, `prepare failed without dev dependencies:\n${run.stdout}${run.stderr}`);
  assert.equal(existsSync(join(image, 'lefthook.yml')), false);
  console.log('✓ prepare succeeds, and installs nothing, without dev dependencies or a git checkout');
} finally {
  rmSync(image, { recursive: true, force: true });
}
