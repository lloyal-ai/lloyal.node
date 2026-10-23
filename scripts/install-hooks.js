/**
 * Installs the commit hooks in a contributor's checkout, and does nothing anywhere else.
 *
 * `npm install` runs this as `prepare`. An install that omits dev dependencies (an image's `--omit=dev`) has no
 * lefthook, and one outside a git checkout has no hooks to install: either is skipped, so the install succeeds.
 */
const { existsSync } = require('node:fs');
const { execFileSync } = require('node:child_process');
const { join } = require('node:path');

let lefthook;
try {
  lefthook = require.resolve('lefthook');
} catch {
  process.exit(0);
}
if (!existsSync(join(__dirname, '..', '.git'))) process.exit(0);
execFileSync(process.execPath, [lefthook, 'install'], { stdio: 'inherit' });
