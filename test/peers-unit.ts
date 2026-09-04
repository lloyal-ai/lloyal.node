/**
 * The binding's peers must admit the set it ships beside — no GPU, no network.
 * tsx + node:assert (the repo's test convention; run via `npm run test:unit`).
 *
 * dist/index.js re-exports values from @lloyal-labs/sdk and @lloyal-labs/lloyal-agents,
 * so both are real peers. A plain range (`>=3.0.0`) never admits a prerelease:
 * semver matches a prerelease only when a comparator names that exact
 * major.minor.patch with a prerelease tag. So the alpha set (sdk 4.0.0-alpha.N,
 * agents 6.0.0-alpha.N) failed every `npm install` that pinned it beside this
 * binding — the scaffold's own install included — until the ranges named the
 * tuples. Checked with the semver library npm resolves with.
 */
import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { satisfies } from 'semver';

// CommonJS on purpose: tsconfig.test.json compiles the tests as CommonJS, so
// `__dirname`, not `import.meta.url` — the compile gate, not tsx, is the judge.
const pkg = JSON.parse(readFileSync(join(__dirname, '..', 'package.json'), 'utf8')) as {
  peerDependencies: Record<string, string>;
};
const peers = pkg.peerDependencies;

/** Each peer: the stable a user has today, the prerelease this arc ships, the stable it becomes. */
const ADMITS: Record<string, string[]> = {
  '@lloyal-labs/sdk': ['3.1.0', '4.0.0-alpha.3', '4.0.0'],
  '@lloyal-labs/lloyal-agents': ['5.5.1', '6.0.0-alpha.3', '6.0.0'],
};
for (const [name, versions] of Object.entries(ADMITS)) {
  assert.ok(peers[name], `${name} is declared as a peer`);
  for (const v of versions) {
    assert.ok(satisfies(v, peers[name]), `${name} "${peers[name]}" admits ${v}`);
  }
}
// The trap, stated: a plain range excludes the prerelease this set is made of.
assert.equal(satisfies('6.0.0-alpha.3', '>=3.0.0'), false);
console.log('peers-unit: ok');
