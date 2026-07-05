/**
 * Stderr egress canary — the RUNTIME half of the boot-only-logging
 * invariant (static half: scripts/check-stderr-allowlist.js).
 *
 * Spawns a child that runs real inference with a canary string riding the
 * prompt, then asserts the child's stderr never contains it. This covers
 * every logging layer at once — the addon's cerr sites, liblloyal (compiled
 * out in release), and llama.cpp's own INFO-level logger — because it
 * observes fd 2 itself rather than trusting any one layer's discipline.
 * Prompt/token content reaching process logs would be a data-egress bug.
 *
 * The child's exit code is NOT hard-asserted: llama.cpp has a known
 * exit-time Metal teardown assert on some darwin boxes (llama.cpp#17869)
 * that is out of scope here. Non-vacuousness comes from requiring the
 * CANARY_RUN_COMPLETE marker — inference must have actually run.
 */

import { spawn } from 'node:child_process';
import * as path from 'node:path';

const CANARY = 'CANARY-9f3adb-lloyal-egress-probe';

async function main(): Promise<void> {
  const child = spawn('npx', ['tsx', path.join(__dirname, '__stderr-canary-child.ts')], {
    cwd: path.join(__dirname, '..'),
    env: { ...process.env, STDERR_CANARY: CANARY },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (d: Buffer) => (stdout += d.toString()));
  child.stderr.on('data', (d: Buffer) => (stderr += d.toString()));
  const exitCode: number | null = await new Promise((resolve) => child.on('close', resolve));

  const complete = stdout.includes('CANARY_RUN_COMPLETE');
  const leaked = stderr.includes(CANARY);

  console.log(`child exit: ${exitCode}; stderr: ${stderr.split('\n').length} lines`);

  if (!complete) {
    console.error('✗ canary run did not complete — the invariant was not exercised');
    console.error('── child stdout ──\n' + stdout);
    console.error('── child stderr ──\n' + stderr);
    process.exit(1);
  }
  if (leaked) {
    const hits = stderr
      .split('\n')
      .filter((l) => l.includes(CANARY))
      .slice(0, 5);
    console.error('✗ EGRESS: prompt canary appeared on stderr:');
    for (const h of hits) console.error(`  ${h}`);
    process.exit(1);
  }

  console.log('✓ inference ran; prompt canary never reached stderr (boot-only logging holds)');
  console.log('\nPASSED: 1');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
