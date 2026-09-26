#!/usr/bin/env node
/**
 * Tripwire: unconditional stderr/stdout writers in the addon's native
 * sources are allowlisted per-file with EXACT counts.
 *
 * The addon's logging contract is boot-only: BackendManager's once-guarded
 * backend-init line and SessionContext's context-creation lines. Inference
 * hot paths (Branch/decode/sample) must never write to stderr — prompt or
 * token content reaching process logs would be a data-egress bug, not a
 * style issue. The runtime counterpart of this static check is
 * test/stderr-canary.ts, which runs real inference in a child process and
 * asserts the prompt never appears on stderr (that one also covers
 * llama.cpp's own logging, which this file-level scan cannot see).
 *
 * Exact counts (not ceilings) so ANY change to the logging surface —
 * addition, removal, or relocation — must touch this allowlist in the
 * same diff, where review sees it.
 */

const fs = require('fs');
const path = require('path');

const SRC = path.join(__dirname, '..', 'src');

/** file (relative to src/) → exact number of writer call sites. */
const ALLOWLIST = {
  'BackendManager.cpp': 2, // once-guarded backend-init provenance + fatal dladdr failure
  'SessionContext.cpp': 15, // initializeContext (4) + initializeMultimodal (1) + CreateContext (10), all boot-scoped
};

// std::cerr / std::cout streams, fprintf(stderr|stdout, and bare printf —
// but not s(n)printf, which formats to buffers.
const WRITER_RE = /std::cerr|std::cout|fprintf\s*\(\s*std(err|out)|(?<![a-z])printf\s*\(/;

let failed = false;
for (const file of fs.readdirSync(SRC).sort()) {
  if (!/\.(cpp|hpp|h|cc)$/.test(file)) continue;
  const lines = fs.readFileSync(path.join(SRC, file), 'utf8').split('\n');
  const hits = lines
    .map((line, i) => ({ line: line.trim(), n: i + 1 }))
    .filter(({ line }) => !line.startsWith('//') && !line.startsWith('*') && WRITER_RE.test(line));
  const expected = ALLOWLIST[file] ?? 0;
  if (hits.length !== expected) {
    failed = true;
    console.error(
      `[stderr-allowlist] src/${file}: ${hits.length} writer call site(s), allowlist says ${expected}.`,
    );
    for (const { line, n } of hits) console.error(`  src/${file}:${n}: ${line}`);
    console.error(
      '  If this change is deliberate (and boot-scoped!), update ALLOWLIST in scripts/check-stderr-allowlist.js in the same PR.',
    );
  }
}

if (failed) process.exit(1);
console.log('[stderr-allowlist] ✓ native logging surface matches the allowlist');
