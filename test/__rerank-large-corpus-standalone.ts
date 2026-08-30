/**
 * Standalone runner for testRerankLargeCorpus — scores the 20-doc corpus
 * without paying for the rest of the integration suite. Useful for bisecting
 * the ranking against model quant, KV quant and nSeqMax.
 *
 * Usage:
 *   LLAMA_RERANK_MODEL=models/qwen3-reranker-0.6b-q8.gguf \
 *     npx tsx test/__rerank-large-corpus-standalone.ts
 */

import * as path from 'node:path';
import * as fs from 'node:fs';
import { createContext, Rerank } from '../dist/index.js';

const RERANK_MODEL_PATH =
  process.env.LLAMA_RERANK_MODEL ||
  path.join(__dirname, '../models/qwen3-reranker-0.6b-q8.gguf');

if (!fs.existsSync(RERANK_MODEL_PATH)) {
  console.error(`Model not found: ${RERANK_MODEL_PATH}`);
  process.exit(1);
}

function assert(cond: boolean, msg: string): void {
  if (!cond) {
    console.error(`FAIL: ${msg}`);
    process.exit(1);
  }
}

async function main() {
  console.log(`Model: ${path.basename(RERANK_MODEL_PATH)}\n`);

  // nSeqMax=10 → trunk + queryBranch + 8 effective leaves. testRerankLargeCorpus
  // hardcodes 8, which under the 2-lease tax for trunk+queryBranch yields only
  // 6 effective leaves.
  //
  // That lease tax is a real difference in batching, but it is NOT what decided
  // the old ranking failure. Measured on q4_k_m + q4_0 KV, the relevant doc
  // scores -5.5427 at BOTH nSeqMax 8 and 10 — identical to four decimals. Only
  // the distractors shuffle (doc 13: -1.347 -> -3.698), moving it 5th to 4th.
  // The judge rejects the correct document either way; the cause is stacked
  // quant loss (see RERANK_MODEL_PATH in test/integration.ts), not tiling.
  const rerankCtx = await createContext({
    modelPath: RERANK_MODEL_PATH,
    nCtx: 4096,
    nSeqMax: 10,
    typeK: 'q4_0',
    typeV: 'q4_0',
  });
  const rerank = await Rerank.create(rerankCtx, { nSeqMax: 10, nCtx: 4096 });

  try {
    const query = 'What is the capital of France?';
    const relevantDoc = 'Paris is the capital and most populous city of France.';

    // Build 20 documents: 1 relevant + 19 irrelevant — verbatim from the
    // original testRerankLargeCorpus fixture.
    const docTexts: string[] = [
      relevantDoc,
      'The Amazon rainforest produces about 20% of the world\'s oxygen.',
      'Berlin is the capital of Germany and its largest city.',
      'The Great Wall of China is over 13,000 miles long.',
      'Tokyo is the most populous metropolitan area in the world.',
      'The Sahara Desert is the largest hot desert in the world.',
      'Mount Everest is the highest mountain above sea level.',
      'The Pacific Ocean is the largest and deepest ocean.',
      'Antarctica is the coldest continent on Earth.',
      'The Nile is traditionally considered the longest river.',
      'Australia is both a country and a continent.',
      'The human body contains approximately 206 bones.',
      'Jupiter is the largest planet in our solar system.',
      'The speed of light is approximately 299,792 kilometers per second.',
      'DNA was first identified by Friedrich Miescher in 1869.',
      'The International Space Station orbits Earth every 90 minutes.',
      'Honey never spoils due to its low moisture content.',
      'Venice is built on more than 100 small islands.',
      'The deepest point in the ocean is the Mariana Trench.',
      'Photosynthesis converts carbon dioxide and water into glucose.',
    ];

    const tokenized: number[][] = await Promise.all(
      docTexts.map((d) => rerank.tokenize(d)),
    );
    assert(tokenized.length === 20, '20 documents tokenized');

    let results!: { score: number; index: number }[];
    let progressCount = 0;
    for await (const p of rerank.score(query, tokenized)) {
      progressCount++;
      assert(p.total === 20, `total is 20 (got ${p.total})`);
      assert(p.filled <= p.total, `filled ${p.filled} <= total ${p.total}`);
      results = p.results;
    }
    assert(
      progressCount >= 3,
      `≥3 progress updates for 20 docs / nSeqMax=8 (got ${progressCount})`,
    );
    assert(results.length === 20, 'all 20 results returned');

    // Scores sorted descending
    for (let i = 1; i < results.length; i++) {
      assert(
        results[i].score <= results[i - 1].score,
        `sorted descending at index ${i} (${results[i - 1].score} → ${results[i].score})`,
      );
    }

    // Relevant doc (index 0) should rank in top 3 — the original B1 failure
    // was rank=3 (off-by-one); the R3 rounding fix should restore < 3.
    const relevantRank = results.findIndex((r) => r.index === 0);

    console.log('\nFull ranking:');
    for (let i = 0; i < results.length; i++) {
      const r = results[i];
      const marker = r.index === 0 ? '  ✓ RELEVANT' : '';
      console.log(`  ${i}: idx=${r.index} score=${r.score.toFixed(4)}${marker}`);
    }

    assert(
      relevantRank < 3,
      `relevant doc ranks ${relevantRank} (expected < 3)`,
    );

    console.log('\nTop 5 results:');
    for (let i = 0; i < 5; i++) {
      const r = results[i];
      const marker = r.index === 0 ? '  ✓ RELEVANT' : '';
      console.log(`  ${i}: idx=${r.index} score=${r.score.toFixed(4)}${marker}`);
    }

    // topK across group boundary
    let top5!: { score: number; index: number }[];
    for await (const p of rerank.score(query, tokenized, 5)) {
      top5 = p.results;
    }
    assert(top5.length === 5, `topK=5 returns 5 results (got ${top5.length})`);
    assert(
      top5[0].score === results[0].score && top5[0].index === results[0].index,
      'topK=5 top result matches full ranking',
    );

    console.log(
      `\nPASS: 20 docs / nSeqMax=8 → relevant doc rank ${relevantRank}, progressCount=${progressCount}`,
    );
  } finally {
    rerank.dispose();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
