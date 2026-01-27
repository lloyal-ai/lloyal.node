#!/usr/bin/env node
/**
 * Semantic Entropy for Repetition Detection
 *
 * Implementation based on:
 * - Farquhar et al. 2024 "Detecting Hallucinations in LLMs Using Semantic Entropy" (Nature)
 * - Quevedo et al. 2024 "Detecting Hallucinations: A Token Probability Approach"
 *
 * Key insight: Instead of detecting token-level repetition (N-grams), we detect
 * SEMANTIC repetition by clustering generations by meaning and measuring entropy.
 *
 * Architecture:
 * - Main context: Generation model (SmolLM2-1.7B)
 * - Sidecar context: NLI model (slim-nli) for entailment checking
 *
 * Flow:
 * 1. Generate K candidate continuations
 * 2. Cluster by semantic equivalence (bidirectional entailment)
 * 3. Compute entropy over cluster assignments
 * 4. Low entropy = semantic repetition = steer away
 */

import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createContext } from '../../lib/index.js';

import {
  sampleWithStrategy,
  Xoroshiro128Plus,
  SamplerWorkspace,
} from '@lloyal-labs/tsampler';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_MODEL = path.resolve(
  __dirname,
  '../../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf'
);
const NLI_MODEL = path.resolve(__dirname, '../../models/slim-nli.gguf');

/**
 * Semantic Entropy Calculator
 *
 * Uses a sidecar NLI model to cluster generations by semantic equivalence,
 * then computes entropy over cluster assignments.
 */
class SemanticEntropyCalculator {
  constructor(nliCtx) {
    this.nliCtx = nliCtx;
    this.entailmentCache = new Map(); // Cache entailment results
  }

  /**
   * Format prompt for slim-nli model
   * Based on llmware/slim-nli expected format
   */
  formatNLIPrompt(evidence, conclusion) {
    return `<human>: Evidence: ${evidence}
Conclusion: ${conclusion}
<classify> evidence </classify>
<bot>:`;
  }

  /**
   * Check if text1 entails text2 using slim-nli sidecar
   * Returns: 2 = entailment, 1 = neutral, 0 = contradiction
   */
  async checkEntailment(text1, text2) {
    // Check cache first
    const cacheKey = `${text1}|||${text2}`;
    if (this.entailmentCache.has(cacheKey)) {
      return this.entailmentCache.get(cacheKey);
    }

    const prompt = this.formatNLIPrompt(text1, text2);
    const promptTokens = await this.nliCtx.tokenize(prompt);

    // Decode prompt
    await this.nliCtx.decode(promptTokens, 0, 0);

    // Generate response (short - just need classification)
    let response = '';
    let pos = promptTokens.length;
    for (let i = 0; i < 20; i++) {
      const token = this.nliCtx.sample({ temperature: 0.1 });
      if (this.nliCtx.isStopToken(token)) break;
      response += this.nliCtx.tokenToText(token);
      await this.nliCtx.decode([token], pos++, 0);
    }

    // Parse response - slim-nli outputs JSON like {"evidence": ["entails"]}
    let result = 1; // Default to neutral
    const lowerResponse = response.toLowerCase();
    if (lowerResponse.includes('entail')) {
      result = 2;
    } else if (lowerResponse.includes('contradict')) {
      result = 0;
    } else if (lowerResponse.includes('neutral')) {
      result = 1;
    }

    // Clear KV cache for next query
    await this.nliCtx.kvCacheClear();

    this.entailmentCache.set(cacheKey, result);
    return result;
  }

  /**
   * Check bidirectional entailment (semantic equivalence)
   * Two texts are semantically equivalent if:
   * - Neither contradicts the other
   * - At least one entails the other (not both neutral)
   */
  async areSemanticallySimilar(text1, text2) {
    const forward = await this.checkEntailment(text1, text2);
    const backward = await this.checkEntailment(text2, text1);

    // From semantic_entropy.py: equivalent if no contradiction and not both neutral
    const noContradiction = forward !== 0 && backward !== 0;
    const notBothNeutral = !(forward === 1 && backward === 1);

    return noContradiction && notBothNeutral;
  }

  /**
   * Cluster samples by semantic equivalence
   * Returns array of cluster IDs (same ID = same semantic meaning)
   *
   * Algorithm from get_semantic_ids() in semantic_entropy.py
   */
  async clusterBySemantic(samples) {
    const n = samples.length;
    const semanticIds = new Array(n).fill(-1);
    let nextId = 0;

    for (let i = 0; i < n; i++) {
      if (semanticIds[i] === -1) {
        // Assign new cluster ID
        semanticIds[i] = nextId;

        // Find all equivalent samples
        for (let j = i + 1; j < n; j++) {
          if (semanticIds[j] === -1) {
            const equivalent = await this.areSemanticallySimilar(
              samples[i],
              samples[j]
            );
            if (equivalent) {
              semanticIds[j] = nextId;
            }
          }
        }
        nextId++;
      }
    }

    return semanticIds;
  }

  /**
   * Compute entropy from cluster assignment frequencies
   * From cluster_assignment_entropy() in semantic_entropy.py
   *
   * High entropy = diverse meanings = good
   * Low entropy = repetitive meanings = bad (semantic loop)
   */
  clusterAssignmentEntropy(semanticIds) {
    const n = semanticIds.length;
    const counts = new Map();

    for (const id of semanticIds) {
      counts.set(id, (counts.get(id) || 0) + 1);
    }

    let entropy = 0;
    for (const count of counts.values()) {
      const p = count / n;
      entropy -= p * Math.log(p);
    }

    return {
      entropy,
      numClusters: counts.size,
      clusterSizes: Array.from(counts.values()),
    };
  }
}

/**
 * Generate K candidate continuations from the current context state
 *
 * Uses KV cache forking to explore multiple paths without modifying
 * the main sequence (seq 0).
 */
async function generateKCandidates(ctx, K, maxTokens, prng, workspace) {
  const candidates = [];
  const originalPos = ctx.kvSeqPosMax(0);

  for (let k = 0; k < K; k++) {
    // Copy sequence 0 to sequence k+1 for exploration
    const seqId = k + 1;
    ctx.kvSeqCopy(0, seqId);

    let text = '';
    let pos = originalPos + 1;

    for (let t = 0; t < maxTokens; t++) {
      const logits = new Float32Array(ctx.getLogits());

      const token = sampleWithStrategy(logits, {
        params: { temperature: 0.9, topP: 0.95 }, // Higher temp for diversity
        workspace,
        prng,
      });

      if (ctx.isStopToken(token)) break;

      text += ctx.tokenToText(token);
      await ctx.decode([token], pos++, seqId);
    }

    candidates.push(text);

    // Clean up forked sequence
    await ctx.kvCacheRemove(seqId, 0, -1);
  }

  return candidates;
}

async function main() {
  const modelPath = process.argv[2] || DEFAULT_MODEL;

  // Parameters
  const nCtx = 2048;
  const K_SAMPLES = 4; // Number of candidate continuations to generate
  const SAMPLE_TOKENS = 30; // Tokens per candidate
  const CHECK_INTERVAL = 50; // Check semantic entropy every N tokens
  const ENTROPY_THRESHOLD = 0.5; // Below this = semantic repetition

  console.log('='.repeat(60));
  console.log('Semantic Entropy Repetition Detection');
  console.log('='.repeat(60));
  console.log(`Main model: ${path.basename(modelPath)}`);
  console.log(`NLI sidecar: ${path.basename(NLI_MODEL)}`);
  console.log(`K samples: ${K_SAMPLES}`);
  console.log(`Check interval: ${CHECK_INTERVAL} tokens`);
  console.log(`Entropy threshold: ${ENTROPY_THRESHOLD}`);
  console.log('='.repeat(60));

  // Load main generation context
  console.log('\nLoading main model...');
  const mainCtx = await createContext({
    modelPath,
    nCtx: nCtx,
    nSeqMax: K_SAMPLES + 2, // Enable multi-sequence for K candidate branches
  });

  // Load NLI sidecar context
  console.log('Loading NLI sidecar model...');
  const nliCtx = await createContext({
    modelPath: NLI_MODEL,
    contextSize: 512, // Small context for NLI queries
  });

  const semanticCalc = new SemanticEntropyCalculator(nliCtx);

  // tsampler setup
  const prng = new Xoroshiro128Plus(42);
  const workspace = new SamplerWorkspace(256);

  const prompt = `Write a comprehensive guide to machine learning, covering the following topics in detail:

1. Linear Regression
2. Neural Networks
3. Optimization

Begin:

# Machine Learning Guide

## Chapter 1: Linear Regression

`;

  console.log(`\nPrompt: "${prompt.slice(0, 80)}..."`);

  const promptTokens = await mainCtx.tokenize(prompt);
  await mainCtx.decode(promptTokens, 0, 0);

  console.log(`\nPrompt tokens: ${promptTokens.length}`);
  console.log('\nGenerating with semantic entropy monitoring...\n');
  console.log('-'.repeat(60));

  process.stdout.write(prompt);

  let cachePos = promptTokens.length;
  let totalTokens = 0;
  let semanticChecks = 0;
  let lowEntropyDetections = 0;

  const TARGET_TOKENS = 500;

  for (let t = 0; t < TARGET_TOKENS; t++) {
    // Periodic semantic entropy check
    if (t > 0 && t % CHECK_INTERVAL === 0) {
      semanticChecks++;

      console.log(`\n  [Semantic check ${semanticChecks} at token ${t}]`);

      // Generate K candidate continuations
      const candidates = await generateKCandidates(
        mainCtx,
        K_SAMPLES,
        SAMPLE_TOKENS,
        prng,
        workspace
      );

      console.log(`  Candidates:`);
      for (let i = 0; i < candidates.length; i++) {
        const preview = candidates[i].slice(0, 50).replace(/\n/g, ' ');
        console.log(`    [${i}] "${preview}..."`);
      }

      // Cluster by semantic similarity
      const semanticIds = await semanticCalc.clusterBySemantic(candidates);
      const { entropy, numClusters, clusterSizes } =
        semanticCalc.clusterAssignmentEntropy(semanticIds);

      console.log(`  Semantic IDs: [${semanticIds.join(', ')}]`);
      console.log(
        `  Clusters: ${numClusters}, Entropy: ${entropy.toFixed(3)}`
      );

      if (entropy < ENTROPY_THRESHOLD) {
        lowEntropyDetections++;
        console.log(
          `  ⚠️  LOW ENTROPY DETECTED - semantic repetition likely!`
        );

        // In a full implementation, we would steer toward underrepresented clusters
        // For now, just log the detection
      }

      console.log('');
    }

    // Normal token generation
    const logits = new Float32Array(mainCtx.getLogits());

    const token = sampleWithStrategy(logits, {
      params: { temperature: 0.8, topP: 0.9 },
      workspace,
      prng,
    });

    if (mainCtx.isStopToken(token)) {
      console.log('\n[EOS token reached]');
      break;
    }

    process.stdout.write(mainCtx.tokenToText(token));
    await mainCtx.decode([token], cachePos++, 0);
    totalTokens++;
  }

  console.log('\n\n' + '='.repeat(60));
  console.log(`Generated: ${totalTokens} tokens`);
  console.log(`Semantic checks: ${semanticChecks}`);
  console.log(`Low entropy detections: ${lowEntropyDetections}`);
  console.log('='.repeat(60));

  mainCtx.dispose();
  nliCtx.dispose();
}

main().catch((err) => {
  console.error('Error:', err.message);
  console.error(err.stack);
  process.exit(1);
});
