# Streaming Examples

Advanced streaming patterns for long-form generation with quality preservation.

## Examples Overview

| Example | Purpose | Key Pattern |
|---------|---------|-------------|
| `streaming.mjs` | Infinite context generation | BlinkKV reseeding |
| `streaming-tsampler.mjs` | TypeScript sampling with N-gram tracking | TTA (Test-Time Alignment) |
| `streaming-semantic-entropy.mjs` | Semantic repetition detection | Sidecar NLI model |

---

## streaming.mjs - BlinkKV Infinite Context

Demonstrates generating beyond the context window limit using the BlinkKV reseeding pattern.

### Usage

```bash
node streaming.mjs /path/to/model.gguf
```

### Parameters (from BlinkKV paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Context size | 2048 | Model's context window |
| Sink tokens | prompt | Structural anchor (entire prompt) |
| Tail size | 256 | Most recent tokens to retain |

### BlinkKV Pattern

When the KV cache fills:
1. **Clear** the entire KV cache
2. **Re-decode sinks** (prompt tokens) at positions [0..N]
3. **Re-decode tail** (256 most recent) at positions [N+1..N+256]
4. **Continue** from position N+257

This maintains cache-local position contiguity, which is necessary and sufficient for streaming quality.

### Key APIs

| Method | Description |
|--------|-------------|
| `clearAndReseed(sinks, tail)` | Clear cache, re-decode at local positions |
| `modelSurprisal(token)` | Measure prediction error |
| `createPerplexityTracker()` | Track quality across stream |

---

## streaming-tsampler.mjs - TypeScript Sampling with N-gram Tracking

Demonstrates using tsampler (TypeScript sampling library) with N-gram sequence tracking for repetition detection.

### Usage

```bash
node streaming-tsampler.mjs /path/to/model.gguf
```

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Native Context (llama.cpp)                             │
│  - KV cache management                                  │
│  - Logits computation via decode()                      │
│  - BlinkKV reseeding                                    │
└─────────────────────────────────────────────────────────┘
                    │ ctx.getLogits()
                    ▼
┌─────────────────────────────────────────────────────────┐
│  tsampler (TypeScript)                                  │
│  - sampleWithStrategy() for token selection             │
│  - Temperature, top-p, top-k filtering                  │
│  - Xoroshiro128Plus PRNG for reproducibility            │
└─────────────────────────────────────────────────────────┘
                    │ sampled token
                    ▼
┌─────────────────────────────────────────────────────────┐
│  NgramTracker (App-level)                               │
│  - Tracks N-gram sequences (configurable N)             │
│  - Threshold-based blocking (block after K repeats)     │
│  - Logit steering: blocked token → -Infinity            │
└─────────────────────────────────────────────────────────┘
```

### Key Insight: Token vs Sequence Penalties

llama.cpp's built-in repetition penalties operate at the **token level**, penalizing individual words regardless of context. This degrades prose quality over long generations as common words ("the", "is", "a") accumulate penalties.

Instead, tsampler + N-gram tracking operates at the **sequence level**:
- Only blocks when an exact N-token sequence repeats
- Threshold-based: only blocks after K occurrences (not first occurrence)
- Preserves natural word reuse while preventing actual loops

### tsampler Integration

```javascript
import {
  sampleWithStrategy,
  Xoroshiro128Plus,
  SamplerWorkspace,
} from 'tsampler';

const prng = new Xoroshiro128Plus(42);  // Deterministic seed
const workspace = new SamplerWorkspace(256);

// Get logits from native layer
const logits = new Float32Array(ctx.getLogits());

// Apply N-gram blocking before sampling
const blockedToken = ngramTracker.getBlockedToken();
if (blockedToken !== null) {
  logits[blockedToken] = -Infinity;
}

// Sample with tsampler
const token = sampleWithStrategy(logits, {
  params: { temperature: 0.8, topP: 0.9 },
  workspace,
  prng,
});
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NGRAM_SIZE` | 6 | N-gram length for sequence tracking |
| `BLOCK_THRESHOLD` | 2 | Block after K occurrences of same pattern |

---

## streaming-semantic-entropy.mjs - Semantic Repetition Detection

Detects **semantic repetition** (same meaning, different words) using a sidecar NLI model and entropy measurement.

### Research Foundation

Based on:
- **Farquhar et al. 2024** - "Detecting Hallucinations in Large Language Models Using Semantic Entropy" ([Nature](https://www.nature.com/articles/s41586-024-07421-0))
- **Quevedo et al. 2024** - "Detecting Hallucinations in Large Language Model Generation: A Token Probability Approach" ([arXiv](https://arxiv.org/abs/2405.19648))

### Key Insight

N-gram blocking catches **token-level** repetition but misses **semantic** repetition:

```
"The guide should include a comprehensive list of references..."
"The guide should include detailed explanations of the formulas..."
"The guide should include code examples for each technique..."
```

These sentences are semantically equivalent (meta-descriptions about the guide) but vary at the token level, evading N-gram detection.

Semantic entropy clusters responses by **meaning** using bidirectional entailment, then measures entropy over clusters. Low entropy = semantic repetition.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Main Context (SmolLM/Llama)                            │
│  - Generation via tsampler                              │
│  - KV forking for K candidate samples                   │
│  - nSeqMax > 1 enables multi-sequence                   │
└─────────────────────────────────────────────────────────┘
                    │ K generated candidates
                    ▼
┌─────────────────────────────────────────────────────────┐
│  Sidecar Context (slim-nli.gguf)                        │
│  - Pairwise entailment: O(K²) checks                    │
│  - Input: "Evidence: A, Conclusion: B"                  │
│  - Output: {"evidence": ["entails"|"neutral"|...]}      │
└─────────────────────────────────────────────────────────┘
                    │ entailment results
                    ▼
┌─────────────────────────────────────────────────────────┐
│  Semantic Clustering                                    │
│  - Bidirectional entailment check (A→B and B→A)         │
│  - Equivalent if: no contradiction, not both neutral    │
│  - Union-find clustering by equivalence                 │
└─────────────────────────────────────────────────────────┘
                    │ semantic_ids
                    ▼
┌─────────────────────────────────────────────────────────┐
│  Entropy Computation                                    │
│  - H = -Σ p(cluster) × log(p(cluster))                  │
│  - High entropy = diverse meanings (good)               │
│  - Low entropy = semantic repetition (bad)              │
└─────────────────────────────────────────────────────────┘
```

### Usage

```bash
node streaming-semantic-entropy.mjs /path/to/model.gguf
```

Requires `models/slim-nli.gguf` for the NLI sidecar.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K_SAMPLES` | 4 | Candidate continuations per check |
| `SAMPLE_TOKENS` | 30 | Tokens per candidate |
| `CHECK_INTERVAL` | 50 | Check entropy every N tokens |
| `ENTROPY_THRESHOLD` | 0.5 | Below this = semantic repetition |

### Multi-Sequence Support

KV cache forking requires `nSeqMax > 1`:

```javascript
const mainCtx = await createContext({
  modelPath,
  nCtx: 2048,
  nSeqMax: K_SAMPLES + 2,  // Enable multi-sequence
});
```

### Semantic Clustering Algorithm

From `get_semantic_ids()` in the Nature paper:

```javascript
async clusterBySemantic(samples) {
  const semanticIds = new Array(n).fill(-1);
  let nextId = 0;

  for (let i = 0; i < n; i++) {
    if (semanticIds[i] === -1) {
      semanticIds[i] = nextId;
      for (let j = i + 1; j < n; j++) {
        if (await this.areSemanticallySimilar(samples[i], samples[j])) {
          semanticIds[j] = nextId;
        }
      }
      nextId++;
    }
  }
  return semanticIds;
}
```

### Entropy Interpretation

| Entropy | K=4 | Meaning |
|---------|-----|---------|
| 1.386 | 4 clusters | Maximum diversity (all unique meanings) |
| 0.693 | 2 clusters | Moderate diversity |
| 0.0 | 1 cluster | All samples semantically equivalent (repetition!) |

---

## Comparison of Approaches

| Aspect | N-gram (tsampler) | Semantic Entropy |
|--------|-------------------|------------------|
| Detects | Exact token sequences | Meaning equivalence |
| Overhead | O(1) per token | O(K²) NLI calls per check |
| False positives | Common code idioms | Rare (meaning-based) |
| False negatives | Paraphrased repetition | Rare |
| Best for | Loop prevention | Quality-critical generation |

### When to Use Each

- **streaming.mjs**: Basic infinite context with BlinkKV
- **streaming-tsampler.mjs**: Long-form generation where exact loops are the concern
- **streaming-semantic-entropy.mjs**: Quality-critical generation where semantic diversity matters

---

## References

1. Han et al. 2024 - "LM-Infinite: Zero-Shot Extreme Length Generalization" (BlinkKV)
2. Farquhar et al. 2024 - "Detecting Hallucinations in Large Language Models Using Semantic Entropy" ([Nature](https://www.nature.com/articles/s41586-024-07421-0))
3. Quevedo et al. 2024 - "Detecting Hallucinations: A Token Probability Approach" ([arXiv](https://arxiv.org/abs/2405.19648))
4. llmware/slim-nli - Specialized NLI model for entailment ([HuggingFace](https://huggingface.co/llmware/slim-nli))
