# Best-of-N Sampling with Perplexity Selection

Demonstrates why best-of-n beats single generation: generate N diverse candidates, select the most coherent by perplexity.

## Run It

```bash
node best-of-n.mjs
```

## What You'll See

```
BASELINE: Single generation (T=0.3)
  PPL: 2.07 | "In the realm where the moon dipped..."

BEST-OF-5: Generate 5 candidates (T=0.9), select lowest PPL
  [1] PPL:  2.95 | "In the heart of a moonlit forest..."
  [2] PPL:  4.41 | "Under the cloak of a midnight moon..."
  [3] PPL:  3.09 | "As the last wisps of sunlight..."
  [4] PPL:  3.42 | "Under the moon's silvery glow..."
  [5] PPL:  3.46 | "Under the emerald canopy..."

RESULTS
  Best candidate [1] (PPL 2.95)
  PPL range: 2.95 - 4.41 (Δ1.46)
```

## How It Works

| Step | What Happens |
|------|--------------|
| 1. Prefill | Decode prompt on seq 0 |
| 2. Capture logits | Copy logits buffer (critical for fair comparison) |
| 3. Generate N candidates | Each forks KV, samples from captured logits, then continues |
| 4. Track PPL | Accumulate surprisal per candidate |
| 5. Select best | Lowest perplexity wins |

## Key Implementation Detail

After prefilling, the logits buffer contains P(next_token | prompt). When we fork to multiple sequences, **each candidate's first token must sample from these same captured logits**:

```javascript
// Capture after prefill
const capturedLogits = new Float32Array(ctx.getLogits());

// Each candidate:
// 1. Sample first token from captured logits (tsampler)
const token = sampleWithStrategy(capturedLogits, { params, workspace, prng });

// 2. Compute surprisal from captured logits (native C++)
const surprisal = ctx.modelSurprisal(token, 'nats', capturedLogits);
```

Without this, later candidates would sample from earlier candidates' states - unfair comparison.

## Why Perplexity?

```
PPL = exp(average surprisal) = "how surprised is the model?"
```

| PPL | Meaning |
|-----|---------|
| Low | Model is confident in what it wrote |
| High | Model was uncertain, may have inconsistencies |

Best-of-N trades compute for quality:
- High temp generates **diverse** candidates (explore)
- PPL filtering selects **coherent** ones (exploit)

## Key APIs

| Method | Description |
|--------|-------------|
| `kvSeqCopy(src, dst)` | Fork KV cache (O(1) tag copy) |
| `getLogits()` | Get raw logits (zero-copy view) |
| `modelSurprisal(token, base?, logits?)` | Surprisal from current or captured logits |
| `createPerplexityTracker()` | Create tracker handle |
| `addSurprisal(tracker, value)` | Accumulate to tracker |
| `getPerplexity(tracker)` | Get current PPL |

## Native Metrics API

The native `modelSurprisal()` accepts an optional `logits` parameter for captured logits:

```javascript
// First token: surprisal from captured logits
const firstSurprisal = ctx.modelSurprisal(token, 'nats', capturedLogits);

// Subsequent tokens: current context logits (default)
const surprisal = ctx.modelSurprisal(token);
```

All math runs in C++ - no JS overhead for softmax/log operations.

## tsampler Integration

[@lloyal-labs/tsampler](https://www.npmjs.com/package/@lloyal-labs/tsampler) handles sampling from captured logits:

```javascript
import { sampleWithStrategy, SamplerWorkspace, Xoroshiro128Plus } from '@lloyal-labs/tsampler';

const token = sampleWithStrategy(capturedLogits, {
  params: { temperature: 0.9, topP: 0.95 },
  workspace,
  prng,
});
```

**Division of labor:**
- **tsampler**: Sampling (temperature, topP, topK) from arbitrary logits
- **Native API**: Metrics (surprisal, entropy, perplexity) from arbitrary logits

## References

- [Stiennon et al. 2020](https://arxiv.org/abs/2009.01325) - "Learning to summarize from human feedback" (Best-of-N in RLHF)
- [tsampler](https://github.com/lloyal-ai/tsampler) - Pure TypeScript sampling with llama.cpp parity
