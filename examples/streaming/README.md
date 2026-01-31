# Streaming Examples

Advanced streaming patterns for long-form generation with quality preservation.

## Examples Overview

| Example | Purpose | Key Pattern |
|---------|---------|-------------|
| `streaming.mjs` | Infinite context generation | BlinkKV reseeding |
| `streaming-tsampler.mjs` | TypeScript sampling with N-gram tracking | TTA (Test-Time Alignment) |
| `streaming-summary.mjs` | Dynamic summary sinks | BlinkKV + summary sidecar |

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

## streaming-summary.mjs - Dynamic Summary Sinks

Extends BlinkKV with a slim-summary sidecar that generates cumulative summaries of evicted content. Summaries become sink tokens on reseed, giving the model compressed semantic memory of what it generated beyond the visible tail.

### Usage

```bash
node streaming-summary.mjs /path/to/model.gguf
node streaming-summary.mjs /path/to/model.gguf --jsonl
```

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Main Context (llama.cpp)                                │
│  - KV cache management + BlinkKV reseeding               │
│  - Token generation loop                                 │
│  - clearAndReseed(sinks, tail) with dynamic sinks        │
└─────────────────────────────────────────────────────────┘
           │ evicted text                    │ reseed
           ▼                                 ▲ sink tokens
┌─────────────────────────────────┐          │
│  Summary Sidecar (slim-summary)  │──────────┘
│  - slim-summarize.gguf (1.7GB)  │
│  - Prompt: <human>/<summarize>  │
│  - Output: Python-style list    │
└─────────────────────────────────┘

After reseed, KV cache layout:
┌──────────┬─────────────┬───────────────┐
│  anchor  │   summary   │     tail      │
│ (prompt) │ (evicted→)  │ (256 recent)  │
└──────────┴─────────────┴───────────────┘
```

### Sidecar Prompt Format

The slim-summarize model uses a specific prompt format:

```
<human>: {text}
<summarize> key points (5) </summarize>
<bot>:
```

Output is a Python-style list: `['point1', 'point2', 'point3']`

When budget is tight, uses `brief description (1)` for a single cohesive summary.

### Budget Management

| Concept | Formula |
|---------|---------|
| Max sink tokens | `nCtx * sinkBudgetRatio` (default 0.4 = 819 tokens) |
| Summary budget | `maxSinkTokens - anchorTokens.length` |
| Over budget? | Re-summarize with `brief description (1)`, maxTokens=100 |

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TAIL_SIZE` | 256 | Most recent tokens to retain |
| `TARGET_TOKENS` | 5000 | Total tokens to generate |
| `sinkBudgetRatio` | 0.4 | Fraction of context allocated to sinks |
| `summaryMaxTokens` | 200 | Max tokens for summary generation |

### Key APIs

| Method | Description |
|--------|-------------|
| `clearAndReseed(sinks, tail)` | Clear cache, re-decode sinks + tail |
| `tokenize(text)` | Tokenize summary text for sink injection |
| `kvCacheClear()` | Clear sidecar KV before each summary |
| `formatChat(messages)` | Format anchor message with chat template |

---

## References

1. Han et al. 2024 - "LM-Infinite: Zero-Shot Extreme Length Generalization" (BlinkKV)
