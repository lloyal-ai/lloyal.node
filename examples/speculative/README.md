# Speculative Decoding with Forkable KV State

Demonstrates the KV cache primitives needed for speculative decoding: draft, fork, verify, accept/reject.

## Run It

```bash
node speculative.mjs
```

## What You'll See

```
Prompt: "The quick brown fox"

Generating 30 tokens with speculative decoding...

The quick brown fox jumps over the lazy dog. The dog...

==================================================
Statistics
==================================================
  Iterations: 13
  Tokens drafted: 48
  Tokens accepted: 6
  Accept rate: 12.5%
  Output tokens: 30
```

## How It Works

| Phase | What Happens |
|-------|--------------|
| **1. DRAFT** | Generate N tokens greedily (fast, low quality ok) |
| **2. FORK** | `kvSeqCopy(0, 1)` - copy KV state for verification |
| **3. VERIFY** | Run target model on all N tokens (one batch) |
| **4. ACCEPT** | Keep tokens where target agrees with draft |
| **5. BONUS** | Sample one token from target at rejection point |
| **6. CLEANUP** | `kvCacheRemove()` rejected tokens, repeat |

## Key Pattern: Accept/Reject with KV Cleanup

```javascript
// Draft N tokens on seq 0
for (let i = 0; i < N; i++) {
  const token = ctx.sample({ temperature: 0.0 });
  await ctx.decode([token], pos++, 0);
  drafts.push(token);
}

// Fork for verification
ctx.kvSeqCopy(0, 1);

// Verify (compare draft vs target distributions)
const acceptedCount = verify(drafts);

// Remove rejected tokens from KV cache
if (acceptedCount < drafts.length) {
  const rejectPos = startPos + acceptedCount;
  await ctx.kvCacheRemove(0, rejectPos, -1);  // Critical!

  // Sample bonus token from target at rejection point
  const bonus = ctx.sample({ temperature: 0.7 });
  await ctx.decode([bonus], rejectPos, 0);
}
```

## Why Fork Before Verify?

In real speculative decoding with two models:
- Draft model: small, fast, generates candidates
- Target model: large, slow, verifies quality

The fork lets you run the target model on seq 1 while keeping the draft state on seq 0. After verification, you collapse to the accepted prefix.

## Key APIs

| Method | Description |
|--------|-------------|
| `kvSeqCopy(src, dst)` | Fork KV cache (O(1) tag copy) |
| `kvCacheRemove(seq, start, end)` | Remove token range from cache |
| `modelEntropy('nats')` | Check draft confidence |
| `nSeqMax` | Context option for multi-sequence |

## Accept Rate

The accept rate determines speedup:

| Accept Rate | Meaning |
|-------------|---------|
| High (>70%) | Draft model matches target well - good speedup |
| Low (<30%) | Draft model diverges - minimal speedup |

This example uses entropy-based simulation (not a real draft model), so accept rates are low. With a properly trained draft model, rates of 60-80% are achievable.

## References

- [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192) - "Fast Inference from Transformers via Speculative Decoding"
- [Chen et al. 2023](https://arxiv.org/abs/2302.01318) - "Accelerating LLM Decoding with Speculative Sampling"
