# Speculative Decoding with Branch API

Demonstrates speculative decoding using the Branch primitive: fork a draft, verify, accept/reject, sample bonus token.

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
| **1. MAIN** | Create main branch tracking committed state |
| **2. FORK** | Fork draft branch (shares KV prefix with main) |
| **3. DRAFT** | produce/commit N tokens on draft branch |
| **4. VERIFY** | Check draft confidence (entropy threshold) |
| **5. PRUNE** | Remove draft branch (cleans up divergent KV) |
| **6. ACCEPT** | Commit accepted tokens to main branch |
| **7. BONUS** | Sample one token from main at rejection point |

## Key Pattern: Fork/Draft/Verify with Branch API

```javascript
// Main branch tracks committed state
const main = Branch.create(ctx, 0, { temperature: 0.7 });
await main.prefill(promptTokens);

while (output.length < maxTokens) {
  // Fork draft from main — shares KV prefix
  const draft = await main.fork();
  draft.reseedSampler(iteration);

  // Draft N tokens
  const drafts = [];
  for (let i = 0; i < N; i++) {
    const entropy = ctx.modelEntropy('nats', draft.getLogits());
    const { token, text, isStop } = draft.produceSync();
    if (isStop) break;
    drafts.push({ token, text, entropy });
    await draft.commit(token);
  }

  // Verify and prune draft
  const acceptedCount = verify(drafts);
  await draft.prune();

  // Commit accepted tokens to main
  for (const d of drafts.slice(0, acceptedCount)) {
    await main.commit(d.token);
  }

  // Bonus token from main at rejection point
  if (acceptedCount < drafts.length) {
    const { token } = main.produceSync();
    await main.commit(token);
  }
}
await main.prune();
```

## Why Branch API?

The produce/commit separation is what makes speculative decoding natural:

- **produce()** samples without writing to KV — inspect before deciding
- **commit()** accepts + decodes — advance state only for accepted tokens
- **fork()** shares KV prefix — draft branch doesn't duplicate the prompt
- **prune()** removes divergent KV — clean rejection without manual bookkeeping

## Key APIs

| Method | Description |
|--------|-------------|
| `Branch.create(ctx, pos, params)` | Create branch at position |
| `branch.fork()` | Fork: shared KV prefix + cloned sampler |
| `branch.produce()` | Sample without KV write |
| `branch.commit(token)` | Accept + decode into KV |
| `branch.prune()` | Remove divergent KV entries |
| `branch.reseedSampler(seed)` | Diversify forked branch |
| `ctx.modelEntropy('nats', logits)` | Check draft confidence |

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
