# Grammar-Constrained Generation with Pull Loop

Demonstrates generator-based token streaming with grammar constraints and forkable state.

## Run It

```bash
node grammar.mjs
```

## What You'll See

```
Generating until "city" field...
  {
  "name": "John Doe",
  "age": 30,
  "city":

Saving KV cache and grammar state at branch point...

Exploring 3 city branches:

  [NYC branch]: { "name": "John Doe", "age": 30, "city": "Seattle" }
  [LA branch]: { "name": "John Doe", "age": 30, "city": "Chicago" }
  [Chicago branch]: { "name": "John Doe", "age": 30, "city": "LA" }
```

## The Pull Loop Pattern

This example uses a **pull loop** via JS generators. The consumer requests tokens one at a time and decides when to stop:

```javascript
function* tokenGenerator(ctx, grammarHandle, maxTokens = 100) {
  for (let i = 0; i < maxTokens; i++) {
    const logits = ctx.getLogits();
    ctx.applySampler(grammarHandle, logits);

    const token = ctx.sample({ temperature: 0.7 });
    if (ctx.isStopToken(token)) return;

    ctx.acceptSamplerToken(grammarHandle, token);

    // Yield control back to caller
    yield { token, text: ctx.tokenToText(token) };
  }
}
```

Consumer decides when to continue:

```javascript
for (const { token, text } of gen) {
  accumulated += text;
  await ctx.decode([token], pos++, 0);

  // Stop at decision point - generator pauses here
  if (accumulated.includes('"city"')) {
    break;  // Generator stays paused, state preserved
  }
}
```

## Why Pull Loop Here?

For this branching use case, pull made the code simpler:

```javascript
// Stop when we see the branch point - just break
for (const { token, text } of gen) {
  accumulated += text;
  if (accumulated.includes('"city"')) break;
}
// Generator paused mid-iteration, grammar state intact
// Now save and branch
```

With a push loop you'd need callbacks or flags to signal "stop here" - doable, but the control flow is inverted. Pull keeps the branching logic linear and readable.

## Branching Pattern

1. **Generate** until decision point (pull loop pauses naturally)
2. **Save** both KV cache and grammar state
3. **Fork** for each branch exploration
4. **Restore** and continue independently

```javascript
// Pause at branch point
if (accumulated.includes('"city"')) break;

// Save state
const kvSnapshot = await ctx.kvCacheSave(0);
const grammarSnapshot = ctx.cloneSampler(grammarHandle);

// Explore branches
for (const branch of branches) {
  await ctx.kvCacheLoad(0, kvSnapshot);
  const branchGrammar = ctx.cloneSampler(grammarSnapshot);

  // Each branch continues independently
  for (const { token, text } of tokenGenerator(ctx, branchGrammar)) {
    // ...
  }
}
```

## Key APIs

| Method | Description |
|--------|-------------|
| `getLogits()` | Get logits buffer (modified in-place by applySampler) |
| `applySampler(handle, logits)` | Apply grammar constraints to logits |
| `sample()` | Sample from modified logits |
| `acceptSamplerToken(handle, id)` | Advance grammar parser state |
| `createSampler(grammar)` | Create grammar handle |
| `cloneSampler(handle)` | Clone grammar state for branching |
| `kvCacheSave(seq)` / `kvCacheLoad(seq, buf)` | Snapshot/restore KV state |

## Grammar + KV Travel Together

For valid branching, fork **both**:
- **KV cache**: Model's memory of what it has seen
- **Grammar state**: Parser's position in the grammar

Missing either causes invalid completions or grammar errors.
