# Grammar-Constrained Generation with Branch Forking

Demonstrates grammar-constrained generation using the Branch API with automatic grammar cloning on fork.

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

Forking into 3 branches at branch point...

  [NYC branch]: { "name": "John Doe", "age": 30, "city": "Seattle" }
  [LA branch]: { "name": "John Doe", "age": 30, "city": "Chicago" }
  [Chicago branch]: { "name": "John Doe", "age": 30, "city": "LA" }
```

## The Branch Fork Pattern

Grammar state is integrated into the branch and cloned automatically on fork:

```javascript
// Create root branch with grammar constraint
const grammar = await ctx.jsonSchemaToGrammar(JSON.stringify(schema));
const root = Branch.create(ctx, 0, params, undefined, grammar);
await root.prefill(promptTokens);

// Generate until branch point
for (let i = 0; i < 100; i++) {
  const { token, text, isStop } = await root.produce();
  if (isStop) break;
  await root.commit(token);
  if (accumulated.includes('"city"')) break;
}

// Fork — grammar state cloned automatically
for (const city of cities) {
  const child = await root.fork();
  child.reseedSampler(seed++);

  for await (const { text } of child) {
    // Each branch generates independently with its own grammar state
  }
  await child.prune();
}
await root.prune();
```

## Why Branch Fork Here?

For grammar-constrained branching, fork handles everything atomically:
- **KV cache**: Shared prefix, divergent-only storage per branch
- **Grammar state**: Parser position cloned automatically
- **Sampler chain**: Penalties and PRNG cloned and reseeded

No manual KV save/load or grammar cloning needed — `fork()` is a single operation.

## Key APIs

| Method | Description |
|--------|-------------|
| `Branch.create(ctx, pos, params, nBatch, grammar)` | Create branch with grammar constraint |
| `branch.fork()` | Clone branch: KV prefix + grammar + sampler |
| `branch.reseedSampler(seed)` | Diversify forked branch's PRNG |
| `branch.produce()` | Sample grammar-valid token |
| `branch.commit(token)` | Advance grammar + KV state |
| `branch.prune()` | Clean up branch resources |
| `ctx.jsonSchemaToGrammar(json)` | Convert JSON schema to GBNF grammar |
