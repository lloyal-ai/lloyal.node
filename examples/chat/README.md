# Simple Chat Example

A minimal chat example demonstrating the lloyal.node API.

## Usage

```bash
npm run example                           # uses default model
npm run example -- /path/to/model.gguf    # custom model
```

## Commands

- `/clear` - Reset conversation and clear terminal
- `/quit` - Exit

## The Pattern: Sync Produce, Async Commit

```javascript
// Sync generator - all operations are synchronous
function* produceTokens(ctx, params) {
  while (true) {
    const tokenId = ctx.sample(params);      // sync
    if (ctx.isStopToken(tokenId)) return;    // sync
    const text = ctx.tokenToText(tokenId);   // sync
    yield { text, tokenId };
  }
}

// Usage - async commit is explicit in caller's loop
for (const { text, tokenId } of produceTokens(ctx, params)) {
  process.stdout.write(text);
  await ctx.decode([tokenId], position);     // async commit to KV
  position += 1;
}
```

**Key insight:** Token production is synchronous. Only the KV cache commit (`decode`) is async. This separation makes the control flow explicit.

## API Reference

| Method | Sync/Async | Purpose |
|--------|------------|---------|
| `sample(params)` | sync | Sample next token from logits |
| `isStopToken(id)` | sync | Check if token ends generation |
| `tokenToText(id)` | sync | Convert token ID to text |
| `decode(tokens, pos)` | async | Commit tokens to KV cache |
| `tokenize(text)` | async | Convert text to token IDs |
| `formatChat(json)` | async | Apply chat template |
| `kvCacheClear()` | async | Reset KV cache |
