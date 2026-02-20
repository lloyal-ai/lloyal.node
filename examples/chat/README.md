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

## The Pattern: Branch Produce/Commit

```javascript
// Create branch and prefill prompt
const branch = Branch.create(ctx, 0, { temperature: 0.7 });
await branch.prefill(promptTokens);

// Async iterator - commit-before-yield
for await (const { token, text } of branch) {
  process.stdout.write(text);
}
await branch.prune();
```

**Key insight:** The async iterator handles produce/commit internally. Each yielded token is already committed to KV. Breaking out is clean — no orphaned state.

## API Reference

| Method | Sync/Async | Purpose |
|--------|------------|---------|
| `Branch.create(ctx, pos, params)` | sync | Create a branch for generation |
| `branch.prefill(tokens)` | async | Feed tokens into branch's KV cache |
| `branch.produce()` | async | Sample next token (no KV write) |
| `branch.commit(token)` | async | Accept + decode into KV |
| `branch.prune()` | async | Discard branch and its KV entries |
| `ctx.isStopToken(id)` | sync | Check if token ends generation |
| `ctx.tokenToText(id)` | sync | Convert token ID to text |
| `ctx.tokenize(text)` | async | Convert text to token IDs |
| `ctx.formatChat(json)` | async | Apply chat template |
| `ctx.kvCacheClear()` | async | Reset KV cache |
