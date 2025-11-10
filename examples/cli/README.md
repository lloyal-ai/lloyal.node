# liblloyal-node CLI Chat Example

An interactive chat interface built with [Ink](https://github.com/vadimdemedes/ink) and TypeScript, demonstrating real-world usage of liblloyal-node with chat templates.

## Features

- 🎨 Beautiful terminal UI with Ink
- 💬 Multi-turn conversations with chat history
- 🔄 Proper chat template formatting
- ⚡ Streaming-style token generation
- 🎯 TypeScript for type safety

## Quick Start

```bash
# Install dependencies
npm install

# Run in development mode (with tsx)
npm run dev

# Or build and run
npm run build
npm start
```

## Usage

```bash
# Use default model (../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf)
npm run dev

# Or specify a custom model path
npm run dev /path/to/your/model.gguf
```

## How It Works

The example demonstrates:

1. **Chat Template Formatting**: Uses `ctx.formatChat()` to properly format messages with the model's built-in chat template
2. **Token Generation**: Generates tokens with configurable sampling parameters (temperature, top-k, top-p)
3. **Stop Token Detection**: Automatically stops generation when encountering EOS/EOT tokens
4. **Multi-turn Context**: Maintains conversation history across multiple exchanges

## Project Structure

```
examples/cli/
├── src/
│   ├── index.tsx              # Entry point & CLI arg parsing
│   └── components/
│       ├── Chat.tsx           # Main chat component with LLM logic
│       └── Message.tsx        # Message display component
├── package.json
├── tsconfig.json
└── README.md
```

## Code Highlights

### Chat Template Usage

```typescript
// Format messages using chat template
const messagesJson = JSON.stringify([
  { role: 'user', content: 'What is the capital of France?' }
]);

const { prompt } = await ctx.formatChat(messagesJson);
const tokens = await ctx.tokenize(prompt);
```

### Sampling Configuration

```typescript
const token = ctx.sample({
  temperature: 0.7,  // Moderate creativity
  topK: 40,          // Consider top 40 tokens
  topP: 0.9,         // Nucleus sampling threshold
  minP: 0.05         // Minimum probability threshold
});
```

## Keyboard Shortcuts

- **Enter**: Send message
- **Ctrl+C**: Exit

## License

MIT
