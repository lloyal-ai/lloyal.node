import React, { useState, useEffect, useRef } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import TextInput from 'ink-text-input';
import { SessionContext, createContext, FormattedChatResult } from 'liblloyal-node';
import { Message } from './Message.js';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatProps {
  modelPath: string;
}

export const Chat: React.FC<ChatProps> = ({ modelPath }) => {
  const { exit } = useApp();
  const [ctx, setCtx] = useState<SessionContext | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [generating, setGenerating] = useState(false);
  const [paused, setPaused] = useState(false); // Default to playing (auto-advance)
  const [kvCursor, setKvCursor] = useState(0);
  const [lastFormattedPrompt, setLastFormattedPrompt] = useState('');
  const [generator, setGenerator] = useState<AsyncGenerator<string, void, void> | null>(null);

  const pausedRef = useRef(paused);

  // Keep ref in sync with state for auto-advance loop
  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  // Initialize model
  useEffect(() => {
    let disposed = false;

    createContext({
      modelPath,
      nCtx: 2048,
      nThreads: 4
    })
      .then((context) => {
        if (!disposed) {
          setCtx(context);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!disposed) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => {
      disposed = true;
    };
  }, [modelPath]);

  // Handle keyboard shortcuts
  useInput((input, key) => {
    if (key.ctrl && input === 'c') {
      if (ctx) {
        ctx.dispose();
      }
      exit();
    }

    // Space to pause/play
    if (input === ' ' && generating) {
      setPaused(!paused);
    }

    // 'c' to clear chat
    if (input === 'c' && !generating) {
      handleClearChat();
    }
  });

  // Auto-advance effect - runs when not paused (Nitro pattern)
  useEffect(() => {
    if (!generating || paused || !generator) return;

    let cancelled = false;

    const autoAdvance = async () => {
      while (!cancelled && !pausedRef.current && generator) {
        try {
          const result = await generator.next();

          if (result.done) {
            setGenerating(false);
            setGenerator(null);
            setPaused(false);
            break;
          }

          // Small delay for UI to update (simpler than requestAnimationFrame in CLI)
          await new Promise(resolve => setTimeout(resolve, 16));
        } catch (error) {
          console.error('[Chat] Error in auto-advance:', error);
          setGenerating(false);
          setGenerator(null);
          setPaused(false);
          setMessages(prev => {
            const updated = [...prev];
            if (updated.length > 0) {
              updated[updated.length - 1].content = `Error: ${error instanceof Error ? error.message : String(error)}`;
            }
            return updated;
          });
          break;
        }
      }
    };

    autoAdvance();

    return () => {
      cancelled = true;
    };
  }, [generating, paused, generator]);

  // Token generator (Nitro pattern from SessionContext.tsx with prompt diffing)
  async function* tokenGenerator(userMessage: string): AsyncGenerator<string, void, void> {
    if (!ctx) return;

    try {
      // Add user message
      const newMessages = [...messages, { role: 'user' as const, content: userMessage }];
      setMessages(newMessages);

      // Add empty assistant message (will be filled as tokens stream)
      setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

      // Format FULL conversation using chat template
      const messagesJson = JSON.stringify(newMessages);
      const { prompt: fullPrompt } = await ctx.formatChat(messagesJson);

      // Detect NEW tokens by comparing to previous prompt (prompt diffing!)
      const previousPrompt = lastFormattedPrompt;
      const newPromptPortion = fullPrompt.startsWith(previousPrompt)
        ? fullPrompt.substring(previousPrompt.length)
        : fullPrompt; // First turn or cache was cleared

      // Tokenize only the NEW portion (efficiency!)
      const newTokens = await ctx.tokenize(newPromptPortion);

      // Decode only new tokens at current KV cursor
      await ctx.decode(newTokens, kvCursor);

      // Update position after decoding prompt
      let currentPosition = kvCursor + newTokens.length;
      let responseText = '';

      // Update state with new formatted prompt
      setLastFormattedPrompt(fullPrompt);

      // Generate tokens until model stop token
      while (true) {
        // Sample next token
        const tokenId = ctx.sample({
          temperature: 0.7,
          topK: 40,
          topP: 0.9,
          minP: 0.05
        });

        // Check if end of generation
        if (ctx.isStopToken(tokenId)) {
          break;
        }

        // Convert token to text
        const text = ctx.tokenToText(tokenId);
        responseText += text;

        // Update the assistant message in real-time
        setMessages(prev => {
          const updated = [...prev];
          if (updated.length > 0) {
            updated[updated.length - 1].content = responseText;
          }
          return updated;
        });

        // Decode the new token
        await ctx.decode([tokenId], currentPosition);
        currentPosition += 1;

        // YIELD control back - wait for next() call from auto-advance
        yield text;
      }

      // Save final KV cursor for next turn
      setKvCursor(currentPosition);

      // Final trim of the response
      setMessages(prev => {
        const updated = [...prev];
        if (updated.length > 0) {
          updated[updated.length - 1].content = responseText.trim();
        }
        return updated;
      });
    } catch (error) {
      console.error('Generation error:', error);
      setMessages(prev => {
        const updated = [...prev];
        if (updated.length > 0) {
          updated[updated.length - 1].content = `Error: ${error instanceof Error ? error.message : String(error)}`;
        }
        return updated;
      });
    }
  }

  // Handle input submission
  const handleSubmit = async (value: string) => {
    if (!ctx || !value.trim() || generating) return;

    const trimmed = value.trim();
    setInput('');
    setGenerating(true);
    setPaused(false); // Default to playing

    try {
      // Create generator and start it
      const gen = tokenGenerator(trimmed);
      setGenerator(gen);
      await gen.next(); // Run setup, pause at first yield
    } catch (error) {
      console.error('[Chat] Error starting generator:', error);
      setGenerating(false);
      setGenerator(null);
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${error}` }]);
    }
  };

  const handleClearChat = async () => {
    if (!ctx || generating) return;

    try {
      await ctx.kvCacheClear();
      setMessages([]);
      setKvCursor(0);
      setLastFormattedPrompt('');
    } catch (error) {
      console.error('[Chat] Error clearing:', error);
    }
  };

  if (loading) {
    return (
      <Box flexDirection="column" padding={1}>
        <Text color="cyan">Loading model...</Text>
        <Text dimColor>Model: {modelPath}</Text>
      </Box>
    );
  }

  if (error) {
    return (
      <Box flexDirection="column" padding={1}>
        <Text color="red">Error loading model:</Text>
        <Text>{error}</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" padding={1}>
      {/* Header */}
      <Box marginBottom={1} borderStyle="round" borderColor="cyan" paddingX={2}>
        <Text bold color="cyan">
          liblloyal-node Chat
        </Text>
        <Text dimColor> • Ctrl+C to exit • Space to pause/play • 'c' to clear</Text>
      </Box>

      {/* Status */}
      {generating && (
        <Box marginBottom={1}>
          <Text color={paused ? "yellow" : "green"}>
            {paused ? "⏸  Paused (press Space to resume)" : "▶  Generating..."}
          </Text>
        </Box>
      )}

      {/* Messages */}
      <Box flexDirection="column" marginBottom={1}>
        {messages.length === 0 && (
          <Box marginBottom={1}>
            <Text dimColor>Start chatting! Type a message and press Enter.</Text>
          </Box>
        )}
        {messages.map((msg, i) => (
          <Message key={i} role={msg.role} content={msg.content} />
        ))}
      </Box>

      {/* Input */}
      <Box>
        <Text color="green" bold>
          {'> '}
        </Text>
        <TextInput
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          placeholder="Type your message..."
          focus={!generating}
        />
      </Box>
    </Box>
  );
};
