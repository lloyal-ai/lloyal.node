import React, { useMemo } from 'react';
import { Box, Text } from 'ink';
import Spinner from 'ink-spinner';
import { marked } from 'marked';
// @ts-ignore - marked-terminal has no types
import { markedTerminal } from 'marked-terminal';

// Configure marked ONCE at module load to avoid repeated registration
let markedConfigured = false;
function ensureMarkedConfigured() {
  if (!markedConfigured) {
    marked.use(markedTerminal());
    markedConfigured = true;
  }
}

interface MessageProps {
  role: 'user' | 'assistant';
  content: string;
  isGenerating?: boolean;
}

export const Message: React.FC<MessageProps> = ({
  role,
  content,
  isGenerating = false,
}) => {
  const isUser = role === 'user';

  // Render markdown for assistant messages (memoized for performance)
  const renderedContent = useMemo(() => {
    if (!isUser && content) {
      ensureMarkedConfigured();
      return (marked.parse(content, { async: false }) as string).trim();
    }
    return content;
  }, [isUser, content]);

  return (
    <Box marginBottom={1}>
      {isGenerating && !content ? (
        <>
          <Text bold color="green">
            ✨
          </Text>
          <Text color="green">
            <Spinner type="dots" />
          </Text>
        </>
      ) : (
        <Box flexDirection="row">
          <Text bold color={isUser ? 'blue' : 'green'}>
            {isUser ? '>  ' : '✨'}
          </Text>
          <Text>{renderedContent}</Text>
        </Box>
      )}
    </Box>
  );
};
