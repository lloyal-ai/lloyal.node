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
  // Parse markdown on newlines during streaming for better real-time display
  const renderedContent = useMemo(() => {
    if (!isUser && content) {
      ensureMarkedConfigured();

      // During streaming, only parse content up to the last newline
      // Keep unparsed trailing content as-is to avoid breaking incomplete markdown
      if (isGenerating) {
        const lastNewlineIndex = content.lastIndexOf('\n');
        if (lastNewlineIndex === -1) {
          // No newlines yet, show raw content
          return content;
        }

        // Parse complete lines, keep trailing content raw
        const completeContent = content.substring(0, lastNewlineIndex + 1);
        const trailingContent = content.substring(lastNewlineIndex + 1);
        const parsed = (marked.parse(completeContent, { async: false }) as string);
        return parsed + trailingContent;
      }

      // Not generating, parse everything
      return (marked.parse(content, { async: false }) as string).trim();
    }
    return content;
  }, [isUser, content, isGenerating]);

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
