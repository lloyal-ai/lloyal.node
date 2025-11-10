import React from 'react';
import { Box, Text } from 'ink';

interface MessageProps {
  role: 'user' | 'assistant';
  content: string;
}

export const Message: React.FC<MessageProps> = ({ role, content }) => {
  const isUser = role === 'user';

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Box>
        <Text bold color={isUser ? 'blue' : 'green'}>
          {isUser ? 'You' : 'Assistant'}:
        </Text>
      </Box>
      <Box paddingLeft={2}>
        <Text>{content}</Text>
      </Box>
    </Box>
  );
};
