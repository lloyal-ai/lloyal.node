import React, { useState, useRef } from 'react';
import { Box, Text, useInput, useFocus, useFocusManager } from 'ink';
import { QuickTextInput } from './QuickTextInput.js';

export interface BootConfig {
  modelPath: string;
  contextSize: number;
  threads: number;
}

interface BootScreenProps {
  initialConfig: BootConfig;
  onStart: (config: BootConfig) => void;
}

export const BootScreen: React.FC<BootScreenProps> = ({ initialConfig, onStart }) => {
  const [modelPath, setModelPath] = useState(initialConfig.modelPath);
  const [contextSize, setContextSize] = useState(initialConfig.contextSize.toString());
  const [threads, setThreads] = useState(initialConfig.threads.toString());

  // Track when we're clearing to prevent onChange from interfering
  const clearingRef = useRef(false);

  // Focus management for inputs
  const { isFocused: modelFocused } = useFocus({ autoFocus: true, id: 'model' });
  const { isFocused: contextFocused } = useFocus({ id: 'context' });
  const { isFocused: threadsFocused } = useFocus({ id: 'threads' });
  const { isFocused: buttonFocused } = useFocus({ id: 'next' });

  const { focusNext, focusPrevious } = useFocusManager();

  // Handle keyboard navigation
  useInput((input, key) => {
    // Cmd/Ctrl+Backspace or Cmd/Ctrl+Delete - clear focused field
    if ((key.ctrl || key.meta) && (key.backspace || key.delete)) {
      clearingRef.current = true;
      if (modelFocused) setModelPath('');
      else if (contextFocused) setContextSize('');
      else if (threadsFocused) setThreads('');
      // Reset flag after React has processed the update
      setTimeout(() => { clearingRef.current = false; }, 0);
      return;
    }

    // Prevent 'u' from Ctrl+U on Unix terminals
    if (key.ctrl && input === 'u') {
      clearingRef.current = true;
      if (modelFocused) setModelPath('');
      else if (contextFocused) setContextSize('');
      else if (threadsFocused) setThreads('');
      setTimeout(() => { clearingRef.current = false; }, 0);
      return;
    }

    // Enter always starts
    if (key.return) {
      handleStart();
      return;
    }

    // Down arrow - navigate to next field
    if (key.downArrow) {
      focusNext();
      return;
    }

    // Up arrow - navigate to previous field
    if (key.upArrow) {
      focusPrevious();
      return;
    }
  });

  const handleStart = () => {
    const config: BootConfig = {
      modelPath: modelPath.trim() || initialConfig.modelPath,
      contextSize: parseInt(contextSize, 10) || initialConfig.contextSize,
      threads: parseInt(threads, 10) || initialConfig.threads
    };
    onStart(config);
  };

  return (
    <Box flexDirection="column" padding={2}>
      {/* Header */}
      <Box marginBottom={2} borderStyle="round" borderColor="cyan" paddingX={2}>
        <Text bold color="cyan">
          Configuration
        </Text>
      </Box>

      {/* Model Path */}
      <Box flexDirection="column" marginBottom={1}>
        <Text bold color={modelFocused ? 'green' : 'gray'}>
          Model (path or URL):
        </Text>
        <Box marginLeft={2} marginTop={0}>
          <QuickTextInput
            value={modelPath}
            onChange={(val) => !clearingRef.current && setModelPath(val)}
            placeholder="Enter model path or URL..."
            focus={modelFocused}
          />
        </Box>
      </Box>

      {/* Context Size */}
      <Box flexDirection="column" marginBottom={1}>
        <Text bold color={contextFocused ? 'green' : 'gray'}>
          Context Size: <Text dimColor>(default 8192)</Text>
        </Text>
        <Box marginLeft={2} marginTop={0}>
          <QuickTextInput
            value={contextSize}
            onChange={(val) => !clearingRef.current && setContextSize(val)}
            placeholder="8192"
            focus={contextFocused}
          />
        </Box>
      </Box>

      {/* Threads */}
      <Box flexDirection="column" marginBottom={2}>
        <Text bold color={threadsFocused ? 'green' : 'gray'}>
          Threads: <Text dimColor>(default 1)</Text>
        </Text>
        <Box marginLeft={2} marginTop={0}>
          <QuickTextInput
            value={threads}
            onChange={(val) => !clearingRef.current && setThreads(val)}
            placeholder="1"
            focus={threadsFocused}
          />
        </Box>
      </Box>

      {/* Next Button */}
      <Box justifyContent="center" marginTop={1}>
        {buttonFocused ? (
          <Text backgroundColor="green" color="black" bold>
            {' '}Start{' '}
          </Text>
        ) : (
          <Text dimColor>
            [Start]
          </Text>
        )}
      </Box>

      {/* Instructions */}
      <Box marginTop={2} justifyContent="center">
        <Text dimColor>
          Tab/↑/↓ to navigate • Enter to start
        </Text>
      </Box>
    </Box>
  );
};
