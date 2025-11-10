import React, { useState, useEffect } from 'react';
import { Box, Text } from 'ink';
import Spinner from 'ink-spinner';
import { SessionContext, createContext } from 'liblloyal-node';

interface ModelLoaderProps {
  modelPath: string;
  contextSize: number;
  threads: number;
  onLoaded: (context: SessionContext) => void;
  onError: (error: string) => void;
}

export const ModelLoader: React.FC<ModelLoaderProps> = ({
  modelPath,
  contextSize,
  threads,
  onLoaded,
  onError
}) => {
  const [logs, setLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let disposed = false;

    // Capture console logs during model loading
    const originalLog = console.log;
    const originalInfo = console.info;
    const capturedLogs: string[] = [];

    const captureLog = (...args: any[]) => {
      const message = args.join(' ');
      capturedLogs.push(message);
      setLogs([...capturedLogs]);
      originalLog(...args);
    };

    console.log = captureLog;
    console.info = captureLog;

    createContext({
      modelPath,
      nCtx: contextSize,
      nThreads: threads
    })
      .then((context) => {
        if (!disposed) {
          setLoading(false);
          // Restore original console methods
          console.log = originalLog;
          console.info = originalInfo;
          onLoaded(context);
        }
      })
      .catch((err) => {
        if (!disposed) {
          setLoading(false);
          // Restore original console methods
          console.log = originalLog;
          console.info = originalInfo;
          onError(err.message);
        }
      });

    return () => {
      disposed = true;
      console.log = originalLog;
      console.info = originalInfo;
    };
  }, [modelPath, contextSize, threads, onLoaded, onError]);

  // Extract model name from path
  const modelName = modelPath.split('/').pop() || 'Unknown';

  return (
    <Box flexDirection="column" paddingX={2}>
      {/* Header */}
      <Box marginBottom={1}>
        {loading ? (
          <Box>
            <Text color="cyan">
              <Spinner type="dots" />
            </Text>
            <Text bold color="cyan">
              {' '}Loading model...
            </Text>
          </Box>
        ) : (
          <Box>
            <Text bold color="green">
              ✓ Model loaded
            </Text>
          </Box>
        )}
      </Box>

      {/* Model info */}
      <Box marginBottom={1}>
        <Text dimColor>
          Model: {modelPath}
        </Text>
      </Box>

      {/* Loading logs */}
      {logs.length > 0 && (
        <Box flexDirection="column" marginTop={1} borderStyle="round" borderColor="gray" paddingX={1}>
          {logs.slice(-15).map((log, i) => (
            <Text key={i} dimColor>
              {log}
            </Text>
          ))}
        </Box>
      )}
    </Box>
  );
};
