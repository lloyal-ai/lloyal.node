import React, { useState, useEffect } from 'react';
import { Box, Text } from 'ink';
import { SessionContext } from 'liblloyal-node';
import { BootScreen, type BootConfig } from './BootScreen.js';
import { DownloadProgress } from './DownloadProgress.js';
import { ModelLoader } from './ModelLoader.js';
import { Chat } from './Chat.js';
import {
  resolveModelPath,
  isUrl,
  getFilenameFromUrl,
  type DownloadProgress as DownloadProgressType
} from '../utils/modelDownloader.js';

interface AppProps {
  modelPath: string;
  contextSize?: number;
  threads?: number;
}

type AppState =
  | { stage: 'boot' }
  | { stage: 'downloading'; url: string; progress: DownloadProgressType; config: BootConfig }
  | { stage: 'loading'; localPath: string; config: BootConfig }
  | { stage: 'ready'; context: SessionContext; config: BootConfig }
  | { stage: 'error'; error: string };

export const App: React.FC<AppProps> = ({
  modelPath,
  contextSize = 8192,
  threads = 1
}) => {
  const [appState, setAppState] = useState<AppState>({ stage: 'boot' });
  const [config, setConfig] = useState<BootConfig | null>(null);

  // Handle boot screen start
  const handleStart = (bootConfig: BootConfig) => {
    setConfig(bootConfig);
  };

  // Handle model resolution (download if URL, or use local path) after boot
  useEffect(() => {
    if (!config) return; // Wait for boot screen to complete

    let cancelled = false;

    const loadModel = async () => {
      try {
        if (isUrl(config.modelPath)) {
          // Download model with progress
          const localPath = await resolveModelPath(
            config.modelPath,
            (progress) => {
              if (!cancelled) {
                setAppState({
                  stage: 'downloading',
                  url: config.modelPath,
                  progress,
                  config
                });
              }
            }
          );

          if (!cancelled) {
            setAppState({ stage: 'loading', localPath, config });
          }
        } else {
          // Local path - go straight to loading
          const localPath = await resolveModelPath(config.modelPath);
          if (!cancelled) {
            setAppState({ stage: 'loading', localPath, config });
          }
        }
      } catch (error) {
        if (!cancelled) {
          setAppState({
            stage: 'error',
            error: error instanceof Error ? error.message : String(error)
          });
        }
      }
    };

    loadModel();

    return () => {
      cancelled = true;
    };
  }, [config]);

  // Render based on current stage
  if (appState.stage === 'boot') {
    return (
      <BootScreen
        initialConfig={{
          modelPath,
          contextSize,
          threads
        }}
        onStart={handleStart}
      />
    );
  }

  if (appState.stage === 'downloading') {
    const filename = getFilenameFromUrl(appState.url);
    return (
      <DownloadProgress
        filename={filename}
        downloadedBytes={appState.progress.downloadedBytes}
        totalBytes={appState.progress.totalBytes}
        speed={appState.progress.speed}
      />
    );
  }

  if (appState.stage === 'loading') {
    return (
      <ModelLoader
        modelPath={appState.localPath}
        contextSize={appState.config.contextSize}
        threads={appState.config.threads}
        onLoaded={(context) => {
          setAppState({ stage: 'ready', context, config: appState.config });
        }}
        onError={(error) => {
          setAppState({ stage: 'error', error });
        }}
      />
    );
  }

  if (appState.stage === 'ready') {
    // Extract model name from path
    const modelNameExtracted = appState.config.modelPath.split('/').pop()?.replace('.gguf', '') || 'Unknown';
    return (
      <Chat
        context={appState.context}
        contextSize={appState.config.contextSize}
        modelName={modelNameExtracted}
      />
    );
  }

  if (appState.stage === 'error') {
    return (
      <Box flexDirection="column" padding={2}>
        <Text bold color="red">
          Error
        </Text>
        <Text>{appState.error}</Text>
      </Box>
    );
  }

  return null;
};
