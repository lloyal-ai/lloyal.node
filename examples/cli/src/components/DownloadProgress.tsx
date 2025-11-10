import React from 'react';
import { Box, Text } from 'ink';

interface DownloadProgressProps {
  filename: string;
  downloadedBytes: number;
  totalBytes: number;
  speed?: number; // bytes per second
}

export const DownloadProgress: React.FC<DownloadProgressProps> = ({
  filename,
  downloadedBytes,
  totalBytes,
  speed
}) => {
  const percentComplete = totalBytes > 0 ? (downloadedBytes / totalBytes) * 100 : 0;

  // Format bytes to human readable
  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };

  // Format speed
  const formatSpeed = (bytesPerSec: number): string => {
    return `${formatBytes(bytesPerSec)}/s`;
  };

  // Determine bar color based on progress
  const getBarColor = (): string => {
    if (percentComplete === 100) return 'green';
    if (percentComplete > 50) return 'cyan';
    if (percentComplete > 25) return 'blue';
    return 'gray';
  };

  // Build progress bar using simple characters
  const barWidth = 50;
  const filledWidth = Math.round((percentComplete / 100) * barWidth);
  const emptyWidth = barWidth - filledWidth;
  const progressBar = '█'.repeat(filledWidth) + '░'.repeat(emptyWidth);

  return (
    <Box flexDirection="column" paddingX={2}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Downloading Model
        </Text>
      </Box>

      <Box marginBottom={1}>
        <Text dimColor>
          {filename}
        </Text>
      </Box>

      {/* Progress Bar */}
      <Box marginBottom={1}>
        <Text color={getBarColor()}>
          {progressBar}
        </Text>
      </Box>

      {/* Progress Stats */}
      <Box justifyContent="space-between">
        <Text color={getBarColor()}>
          {percentComplete.toFixed(1)}%
        </Text>
        <Text dimColor>
          {formatBytes(downloadedBytes)} / {formatBytes(totalBytes)}
          {speed && speed > 0 && ` • ${formatSpeed(speed)}`}
        </Text>
      </Box>
    </Box>
  );
};
