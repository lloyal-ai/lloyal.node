import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as https from 'https';
import * as http from 'http';
import { URL } from 'url';

export interface DownloadProgress {
  downloadedBytes: number;
  totalBytes: number;
  speed: number; // bytes per second
}

export const MODELS_DIR = path.join(os.homedir(), '.clai', 'models');

/**
 * Ensures the models directory exists
 */
export function ensureModelsDir(): void {
  if (!fs.existsSync(MODELS_DIR)) {
    fs.mkdirSync(MODELS_DIR, { recursive: true });
  }
}

/**
 * Extracts filename from URL or generates one
 */
export function getFilenameFromUrl(url: string): string {
  const urlObj = new URL(url);
  const pathname = urlObj.pathname;
  const filename = path.basename(pathname);

  // If no filename or doesn't look like a model file, generate one
  if (!filename || !filename.includes('.')) {
    return `model_${Date.now()}.gguf`;
  }

  return filename;
}

/**
 * Gets the local path for a model file
 */
export function getLocalModelPath(filename: string): string {
  return path.join(MODELS_DIR, filename);
}

/**
 * Checks if a model is already downloaded
 */
export function isModelCached(filename: string): boolean {
  const localPath = getLocalModelPath(filename);
  return fs.existsSync(localPath);
}

/**
 * Downloads a model from a URL with progress tracking
 */
export async function downloadModel(
  url: string,
  onProgress: (progress: DownloadProgress) => void
): Promise<string> {
  ensureModelsDir();

  const filename = getFilenameFromUrl(url);
  const localPath = getLocalModelPath(filename);

  // Check if already downloaded
  if (isModelCached(filename)) {
    // Get file size and report as complete
    const stats = fs.statSync(localPath);
    onProgress({
      downloadedBytes: stats.size,
      totalBytes: stats.size,
      speed: 0
    });
    return localPath;
  }

  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https') ? https : http;

    const request = protocol.get(url, (response) => {
      // Handle redirects
      if (response.statusCode === 301 || response.statusCode === 302) {
        const redirectUrl = response.headers.location;
        if (!redirectUrl) {
          reject(new Error('Redirect without location header'));
          return;
        }
        // Recursively follow redirect
        downloadModel(redirectUrl, onProgress)
          .then(resolve)
          .catch(reject);
        return;
      }

      if (response.statusCode !== 200) {
        reject(new Error(`Failed to download: HTTP ${response.statusCode}`));
        return;
      }

      const totalBytes = parseInt(response.headers['content-length'] || '0', 10);
      let downloadedBytes = 0;

      // Speed calculation
      let lastBytes = 0;
      let lastTime = Date.now();
      let speed = 0;

      const speedInterval = setInterval(() => {
        const now = Date.now();
        const timeDiff = (now - lastTime) / 1000; // seconds
        const bytesDiff = downloadedBytes - lastBytes;
        speed = timeDiff > 0 ? bytesDiff / timeDiff : 0;
        lastBytes = downloadedBytes;
        lastTime = now;
      }, 1000);

      const fileStream = fs.createWriteStream(localPath);

      response.on('data', (chunk) => {
        downloadedBytes += chunk.length;
        onProgress({
          downloadedBytes,
          totalBytes,
          speed
        });
      });

      response.pipe(fileStream);

      fileStream.on('finish', () => {
        clearInterval(speedInterval);
        fileStream.close();
        resolve(localPath);
      });

      fileStream.on('error', (err) => {
        clearInterval(speedInterval);
        fs.unlink(localPath, () => {}); // Clean up partial download
        reject(err);
      });
    });

    request.on('error', (err) => {
      reject(err);
    });
  });
}

/**
 * Determines if a string is a URL or local path
 */
export function isUrl(modelPath: string): boolean {
  try {
    new URL(modelPath);
    return modelPath.startsWith('http://') || modelPath.startsWith('https://');
  } catch {
    return false;
  }
}

/**
 * Resolves a model path (URL or local) to a local file path
 */
export async function resolveModelPath(
  modelPath: string,
  onProgress?: (progress: DownloadProgress) => void
): Promise<string> {
  if (isUrl(modelPath)) {
    const filename = getFilenameFromUrl(modelPath);

    // Check cache first
    if (isModelCached(filename)) {
      const localPath = getLocalModelPath(filename);
      // Report as already downloaded
      if (onProgress) {
        const stats = fs.statSync(localPath);
        onProgress({
          downloadedBytes: stats.size,
          totalBytes: stats.size,
          speed: 0
        });
      }
      return localPath;
    }

    // Download if not cached
    return downloadModel(modelPath, onProgress || (() => {}));
  }

  // Expand ~ to home directory
  if (modelPath.startsWith('~')) {
    return path.join(os.homedir(), modelPath.slice(1));
  }

  // Return as-is if absolute path
  if (path.isAbsolute(modelPath)) {
    return modelPath;
  }

  // Resolve relative path
  return path.resolve(process.cwd(), modelPath);
}
