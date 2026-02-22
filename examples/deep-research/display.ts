let _jsonlMode = false;

export function setJsonlMode(on: boolean): void { _jsonlMode = on; }

const isTTY = process.stdout.isTTY;

export const c = isTTY ? {
  bold: '\x1b[1m', dim: '\x1b[2m', reset: '\x1b[0m',
  green: '\x1b[32m', cyan: '\x1b[36m', yellow: '\x1b[33m', red: '\x1b[31m',
} : { bold: '', dim: '', reset: '', green: '', cyan: '', yellow: '', red: '' };

export const log = (...a: unknown[]): void => { if (!_jsonlMode) console.log(...a); };

export function emit(event: string, data: Record<string, unknown>): void {
  if (_jsonlMode) console.log(JSON.stringify({ event, ...data }));
}

export const sec = (a: number, b: number): string => ((b - a) / 1000).toFixed(1);
export const pad = (s: unknown, n: number): string => String(s).padStart(n);
export const fmtSize = (bytes: number): string => bytes > 1e9
  ? (bytes / 1e9).toFixed(1) + ' GB'
  : (bytes / 1e6).toFixed(0) + ' MB';
