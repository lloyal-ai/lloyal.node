import * as fs from 'node:fs';
import * as path from 'node:path';
import { loadBinary } from '../../../dist/index.js';
import type { Resource, Chunk } from './types.js';

interface Section { heading: string; level: number; startLine: number; endLine: number }
const { parseMarkdown } = loadBinary() as unknown as { parseMarkdown(text: string): Section[] };

export function loadResources(dir: string): Resource[] {
  if (!fs.existsSync(dir)) {
    process.stdout.write(`Error: corpus not found: ${dir}\n`);
    process.exit(1);
  }
  const stat = fs.statSync(dir);
  if (stat.isFile()) {
    return [{ name: path.basename(dir), content: fs.readFileSync(dir, 'utf8') }];
  }
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.md'));
  if (!files.length) {
    process.stdout.write(`Error: no .md files in: ${dir}\n`);
    process.exit(1);
  }
  return files.map((f) => ({
    name: f,
    content: fs.readFileSync(path.join(dir, f), 'utf8'),
  }));
}

export function chunkResources(resources: Resource[]): Chunk[] {
  const out: Chunk[] = [];
  for (const res of resources) {
    const sections = parseMarkdown(res.content);
    const lines = res.content.split('\n');
    for (const sec of sections) {
      const text = lines.slice(sec.startLine - 1, sec.endLine).join('\n').trim();
      if (!text) continue;
      out.push({
        resource: res.name, heading: sec.heading || res.name, text, tokens: [],
        startLine: sec.startLine, endLine: sec.endLine,
      });
    }
  }
  return out;
}
