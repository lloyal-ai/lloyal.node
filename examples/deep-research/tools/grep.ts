import type { Resource } from '../resources/types.js';
import type { Tool } from './types.js';

export function createGrepTool(resources: Resource[]): Tool {
  return {
    name: 'grep',
    schema: {
      type: 'function',
      function: {
        name: 'grep',
        description: 'Search the entire corpus for a regex pattern. Returns every matching line with line numbers and total match count. Complements search() which ranks by relevance — grep scans exhaustively.',
        parameters: {
          type: 'object',
          properties: {
            pattern: { type: 'string', description: 'Regex pattern (e.g. "\\bshor\\b" for whole-word, "hidden_secret" for literal)' },
            ignoreCase: { type: 'boolean', description: 'Case-insensitive matching (default: true)' },
          },
          required: ['pattern'],
        },
      },
    },
    async execute(args) {
      const pattern = (args.pattern as string)?.trim();
      if (!pattern) return { error: 'pattern must not be empty' };
      const flags = (args.ignoreCase === false) ? 'g' : 'gi';
      let re: RegExp;
      try { re = new RegExp(pattern, flags); }
      catch { return { error: `Invalid regex: ${pattern}` }; }

      const matches: { file: string; line: number; text: string }[] = [];
      let totalMatches = 0;

      for (const res of resources) {
        const lines = res.content.split('\n');
        for (let i = 0; i < lines.length; i++) {
          const hits = lines[i].match(re);
          if (hits) {
            totalMatches += hits.length;
            const raw = lines[i].trim();
            let text: string;
            if (raw.length <= 200) {
              text = raw;
            } else {
              // Truncate around first match so the matched term is always visible
              const idx = raw.search(re);
              const start = Math.max(0, idx - 40);
              const end = Math.min(raw.length, start + 200);
              text = (start > 0 ? '…' : '') + raw.slice(start, end) + (end < raw.length ? '…' : '');
            }
            matches.push({ file: res.name, line: i + 1, text });
          }
        }
      }

      if (totalMatches === 0) {
        return {
          totalMatches: 0, matchingLines: 0, matches: [],
          note: 'Zero matches does NOT mean the topic is absent — only that this exact pattern was not found. Try search() for semantic matching or a broader/simpler regex.',
        };
      }

      const limit = 50;
      const truncated = matches.length > limit;
      return { totalMatches, matchingLines: matches.length, truncated, matches: matches.slice(0, limit) };
    },
  };
}
