#!/usr/bin/env node
/**
 * Infinite context generation with dynamic summary sinks
 *
 * Usage:
 *   node streaming-summary.mjs [model-path]          # Human-readable output
 *   node streaming-summary.mjs [model-path] --jsonl  # JSONL output for testing
 *
 * This example demonstrates:
 * - BlinkKV reseeding with ghostwritten progress sinks
 * - Sidecar model (slim-summarize) for evicted content summarization
 * - Outline detection with structural progress tracking
 * - Pattern matching (not instruction following) to guide continuation
 * - Graceful degradation when sidecar model is missing
 *
 * After reseed, KV cache contains: [progress][tail]
 * - progress = minimal anchor + checklist of done/current sections + summary
 * - tail     = recent 256 tokens for continuity
 *
 * The progress sink uses "done" / "continue from here" markers that the
 * model pattern-matches against, rather than relying on instruction following.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createContext } from '../../lib/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_MODEL = path.resolve(
  __dirname,
  '../../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf'
);
const SUMMARY_MODEL = path.resolve(
  __dirname,
  '../../models/slim-summarize.gguf'
);

// Parse args
const args = process.argv.slice(2);
const jsonlMode = args.includes('--jsonl');
const modelPath = args.find(a => !a.startsWith('--')) || DEFAULT_MODEL;

/** Emit output - JSONL or human-readable */
function emit(event, data) {
  if (jsonlMode) {
    console.log(JSON.stringify({ event, ...data }));
  }
}

/**
 * Parse slim-summarize output (Python-style list) into readable text
 */
function parseSummaryOutput(raw) {
  // Output is Python-style list: ['point1', 'point2', ...]
  // Items may contain apostrophes (e.g., "It's"), so we can't match between quotes.
  // Instead, strip outer brackets + quotes, then split on the item boundary: ', '
  let inner = raw.trim();
  if (inner.startsWith('[')) inner = inner.slice(1);
  if (inner.endsWith(']')) inner = inner.slice(0, -1);
  inner = inner.trim();
  if (inner.startsWith("'") || inner.startsWith('"')) inner = inner.slice(1);
  if (inner.endsWith("'") || inner.endsWith('"')) inner = inner.slice(0, -1);

  if (!inner) return raw.trim();

  // Split on quote-comma-quote boundaries (handles apostrophes within items)
  const items = inner.split(/['"]\s*,\s*['"]/)
    .map(s => s.trim())
    .filter(Boolean);

  if (items.length > 0) return items.join('\n');
  return inner;
}

/**
 * Generate a summary using the sidecar model
 */
async function generateSummary(summaryCtx, text, options = {}) {
  const maxTokens = options.maxTokens || 200;
  const paramStr = options.brief
    ? 'brief description (1)'
    : 'key points (5)';

  const prompt = `<human> ${text.slice(-10000)}\n<summarize> ${paramStr}</summarize>\n<bot>:`;
  const tokens = await summaryCtx.tokenize(prompt);

  await summaryCtx.kvCacheClear();
  await summaryCtx.decode(tokens, 0, 0);

  let response = '';
  let pos = tokens.length;
  for (let i = 0; i < maxTokens; i++) {
    const token = summaryCtx.sample({ temperature: 0.3 });
    if (summaryCtx.isStopToken(token)) break;
    response += summaryCtx.tokenToText(token);
    await summaryCtx.decode([token], pos++, 0);
  }

  return parseSummaryOutput(response.trim());
}

/**
 * Parse numbered outline items from prompt text.
 */
function parseOutline(text) {
  const items = [];
  const regex = /^\s*(\d+)\.\s+(.+?)(?:\s*[-–—:]\s*.*)?$/gm;
  let match;
  while ((match = regex.exec(text)) !== null) {
    items.push({
      number: parseInt(match[1]),
      title: match[2].trim(),
    });
  }
  return items;
}

/**
 * Extract instruction part of prompt, before any numbered outline.
 */
function extractMinimalAnchor(text) {
  const listMatch = text.match(/^\s*1\.\s/m);
  if (listMatch && listMatch.index > 0) {
    return text.slice(0, listMatch.index).trim();
  }
  return text.slice(0, 200).trim();
}

/**
 * Build ghostwritten progress sink.
 * Completed items show "- done", current shows "- continue from here".
 * Model pattern-matches to continue from the right section.
 */
function buildProgressSink(anchor, outline, allGeneratedText, summaryChain) {
  const lower = allGeneratedText.toLowerCase();

  let lastCoveredIdx = -1;
  for (let i = outline.length - 1; i >= 0; i--) {
    if (lower.includes(outline[i].title.toLowerCase())) {
      lastCoveredIdx = i;
      break;
    }
  }

  let text = `${anchor}\n\n`;

  for (let i = 0; i < outline.length; i++) {
    const item = outline[i];
    if (i < lastCoveredIdx) {
      text += `${item.number}. ${item.title} - done\n`;
    } else if (i === lastCoveredIdx) {
      text += `${item.number}. ${item.title} - continue from here\n`;
    } else {
      text += `${item.number}. ${item.title}\n`;
    }
  }

  if (summaryChain) {
    text += `\nKey points so far:\n${summaryChain}\n`;
  }

  return text;
}

async function main() {
  // Constants
  const nCtx = 2048;
  const TAIL_SIZE = 256;
  const MAX_SINK_RATIO = 0.4;
  const MAX_SINK_TOKENS = Math.floor(nCtx * MAX_SINK_RATIO);
  const TARGET_TOKENS = 5000;
  const SUMMARY_MAX_TOKENS = 200;

  if (!jsonlMode) {
    console.log(`Loading model: ${modelPath}`);
  }

  emit('start', {
    model: path.basename(modelPath),
    nCtx,
    tailSize: TAIL_SIZE,
    maxSinkTokens: MAX_SINK_TOKENS,
    targetTokens: TARGET_TOKENS,
  });

  const ctx = await createContext({
    modelPath,
    contextSize: nCtx,
  });

  // Summary sidecar — preload in background (overlaps with prompt decode + generation)
  const summaryModelAvailable = fs.existsSync(SUMMARY_MODEL);
  let summaryCtx = null;
  const summaryCtxPromise = summaryModelAvailable
    ? createContext({ modelPath: SUMMARY_MODEL, contextSize: 4096 })
    : null;
  if (!summaryModelAvailable) {
    if (!jsonlMode) {
      console.log('Summary model not found - running without summary sinks');
    }
    emit('summary_missing', { message: 'slim-summarize.gguf not found' });
  }

  const prompt = `Write a comprehensive guide to machine learning, covering the following topics in extreme detail with examples, code snippets, and mathematical formulas:

1. Linear Regression - derivation, implementation, regularization
2. Logistic Regression - binary and multiclass
3. Neural Networks - backpropagation, activation functions
4. Convolutional Neural Networks - architectures, pooling, stride
5. Recurrent Neural Networks - LSTM, GRU, attention
6. Transformers - self-attention, positional encoding
7. Optimization - SGD, Adam, learning rate schedules
8. Regularization - dropout, batch normalization, weight decay

Begin:

# Comprehensive Machine Learning Guide

## Chapter 1: Linear Regression

`;

  // Parse outline for ghostwritten progress sinks
  const outline = parseOutline(prompt);
  const minimalAnchor = outline.length > 0
    ? extractMinimalAnchor(prompt)
    : null;

  if (!jsonlMode) {
    console.log(`\nPrompt: "${prompt.slice(0, 100)}..."`);
    if (outline.length > 0) {
      console.log(`Outline detected: ${outline.length} sections`);
      console.log(`Minimal anchor: "${minimalAnchor}"`);
    }
  }

  const promptTokens = await ctx.tokenize(prompt);

  // Fallback anchor for prompts without outlines
  let anchorTokens = null;
  if (outline.length === 0) {
    anchorTokens = [...promptTokens];
  }

  const summaryBudget = outline.length > 0
    ? MAX_SINK_TOKENS
    : MAX_SINK_TOKENS - (anchorTokens?.length || 0);

  await ctx.decode(promptTokens, 0, 0);

  if (!jsonlMode) {
    console.log(`\nContext size: ${nCtx}`);
    console.log(`Target tokens: ${TARGET_TOKENS}`);
    console.log(`Sink budget: ${MAX_SINK_TOKENS} tokens`);
    console.log(`Tail size: ${TAIL_SIZE}`);
    console.log(`\nGenerating...\n`);
    process.stdout.write(prompt);
  }

  const allTokens = [...promptTokens];
  const tracker = ctx.createPerplexityTracker();
  let cachePos = promptTokens.length;
  let reseedCount = 0;
  let currentSegmentText = '';
  let allGeneratedText = '';
  const summaries = [];
  let pendingSummaryTokens = [];

  for (let t = 0; t < TARGET_TOKENS; t++) {
    const token = ctx.sample({
      temperature: 0.8,
      topP: 0.9,
    });

    if (ctx.isStopToken(token)) {
      if (!jsonlMode) {
        console.log('\n[EOS token reached]');
      }
      emit('eos', { tokenIndex: t });
      break;
    }

    const surprisal = ctx.modelSurprisal(token);
    ctx.addSurprisal(tracker, surprisal);

    const text = ctx.tokenToText(token);
    if (!jsonlMode) {
      process.stdout.write(text);
    }
    emit('token', { source: 'main', index: t, token, text, surprisal });

    currentSegmentText += text;
    allGeneratedText += text;
    allTokens.push(token);
    await ctx.decode([token], cachePos++, 0);

    // Cache full? Reseed with dynamic sinks
    if (cachePos >= nCtx) {
      // Estimate evicted portion of current segment only
      const tailCharsEstimate = TAIL_SIZE * 4;
      const evictedFromSegment = currentSegmentText.length > tailCharsEstimate
        ? currentSegmentText.slice(0, -tailCharsEstimate)
        : '';

      let sinks;

      // Resolve preloaded sidecar (should already be loaded by now)
      if (summaryModelAvailable && !summaryCtx) {
        summaryCtx = await summaryCtxPromise;
        if (!jsonlMode) {
          console.log('\n  [Summary sidecar loaded: slim-summarize.gguf]');
        }
        emit('summary_loaded', { model: 'slim-summarize.gguf' });
      }

      // Run summary sidecar if available
      let chainText = null;
      if (summaryCtx && evictedFromSegment.length > 0) {
        emit('summary_start', { reseedCount: reseedCount + 1 });
        const summaryStartTime = Date.now();

        if (!jsonlMode) {
          process.stdout.write(`\n  [Summarizing ${evictedFromSegment.length} evicted chars (page ${summaries.length + 1})...`);
        }

        const newPage = await generateSummary(summaryCtx, evictedFromSegment, {
          maxTokens: SUMMARY_MAX_TOKENS,
        });
        summaries.push(newPage);
        chainText = summaries.join('\n');

        // Fold oldest pages if chain is getting large
        let testTokens = await ctx.tokenize(chainText);
        if (testTokens.length > summaryBudget * 0.6) {
          if (!jsonlMode) {
            process.stdout.write(' (folding oldest pages)');
          }

          const foldCount = Math.max(1, Math.ceil(summaries.length / 2));
          const toFold = summaries.splice(0, foldCount);
          const folded = await generateSummary(summaryCtx, toFold.join('\n'), {
            brief: true,
            maxTokens: 100,
          });
          summaries.unshift(folded);
          chainText = summaries.join('\n');
        }

        const compressionRatio = evictedFromSegment.length > 0
          ? (evictedFromSegment.length / newPage.length).toFixed(1)
          : '0';
        const durationMs = Date.now() - summaryStartTime;

        emit('summary_complete', {
          reseedCount: reseedCount + 1,
          summary: newPage,
          summaryTokens: (await ctx.tokenize(chainText)).length,
          compressionRatio: parseFloat(compressionRatio),
          durationMs,
          pages: summaries.length,
        });

        if (!jsonlMode) {
          process.stdout.write(` ${compressionRatio}x, ${summaries.length} pages]`);
        }
      }

      // Build sinks — progress mode (outline detected) or fallback
      if (outline.length > 0) {
        const progressText = buildProgressSink(
          minimalAnchor, outline, allGeneratedText, chainText
        );
        let progressTokens = await ctx.tokenize(progressText);

        if (progressTokens.length > MAX_SINK_TOKENS) {
          // Drop summary details to fit budget
          const trimmedText = buildProgressSink(
            minimalAnchor, outline, allGeneratedText, null
          );
          progressTokens = await ctx.tokenize(trimmedText);
        }

        sinks = progressTokens;
        pendingSummaryTokens = progressTokens;

        if (!jsonlMode) {
          console.log(`\n  [Progress sink: ${progressTokens.length} tok]`);
          // Show progress state
          const lower = allGeneratedText.toLowerCase();
          let lastIdx = -1;
          for (let i = outline.length - 1; i >= 0; i--) {
            if (lower.includes(outline[i].title.toLowerCase())) {
              lastIdx = i; break;
            }
          }
          if (lastIdx >= 0) {
            console.log(`  [Sections done: ${lastIdx}, continuing: ${outline[lastIdx].title}]`);
          }
        }

        emit('sink_update', {
          anchorTokens: 0,
          summaryTokens: progressTokens.length,
          totalSinkTokens: progressTokens.length,
          budgetUsed: ((progressTokens.length / MAX_SINK_TOKENS) * 100).toFixed(1),
          budgetMax: MAX_SINK_TOKENS,
          pages: summaries.length,
          mode: 'progress',
        });
      } else if (chainText) {
        const wrapped = `Previously:\n${chainText}\n`;
        const summaryTokens = await ctx.tokenize(wrapped);
        sinks = [...anchorTokens, ...summaryTokens];
        pendingSummaryTokens = summaryTokens;

        if (!jsonlMode) {
          process.stdout.write(` ${summaryTokens.length} summary tok]`);
        }

        emit('sink_update', {
          anchorTokens: anchorTokens.length,
          summaryTokens: summaryTokens.length,
          totalSinkTokens: sinks.length,
          budgetUsed: ((sinks.length / MAX_SINK_TOKENS) * 100).toFixed(1),
          budgetMax: MAX_SINK_TOKENS,
          pages: summaries.length,
          mode: 'anchor',
        });
      } else {
        sinks = [...(anchorTokens || [])];
      }

      const tail = allTokens.slice(-TAIL_SIZE);
      await ctx.clearAndReseed(sinks, tail);
      cachePos = sinks.length + TAIL_SIZE;
      reseedCount++;

      const ppl = ctx.getPerplexity(tracker);
      emit('reseed', {
        count: reseedCount,
        tokenIndex: t + 1,
        ppl,
        sinkTokens: sinks.length,
        tailTokens: TAIL_SIZE,
        summaryPages: summaries.length,
        summaryPreview: summaries[summaries.length - 1]?.slice(0, 100) || '',
      });

      if (!jsonlMode) {
        console.log(`  [Reseed ${reseedCount} at token ${t + 1}/${TARGET_TOKENS} | PPL: ${ppl.toFixed(2)} | Sinks: ${sinks.length} tok | Pages: ${summaries.length}]`);
      }

      currentSegmentText = '';
    }

    // Progress indicator every 1000 tokens
    if ((t + 1) % 1000 === 0 && reseedCount === 0 && !jsonlMode) {
      console.log(`\n  [${t + 1}/${TARGET_TOKENS} tokens]`);
    }
  }

  const finalPpl = ctx.getPerplexity(tracker);
  ctx.freePerplexityTracker(tracker);

  const generatedTokens = allTokens.length - promptTokens.length;
  const finalChain = summaries.join('\n');
  emit('complete', {
    generatedTokens,
    reseeds: reseedCount,
    finalPpl,
    finalSummary: finalChain.slice(0, 300),
    finalSummaryTokens: pendingSummaryTokens.length,
    summaryPages: summaries.length,
  });

  if (!jsonlMode) {
    console.log('\n\n' + '='.repeat(50));
    console.log(`Generated: ${generatedTokens} tokens`);
    console.log(`Reseeds: ${reseedCount}`);
    console.log(`Final perplexity: ${finalPpl.toFixed(2)}`);
    if (summaries.length > 0) {
      console.log(`Summary pages: ${summaries.length}`);
      console.log(`Final chain (${pendingSummaryTokens.length} tok): ${finalChain.slice(0, 200)}`);
    }
    console.log('='.repeat(50));
  }

  ctx.dispose();
  if (summaryCtx) summaryCtx.dispose();
}

main().catch((err) => {
  console.error('Error:', err.message);
  process.exit(1);
});
