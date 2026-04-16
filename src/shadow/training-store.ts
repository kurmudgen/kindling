/**
 * Training Data Store — Phase 4
 *
 * Structured JSONL storage for labeled training examples produced by
 * shadow evaluation. Each record captures the full signal state at
 * decision time plus ground-truth quality comparison (local vs API).
 *
 * This is the dataset that feeds the ML meta-confidence classifier.
 * Records accumulate over sessions; the dream consolidator reads them
 * in batches for training and analysis.
 *
 * Storage: logs/shadow/training.jsonl (append-only, serialized writes)
 */

import { appendFile, mkdir, readFile, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import pino from 'pino';
import type { EscalationSignals, ValenceScore } from '../tiers/tier-interface.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const SHADOW_DIR = resolve(__dirname, '../../logs/shadow');
const TRAINING_FILE = resolve(SHADOW_DIR, 'training.jsonl');

const log = pino({ level: process.env.KINDLING_LOG_LEVEL ?? 'info' });

/**
 * A single labeled training example from shadow evaluation.
 *
 * The key insight: localCoherence vs apiCoherence tells us whether
 * the local tier was "good enough" or whether escalation was needed.
 * Combined with signals + valence at decision time, this gives us
 * labeled data for supervised learning.
 */
export interface TrainingExample {
  timestamp: string;
  sessionId: string;
  queryHash: string;

  // Signal state at decision time
  signals: EscalationSignals;
  valence: ValenceScore;

  // What happened
  tierUsed: 1 | 2 | 3;
  routerDecision: 'escalate' | 'stay';
  metaAction: 'confirm' | 'suppress' | 'force';

  // Local tier output quality
  localTokenCount: number;
  localCoherence: number;
  localLatencyMs: number;

  // API ground truth quality
  apiTokenCount: number;
  apiCoherence: number;
  apiLatencyMs: number;
  apiModel: string;

  // Derived label: was escalation actually needed?
  // true = API produced meaningfully better output (local was insufficient)
  // false = local was good enough (escalation would have been wasteful)
  escalationNeeded: boolean;

  // Quality delta: apiCoherence - localCoherence
  // Positive = API was better, negative = local was surprisingly better
  qualityDelta: number;
}

// Serialized write queue — same pattern as logger.ts
let writeQueue: Promise<void> = Promise.resolve();

function enqueueWrite(line: string): void {
  writeQueue = writeQueue
    .then(async () => {
      if (!existsSync(SHADOW_DIR)) {
        await mkdir(SHADOW_DIR, { recursive: true });
      }
      await appendFile(TRAINING_FILE, line, 'utf-8');
    })
    .catch((err) => {
      log.warn({ err }, 'Failed to write training example');
    });
}

/**
 * Append a labeled training example to the store.
 */
export function recordTrainingExample(example: TrainingExample): void {
  enqueueWrite(JSON.stringify(example) + '\n');
  log.debug(
    {
      tier: example.tierUsed,
      localCoherence: example.localCoherence.toFixed(2),
      apiCoherence: example.apiCoherence.toFixed(2),
      delta: example.qualityDelta.toFixed(2),
      needed: example.escalationNeeded,
    },
    'Shadow training example recorded'
  );
}

/**
 * Read all training examples from the store.
 * Used by dream consolidator and ML training pipeline.
 */
export async function loadTrainingExamples(): Promise<TrainingExample[]> {
  if (!existsSync(TRAINING_FILE)) return [];

  try {
    const content = await readFile(TRAINING_FILE, 'utf-8');
    const lines = content.trim().split('\n').filter(Boolean);
    const examples: TrainingExample[] = [];

    for (const line of lines) {
      try {
        examples.push(JSON.parse(line) as TrainingExample);
      } catch {
        // Skip malformed lines
      }
    }

    return examples;
  } catch {
    return [];
  }
}

/**
 * Get training store stats without loading all examples into memory.
 */
export async function getTrainingStats(): Promise<{
  totalExamples: number;
  fileSizeBytes: number;
  exists: boolean;
}> {
  if (!existsSync(TRAINING_FILE)) {
    return { totalExamples: 0, fileSizeBytes: 0, exists: false };
  }

  try {
    const content = await readFile(TRAINING_FILE, 'utf-8');
    const lines = content.trim().split('\n').filter(Boolean);
    const fileInfo = await stat(TRAINING_FILE);
    return {
      totalExamples: lines.length,
      fileSizeBytes: fileInfo.size,
      exists: true,
    };
  } catch {
    return { totalExamples: 0, fileSizeBytes: 0, exists: false };
  }
}

/**
 * Load the most recent N training examples (for dream consolidation).
 * More memory-efficient than loading everything when the store is large.
 */
export async function loadRecentExamples(count: number): Promise<TrainingExample[]> {
  const all = await loadTrainingExamples();
  return all.slice(-count);
}

export function getTrainingFilePath(): string {
  return TRAINING_FILE;
}
