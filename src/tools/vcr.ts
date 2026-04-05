/**
 * VCR Session Replay
 *
 * Record mode: writes every routing decision to logs/vcr/SESSION_ID.jsonl
 * Replay mode: re-runs decisions through current confidence + meta models
 *              and diffs against the recorded decisions.
 *
 * Activated via environment variables:
 *   KINDLING_VCR=record          — records to logs/vcr/<sessionId>.jsonl
 *   KINDLING_VCR=replay          — replays via CLI (this file)
 *   KINDLING_VCR_FILE=<path>     — replay target file
 *
 * Privacy: only prompt HASHES are recorded, never raw prompt text.
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { readFileSync, existsSync, appendFileSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import type { EscalationSignals, ValenceScore } from '../tiers/tier-interface.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const VCR_DIR = resolve(__dirname, '../../logs/vcr');

export interface VcrEntry {
  timestamp: string;
  queryHash: string;
  valence: ValenceScore;
  signals: EscalationSignals;
  signalsAtEscalation?: EscalationSignals;
  startTier: number;
  finalTier: number;
  confidence: number;
  routerDecision: 'escalate' | 'stay';
  metaAction: 'confirm' | 'suppress' | 'force';
  escalated: boolean;
  responseTokens: number;
  recoveryEventCount: number;
}

// ==================== RECORD MODE API ====================

let _currentSessionFile: string | null = null;

export function isRecordingActive(): boolean {
  return process.env.KINDLING_VCR === 'record';
}

export function getVcrSessionFile(): string {
  if (_currentSessionFile) return _currentSessionFile;
  if (!existsSync(VCR_DIR)) mkdirSync(VCR_DIR, { recursive: true });
  const sessionId = randomUUID();
  _currentSessionFile = resolve(VCR_DIR, `${sessionId}.jsonl`);
  return _currentSessionFile;
}

export function recordVcrEntry(entry: VcrEntry): void {
  if (!isRecordingActive()) return;
  const file = getVcrSessionFile();
  try {
    appendFileSync(file, JSON.stringify(entry) + '\n', 'utf-8');
  } catch {
    // Swallow — VCR must not crash inference
  }
}

// ==================== REPLAY MODE ====================

interface ReplayResult {
  total: number;
  matches: number;
  mismatches: Array<{
    queryHash: string;
    recorded: { startTier: number; finalTier: number; confidence: number; escalated: boolean };
    current: { startTier: number; finalTier: number; confidence: number; escalated: boolean };
    reason: string;
  }>;
}

async function replayMain(): Promise<void> {
  const filePath = process.env.KINDLING_VCR_FILE;
  if (!filePath) {
    console.error('KINDLING_VCR_FILE not set. Usage:');
    console.error('  KINDLING_VCR_FILE=logs/vcr/SESSION_ID.jsonl npx tsx src/tools/vcr.ts');
    process.exit(1);
  }

  const absPath = resolve(filePath);
  if (!existsSync(absPath)) {
    console.error(`VCR file not found: ${absPath}`);
    process.exit(1);
  }

  const raw = readFileSync(absPath, 'utf-8').trim();
  const entries: VcrEntry[] = raw
    .split('\n')
    .map(l => {
      try {
        return JSON.parse(l) as VcrEntry;
      } catch {
        return null;
      }
    })
    .filter((e): e is VcrEntry => e !== null);

  console.log(`\n=== VCR REPLAY: ${absPath} ===`);
  console.log(`Loaded ${entries.length} recorded entries\n`);

  // Dynamic imports so config loads with correct env
  const { loadConfig } = await import('../config/config.js');
  const { ConfidenceAggregator } = await import('../router/confidence.js');
  const { MetaConfidenceModel } = await import('../meta/meta-confidence.js');

  loadConfig();
  const result: ReplayResult = { total: entries.length, matches: 0, mismatches: [] };

  for (const entry of entries) {
    // Fresh aggregator per entry — we're replaying individual decisions, not a continuous session
    const conf = new ConfidenceAggregator();
    const meta = new MetaConfidenceModel();

    const decision = conf.decide(
      entry.startTier as 1 | 2 | 3,
      entry.signals,
      entry.valence
    );
    const metaDecision = meta.evaluate(
      decision,
      entry.signals,
      entry.valence,
      entry.startTier
    );

    // Compute current "would-be" final tier
    let currentFinalTier = entry.startTier;
    let currentEscalated = false;
    let shouldEscalate = decision.shouldEscalate;
    if (metaDecision.action === 'suppress' && shouldEscalate) {
      shouldEscalate = false;
    } else if (metaDecision.action === 'force' && !shouldEscalate && entry.startTier < 3) {
      shouldEscalate = true;
    }
    if (shouldEscalate && decision.targetTier !== entry.startTier) {
      currentFinalTier = decision.targetTier;
      currentEscalated = true;
    }

    const currentConfidence = decision.confidence;
    const matches =
      currentFinalTier === entry.finalTier &&
      currentEscalated === entry.escalated;

    if (matches) {
      result.matches++;
    } else {
      // Figure out why it changed
      const reasons: string[] = [];
      if (Math.abs(currentConfidence - entry.confidence) > 0.01) {
        reasons.push(
          `confidence shift ${entry.confidence.toFixed(3)}→${currentConfidence.toFixed(3)}`
        );
      }
      if (currentEscalated !== entry.escalated) {
        reasons.push(
          `escalation flipped ${entry.escalated}→${currentEscalated}`
        );
      }
      if (currentFinalTier !== entry.finalTier) {
        reasons.push(
          `tier changed T${entry.finalTier}→T${currentFinalTier}`
        );
      }
      if (reasons.length === 0) {
        reasons.push('decision drift (internal state)');
      }

      result.mismatches.push({
        queryHash: entry.queryHash.slice(0, 12),
        recorded: {
          startTier: entry.startTier,
          finalTier: entry.finalTier,
          confidence: entry.confidence,
          escalated: entry.escalated,
        },
        current: {
          startTier: entry.startTier,
          finalTier: currentFinalTier,
          confidence: currentConfidence,
          escalated: currentEscalated,
        },
        reason: reasons.join(', '),
      });
    }
  }

  // Print report
  console.log(`Matches:    ${result.matches}/${result.total} (${((result.matches / result.total) * 100).toFixed(0)}%)`);
  console.log(`Mismatches: ${result.mismatches.length}`);

  if (result.mismatches.length > 0) {
    console.log('\n--- DIFFS ---');
    for (const m of result.mismatches) {
      console.log(
        `  [${m.queryHash}] T${m.recorded.startTier}→T${m.recorded.finalTier} (conf ${m.recorded.confidence.toFixed(3)}, esc=${m.recorded.escalated})`
      );
      console.log(
        `              T${m.current.startTier}→T${m.current.finalTier} (conf ${m.current.confidence.toFixed(3)}, esc=${m.current.escalated}) — ${m.reason}`
      );
    }
  }
}

// Only auto-run when this file is the CLI entry point, not when imported.
// Use a filename suffix match anchored on path separators to avoid false
// positives like test-vcr.ts matching "vcr.ts".
const entryArg = (process.argv[1] ?? '').replace(/\\/g, '/');
const isVcrCliEntry =
  entryArg.endsWith('/vcr.ts') ||
  entryArg.endsWith('/vcr.js') ||
  entryArg === 'vcr.ts' ||
  entryArg === 'vcr.js';
if (isVcrCliEntry) {
  replayMain().catch(err => {
    console.error('VCR replay failed:', err);
    process.exit(1);
  });
}
