import { appendFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash, randomUUID } from 'node:crypto';
import type { ValenceScore } from '../tiers/tier-interface.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const LOG_DIR = resolve(__dirname, '../../logs');
const LOG_FILE = resolve(LOG_DIR, 'escalation.jsonl');

export interface EscalationEvent {
  timestamp: string;
  sessionId: string;
  queryHash: string;
  valenceScore: ValenceScore;
  tier1ConfidenceAtHandoff: number;
  escalatedToTier: 1 | 2 | 3;
  tokensBeforeEscalation: number;
  handoffSuccessful: boolean;
  totalLatencyMs: number;
}

let sessionId = randomUUID();

export function resetSessionId(): void {
  sessionId = randomUUID();
}

export function getSessionId(): string {
  return sessionId;
}

export function hashPrompt(prompt: string): string {
  return createHash('sha256').update(prompt).digest('hex');
}

export function logEscalation(event: Omit<EscalationEvent, 'timestamp' | 'sessionId'>): void {
  const fullEvent: EscalationEvent = {
    timestamp: new Date().toISOString(),
    sessionId,
    ...event,
  };
  enqueueWrite(JSON.stringify(fullEvent) + '\n');
}

export function getLogFilePath(): string {
  return LOG_FILE;
}

export interface RecoveryLogEntry {
  timestamp: string;
  sessionId: string;
  type: 'recovery';
  layer: string;
  reason: string;
  originalTier: number;
  recoveredTier: number;
  success: boolean;
}

export function logRecoveryEvent(event: {
  layer: string;
  reason: string;
  originalTier: number;
  recoveredTier: number;
  success: boolean;
  timestamp?: string;
}): void {
  const entry: RecoveryLogEntry = {
    timestamp: event.timestamp ?? new Date().toISOString(),
    sessionId,
    type: 'recovery',
    layer: event.layer,
    reason: event.reason,
    originalTier: event.originalTier,
    recoveredTier: event.recoveredTier,
    success: event.success,
  };

  enqueueWrite(JSON.stringify(entry) + '\n');
}

// Serialized write queue — prevents interleaved writes from concurrent queries
let writeQueue: Promise<void> = Promise.resolve();

function enqueueWrite(line: string): void {
  writeQueue = writeQueue
    .then(async () => {
      if (!existsSync(LOG_DIR)) {
        await mkdir(LOG_DIR, { recursive: true });
      }
      await appendFile(LOG_FILE, line, 'utf-8');
    })
    .catch(() => {
      // Swallow — logging must not crash the runtime
    });
}
