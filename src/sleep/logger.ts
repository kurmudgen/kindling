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

  // Fire-and-forget async write — don't block the hot path
  (async () => {
    if (!existsSync(LOG_DIR)) {
      await mkdir(LOG_DIR, { recursive: true });
    }
    await appendFile(LOG_FILE, JSON.stringify(fullEvent) + '\n', 'utf-8');
  })().catch(() => {
    // Swallow write errors — logging should never crash the runtime
  });
}

export function getLogFilePath(): string {
  return LOG_FILE;
}
