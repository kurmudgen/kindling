import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CONFIG_DIR = resolve(__dirname, '../../config');

export interface KindlingConfig {
  profile: string;
  tier1: {
    model: string;
    fallbackModel?: string;
    apiFallbackModel?: string;
    maxCores: number;
    quantization: string;
  };
  tier2: {
    model: string;
    fallbackModel?: string;
    apiFallbackModel?: string;
    maxCores: number;
    quantization: string;
    apiFallback: boolean;
  };
  tier3: {
    model: string;
    apiOnly: boolean;
    localModel?: string;
    fallbackLocalModel?: string;
    apiFallbackModel?: string;
    keepAlive?: string;
  };
  buffer: {
    size: number;
  };
  escalation: {
    tokenProbabilitySpreadWeight: number;
    semanticVelocityWeight: number;
    surpriseScoreWeight: number;
    attentionAnomalyWeight: number;
    escalationThreshold: number;
    deescalationThreshold: number;
    deescalationTokenWindow: number;
  };
  sleep: {
    idleThresholdMinutes: number;
    apiModel: string;
    decayHalfLifeSessions?: number;
    autoSleepMinQueries?: number;
  };
  recovery?: {
    maxRetries: number;
    backoffBaseMs: number;
    tierGenerationTimeoutMs: number;
    circuitBreakerThreshold: number;
  };
}

/**
 * Schema for learned.json — decay-aware routing adjustments.
 * Each adjustment has a session age that ticks up over time,
 * and an applied timestamp for audit purposes.
 */
export interface LearnedStore {
  sessionCount: number;
  adjustments: LearnedAdjustment[];
}

export interface LearnedAdjustment {
  signal: string; // config key, e.g. "tokenProbabilitySpreadWeight"
  suggestedWeight: number;
  appliedAt: string; // ISO timestamp
  sessionAge: number; // 0 = fresh, increments each session
}

function loadJsonFile(path: string): Record<string, unknown> {
  return JSON.parse(readFileSync(path, 'utf-8')) as Record<string, unknown>;
}

/**
 * Exponential decay. An adjustment loses half its influence after HALF_LIFE sessions.
 * Returns the effective weight blended between default and learned based on age.
 */
export function applyDecay(
  defaultWeight: number,
  suggestedWeight: number,
  sessionAge: number,
  halfLifeSessions: number
): number {
  if (halfLifeSessions <= 0) return suggestedWeight;
  const decayFactor = Math.pow(0.5, sessionAge / halfLifeSessions);
  return defaultWeight + (suggestedWeight - defaultWeight) * decayFactor;
}

let _config: KindlingConfig | null = null;

export function loadConfig(profileOverride?: string): KindlingConfig {
  const profileName = profileOverride ?? process.env.KINDLING_PROFILE ?? 'default';
  const profilePath = resolve(CONFIG_DIR, `${profileName}.json`);

  if (!existsSync(profilePath)) {
    throw new Error(`Hardware profile not found: ${profilePath}`);
  }

  const profileConfig = loadJsonFile(profilePath) as unknown as KindlingConfig;

  // Apply learned adjustments with decay (unless explicitly disabled)
  if (!process.env.KINDLING_IGNORE_LEARNED) {
    const learnedPath = resolve(CONFIG_DIR, 'learned.json');
    if (existsSync(learnedPath)) {
      const learned = readLearnedStore(learnedPath);
      if (learned && learned.adjustments.length > 0) {
        applyLearnedAdjustments(profileConfig, learned);
      }
    }
  }

  _config = profileConfig;
  return _config;
}

/**
 * Read learned.json. Handles both new (Phase 3.5) schema and legacy
 * (Phase 3) schema. Legacy entries are migrated to session age 0.
 */
export function readLearnedStore(path: string): LearnedStore | null {
  if (!existsSync(path)) return null;
  try {
    const raw = JSON.parse(readFileSync(path, 'utf-8')) as Record<string, unknown>;

    // New schema
    if (Array.isArray(raw.adjustments) && typeof raw.sessionCount === 'number') {
      return raw as unknown as LearnedStore;
    }

    // Legacy schema: { escalation: { ... } }
    if (raw.escalation && typeof raw.escalation === 'object') {
      const escalation = raw.escalation as Record<string, number>;
      const adjustments: LearnedAdjustment[] = Object.entries(escalation).map(
        ([signal, suggestedWeight]) => ({
          signal,
          suggestedWeight,
          appliedAt: new Date().toISOString(),
          sessionAge: 0, // migrated entries treated as fresh
        })
      );
      return {
        sessionCount: 1,
        adjustments,
      };
    }

    return null;
  } catch {
    return null;
  }
}

function applyLearnedAdjustments(config: KindlingConfig, learned: LearnedStore): void {
  const halfLife = config.sleep.decayHalfLifeSessions ?? 10;
  const escalation = config.escalation as unknown as Record<string, number>;

  for (const adj of learned.adjustments) {
    if (escalation[adj.signal] === undefined) continue;
    const defaultWeight = escalation[adj.signal];
    const effectiveWeight = applyDecay(
      defaultWeight,
      adj.suggestedWeight,
      adj.sessionAge,
      halfLife
    );
    escalation[adj.signal] = effectiveWeight;
  }
}

export function getConfig(): KindlingConfig {
  if (!_config) {
    return loadConfig();
  }
  return _config;
}

export function getOllamaHost(): string {
  return process.env.OLLAMA_HOST ?? 'http://localhost:11434';
}
