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
    maxCores: number;
    quantization: string;
  };
  tier2: {
    model: string;
    maxCores: number;
    quantization: string;
    apiFallback: boolean;
  };
  tier3: {
    model: string;
    apiOnly: boolean;
    localModel?: string;
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
  };
}

function loadJsonFile(path: string): Record<string, unknown> {
  return JSON.parse(readFileSync(path, 'utf-8')) as Record<string, unknown>;
}

function deepMerge(base: Record<string, unknown>, overlay: Record<string, unknown>): Record<string, unknown> {
  const result = { ...base };
  for (const key of Object.keys(overlay)) {
    if (
      typeof overlay[key] === 'object' &&
      overlay[key] !== null &&
      !Array.isArray(overlay[key]) &&
      typeof base[key] === 'object' &&
      base[key] !== null
    ) {
      result[key] = deepMerge(
        base[key] as Record<string, unknown>,
        overlay[key] as Record<string, unknown>
      );
    } else {
      result[key] = overlay[key];
    }
  }
  return result;
}

let _config: KindlingConfig | null = null;

export function loadConfig(profileOverride?: string): KindlingConfig {
  const profileName = profileOverride ?? process.env.KINDLING_PROFILE ?? 'default';
  const profilePath = resolve(CONFIG_DIR, `${profileName}.json`);

  if (!existsSync(profilePath)) {
    throw new Error(`Hardware profile not found: ${profilePath}`);
  }

  let config = loadJsonFile(profilePath);

  // Merge learned adjustments if they exist (unless explicitly disabled)
  if (!process.env.KINDLING_IGNORE_LEARNED) {
    const learnedPath = resolve(CONFIG_DIR, 'learned.json');
    if (existsSync(learnedPath)) {
      const learned = loadJsonFile(learnedPath);
      config = deepMerge(config, learned);
    }
  }

  _config = config as unknown as KindlingConfig;
  return _config;
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
