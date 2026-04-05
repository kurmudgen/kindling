/**
 * 5-Layer Error Recovery Cascade
 *
 * Layer 1 — Retry with backoff (transient errors)
 * Layer 2 — Tier downgrade (after retries exhausted)
 * Layer 3 — Model fallback (swap to a different model within the same tier)
 * Layer 4 — API fallback (local -> API equivalent)
 * Layer 5 — Circuit breaker (sustained degradation warning)
 */

import pino from 'pino';
import { getConfig } from '../config/config.js';
import type { TierQuery, TierResponse } from '../tiers/tier-interface.js';

const log = pino({ level: process.env.KINDLING_LOG_LEVEL ?? 'info' });

export type RecoveryLayer = 'retry' | 'tier_downgrade' | 'model_fallback' | 'api_fallback' | 'circuit_open';

export interface RecoveryEvent {
  timestamp: string;
  layer: RecoveryLayer;
  reason: string;
  originalTier: number;
  recoveredTier: number;
  success: boolean;
}

export interface RecoveryContext {
  tierId: 1 | 2 | 3;
  query: TierQuery;
  primaryGenerate: () => Promise<TierResponse>;
  fallbackModelGenerate?: () => Promise<TierResponse>;
  apiFallbackGenerate?: () => Promise<TierResponse>;
  tierDowngradeGenerate?: () => Promise<TierResponse>;
}

export interface RecoveryResult {
  response: TierResponse | null;
  events: RecoveryEvent[];
  finalLayer: RecoveryLayer | null;
}

/**
 * Errors that should trigger retries (transient network/timeout issues).
 */
function isTransientError(err: unknown): boolean {
  if (!err) return false;
  const e = err as { code?: string; name?: string; message?: string };
  const code = e.code ?? '';
  const name = e.name ?? '';
  const msg = (e.message ?? '').toLowerCase();

  return (
    code === 'ECONNRESET' ||
    code === 'ETIMEDOUT' ||
    code === 'ENOTFOUND' ||
    code === 'ECONNREFUSED' ||
    name === 'AbortError' ||
    name === 'TimeoutError' ||
    msg.includes('timeout') ||
    msg.includes('fetch failed') ||
    msg.includes('headers timeout') ||
    msg.includes('socket hang up')
  );
}

/**
 * Errors that mean the model is unavailable (not found, OOM).
 * These should skip retries and go straight to model fallback.
 */
function isModelUnavailableError(err: unknown): boolean {
  if (!err) return false;
  const e = err as { message?: string; status_code?: number };
  const msg = (e.message ?? '').toLowerCase();
  return (
    msg.includes('not found') ||
    msg.includes('out of memory') ||
    msg.includes('oom') ||
    e.status_code === 404
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Global circuit breaker state. Shared across the process.
 * Trips after N consecutive queries require Layer 3+ recovery.
 */
class CircuitBreaker {
  private consecutiveDegradations = 0;
  private tripped = false;

  recordDegradation(): void {
    this.consecutiveDegradations++;
    const threshold = getConfig().recovery?.circuitBreakerThreshold ?? 3;
    if (this.consecutiveDegradations >= threshold && !this.tripped) {
      this.tripped = true;
      log.warn(
        { consecutiveDegradations: this.consecutiveDegradations },
        'CIRCUIT BREAKER TRIPPED — Kindling is in degraded mode. Check Ollama health.'
      );
    }
  }

  recordCleanRun(): void {
    if (this.tripped) {
      log.info('Circuit breaker reset — clean query after degradation');
      this.tripped = false;
    }
    this.consecutiveDegradations = 0;
  }

  isTripped(): boolean {
    return this.tripped;
  }
}

export const circuitBreaker = new CircuitBreaker();

/**
 * Run a tier generation through the 5-layer recovery cascade.
 */
export async function runWithRecovery(ctx: RecoveryContext): Promise<RecoveryResult> {
  const events: RecoveryEvent[] = [];
  const config = getConfig();
  const maxRetries = config.recovery?.maxRetries ?? 3;
  const backoffBase = config.recovery?.backoffBaseMs ?? 1000;
  const timeoutMs = config.recovery?.tierGenerationTimeoutMs ?? 120000;

  // Layer 1: Retry with backoff
  let lastError: unknown = null;
  let usedLayer3Plus = false;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await withTimeout(ctx.primaryGenerate(), timeoutMs);
      if (attempt === 0) {
        // Clean success on first try
        circuitBreaker.recordCleanRun();
      }
      return { response, events, finalLayer: attempt > 0 ? 'retry' : null };
    } catch (err) {
      lastError = err;

      if (isModelUnavailableError(err)) {
        // Don't retry — go to model fallback immediately
        log.warn({ tier: ctx.tierId }, 'Model unavailable, skipping retries');
        break;
      }

      if (isTransientError(err) && attempt < maxRetries - 1) {
        const backoffMs = backoffBase * Math.pow(2, attempt);
        log.info(
          { tier: ctx.tierId, attempt: attempt + 1, backoffMs },
          'Transient error, retrying with backoff'
        );
        events.push({
          timestamp: new Date().toISOString(),
          layer: 'retry',
          reason: `transient error, attempt ${attempt + 1}`,
          originalTier: ctx.tierId,
          recoveredTier: ctx.tierId,
          success: false,
        });
        await sleep(backoffMs);
        continue;
      }

      // Non-transient error, stop retrying
      break;
    }
  }

  // Layer 3: Model fallback (within same tier)
  if (ctx.fallbackModelGenerate) {
    usedLayer3Plus = true;
    try {
      log.warn({ tier: ctx.tierId }, 'Primary model failed, trying fallback model');
      const response = await withTimeout(ctx.fallbackModelGenerate(), timeoutMs);
      events.push({
        timestamp: new Date().toISOString(),
        layer: 'model_fallback',
        reason: `primary model failed: ${errMsg(lastError)}`,
        originalTier: ctx.tierId,
        recoveredTier: ctx.tierId,
        success: true,
      });
      circuitBreaker.recordDegradation();
      return { response, events, finalLayer: 'model_fallback' };
    } catch (err) {
      lastError = err;
      events.push({
        timestamp: new Date().toISOString(),
        layer: 'model_fallback',
        reason: `fallback model also failed: ${errMsg(err)}`,
        originalTier: ctx.tierId,
        recoveredTier: ctx.tierId,
        success: false,
      });
    }
  }

  // Layer 4: API fallback
  if (ctx.apiFallbackGenerate) {
    usedLayer3Plus = true;
    try {
      log.warn({ tier: ctx.tierId }, 'Local models failed, trying API fallback');
      const response = await withTimeout(ctx.apiFallbackGenerate(), timeoutMs);
      events.push({
        timestamp: new Date().toISOString(),
        layer: 'api_fallback',
        reason: `local tier failed: ${errMsg(lastError)}`,
        originalTier: ctx.tierId,
        recoveredTier: ctx.tierId,
        success: true,
      });
      circuitBreaker.recordDegradation();
      return { response, events, finalLayer: 'api_fallback' };
    } catch (err) {
      lastError = err;
      events.push({
        timestamp: new Date().toISOString(),
        layer: 'api_fallback',
        reason: `API fallback also failed: ${errMsg(err)}`,
        originalTier: ctx.tierId,
        recoveredTier: ctx.tierId,
        success: false,
      });
    }
  }

  // Layer 2: Tier downgrade (fall DOWN to a cheaper tier)
  if (ctx.tierDowngradeGenerate && ctx.tierId > 1) {
    usedLayer3Plus = true;
    try {
      log.warn({ from: ctx.tierId, to: ctx.tierId - 1 }, 'All recovery failed at this tier, downgrading');
      const response = await withTimeout(ctx.tierDowngradeGenerate(), timeoutMs);
      events.push({
        timestamp: new Date().toISOString(),
        layer: 'tier_downgrade',
        reason: `tier ${ctx.tierId} fully unavailable: ${errMsg(lastError)}`,
        originalTier: ctx.tierId,
        recoveredTier: ctx.tierId - 1,
        success: true,
      });
      circuitBreaker.recordDegradation();
      return { response, events, finalLayer: 'tier_downgrade' };
    } catch (err) {
      lastError = err;
    }
  }

  // Layer 5: All recovery exhausted — circuit breaker state
  if (usedLayer3Plus) {
    circuitBreaker.recordDegradation();
  }
  if (circuitBreaker.isTripped()) {
    events.push({
      timestamp: new Date().toISOString(),
      layer: 'circuit_open',
      reason: 'circuit breaker tripped, all recovery exhausted',
      originalTier: ctx.tierId,
      recoveredTier: ctx.tierId,
      success: false,
    });
  }

  log.error({ err: lastError, tier: ctx.tierId }, 'All 5 recovery layers exhausted');
  return { response: null, events, finalLayer: null };
}

function errMsg(err: unknown): string {
  if (!err) return 'unknown';
  const e = err as { message?: string };
  return e.message ?? String(err);
}

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => {
      const err = new Error(`Tier generation timeout after ${ms}ms`);
      err.name = 'TimeoutError';
      reject(err);
    }, ms);
    p.then(
      v => {
        clearTimeout(t);
        resolve(v);
      },
      e => {
        clearTimeout(t);
        reject(e);
      }
    );
  });
}
