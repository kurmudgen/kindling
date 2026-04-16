/**
 * Shadow Evaluator — Phase 4
 *
 * The API-as-teacher system. After a local tier generates a response,
 * the shadow evaluator optionally sends the same query to the API in
 * the background and compares output quality. This produces labeled
 * training data for the ML meta-confidence classifier.
 *
 * Two modes:
 *
 * 1. REAL-TIME SHADOW (during queries):
 *    Every Nth query, fire-and-forget an API comparison in the background.
 *    Non-blocking — the user gets the local response immediately.
 *    Rate-limited to avoid burning API credits.
 *
 * 2. DREAM SHADOW (during idle):
 *    The dream task can batch-shadow recent queries that weren't
 *    sampled during real-time. Uses stored prompts (privacy-controlled)
 *    to replay through the API while the system is idle.
 *
 * The GPU kickstart angle: training data accumulates here. When enough
 * examples exist, a training pass runs the classifier on GPU, exports
 * weights, and the meta-confidence model loads them into RAM for
 * CPU-only inference going forward.
 *
 * PRIVACY: prompts are stored with the same hash-only approach as VCR
 * for the training store. However, shadow evaluation needs the raw
 * prompt to send to the API — it holds it in memory only for the
 * duration of the API call, never persists raw text to disk.
 */

import Anthropic from '@anthropic-ai/sdk';
import pino from 'pino';
import { getConfig } from '../config/config.js';
import { getSessionId, hashPrompt } from '../sleep/logger.js';
import { recordTrainingExample } from './training-store.js';
import type { EscalationSignals, ValenceScore, TierQuery } from '../tiers/tier-interface.js';

const log = pino({ level: process.env.KINDLING_LOG_LEVEL ?? 'info' });

export interface ShadowConfig {
  /** Shadow every Nth query (0 = disabled) */
  sampleRate: number;
  /** API model to use as ground truth teacher */
  apiModel: string;
  /** Max tokens for shadow API call */
  maxTokens: number;
  /** Quality delta threshold: API must beat local by this much to count as "escalation needed" */
  qualityThreshold: number;
  /** Max concurrent shadow evaluations */
  maxConcurrent: number;
}

const DEFAULT_CONFIG: ShadowConfig = {
  sampleRate: 5, // every 5th query
  apiModel: 'claude-haiku-4-5-20251001', // cheap and fast for ground truth
  maxTokens: 1024,
  qualityThreshold: 0.15, // API must beat local coherence by 0.15 to flag as "needed"
  maxConcurrent: 2,
};

interface PendingShadow {
  prompt: string;
  context: string[];
  signals: EscalationSignals;
  valence: ValenceScore;
  tierUsed: 1 | 2 | 3;
  routerDecision: 'escalate' | 'stay';
  metaAction: 'confirm' | 'suppress' | 'force';
  localTokenCount: number;
  localCoherence: number;
  localLatencyMs: number;
}

export class ShadowEvaluator {
  private client: Anthropic | null = null;
  private config: ShadowConfig;
  private queryCounter = 0;
  private activeShadows = 0;
  private enabled: boolean;

  constructor(config?: Partial<ShadowConfig>) {
    // Merge: explicit overrides > profile config > defaults
    const profileConfig = getConfig().shadow ?? {};
    this.config = { ...DEFAULT_CONFIG, ...profileConfig, ...config };
    this.enabled = this.config.sampleRate > 0
      && process.env.KINDLING_SHADOW !== 'false'
      && !!process.env.ANTHROPIC_API_KEY;

    if (this.enabled) {
      this.client = new Anthropic();
      log.info(
        { sampleRate: this.config.sampleRate, model: this.config.apiModel },
        'Shadow evaluator enabled'
      );
    } else {
      log.info('Shadow evaluator disabled (no API key, KINDLING_SHADOW=false, or sampleRate=0)');
    }
  }

  /**
   * Called after every query. Decides whether to shadow this one.
   * If yes, fires the API call in the background — non-blocking.
   *
   * The raw prompt is held in memory only for the API call duration.
   */
  maybeShadow(pending: PendingShadow): void {
    if (!this.enabled || !this.client) return;

    this.queryCounter++;

    // Sample rate check
    if (this.queryCounter % this.config.sampleRate !== 0) return;

    // Concurrency limit
    if (this.activeShadows >= this.config.maxConcurrent) {
      log.debug('Shadow evaluation skipped — max concurrent reached');
      return;
    }

    // Fire and forget — do not await
    this.runShadow(pending).catch(err => {
      log.warn({ err }, 'Shadow evaluation failed (non-blocking)');
    });
  }

  /**
   * Run a single shadow evaluation: send the query to the API,
   * compare quality, and record a training example.
   */
  private async runShadow(pending: PendingShadow): Promise<void> {
    this.activeShadows++;
    const start = performance.now();

    try {
      const messages: Anthropic.MessageParam[] = [];

      if (pending.context.length > 0) {
        messages.push({ role: 'user', content: pending.context.join('\n') });
        messages.push({ role: 'assistant', content: 'Understood.' });
      }
      messages.push({ role: 'user', content: pending.prompt });

      const response = await this.client!.messages.create({
        model: this.config.apiModel,
        max_tokens: this.config.maxTokens,
        messages,
      });

      const apiLatencyMs = performance.now() - start;
      const apiText = response.content
        .filter((b): b is Anthropic.TextBlock => b.type === 'text')
        .map(b => b.text)
        .join('');
      const apiTokens = apiText.split(/\s+/).filter(Boolean);
      const apiCoherence = this.estimateCoherence(apiText);

      const qualityDelta = apiCoherence - pending.localCoherence;
      // Two conditions for "escalation was needed":
      // 1. API was meaningfully better than local (API teacher says local was insufficient)
      // 2. Router already escalated to a higher tier (ground truth: query needed escalation)
      const escalationNeeded =
        qualityDelta > this.config.qualityThreshold ||
        pending.tierUsed > 1;

      recordTrainingExample({
        timestamp: new Date().toISOString(),
        sessionId: getSessionId(),
        queryHash: hashPrompt(pending.prompt),

        signals: pending.signals,
        valence: pending.valence,

        tierUsed: pending.tierUsed,
        routerDecision: pending.routerDecision,
        metaAction: pending.metaAction,

        localTokenCount: pending.localTokenCount,
        localCoherence: pending.localCoherence,
        localLatencyMs: pending.localLatencyMs,

        apiTokenCount: apiTokens.length,
        apiCoherence,
        apiLatencyMs,
        apiModel: this.config.apiModel,

        escalationNeeded,
        qualityDelta,
      });

      log.info(
        {
          tier: pending.tierUsed,
          localCoherence: pending.localCoherence.toFixed(2),
          apiCoherence: apiCoherence.toFixed(2),
          delta: qualityDelta.toFixed(2),
          needed: escalationNeeded,
        },
        'Shadow evaluation complete'
      );
    } finally {
      this.activeShadows--;
      // Prompt reference goes out of scope here — no persistence of raw text
    }
  }

  /**
   * Dream shadow: evaluate a specific query during idle time.
   * Used by the dream consolidator to shadow queries that weren't
   * sampled during real-time operation.
   *
   * Unlike maybeShadow, this is awaitable and bypasses the sample rate.
   */
  async shadowDirect(pending: PendingShadow): Promise<void> {
    if (!this.client) {
      throw new Error('Shadow evaluator not configured (no API key)');
    }
    await this.runShadow(pending);
  }

  /**
   * Coherence estimation — mirrors router.ts estimateCoherence.
   * Duplicated here intentionally to avoid circular dependency on Router.
   */
  private estimateCoherence(text: string): number {
    if (!text || text.length === 0) return 0;
    const words = text.split(/\s+/).filter(Boolean);
    if (words.length === 0) return 0;
    const hasSentences = /[.!?]/.test(text) ? 0.3 : 0;
    const uniqueRatio = new Set(words.map(w => w.toLowerCase())).size / words.length;
    const diversityScore = Math.min(0.4, uniqueRatio * 0.5);
    const lengthScore = Math.min(0.3, words.length / 100);
    return Math.min(1, hasSentences + diversityScore + lengthScore);
  }

  /** Check if shadow evaluation is currently active */
  isActive(): boolean {
    return this.enabled;
  }

  /** Get current shadow evaluation stats */
  getStats(): {
    enabled: boolean;
    queryCounter: number;
    activeShadows: number;
    sampleRate: number;
  } {
    return {
      enabled: this.enabled,
      queryCounter: this.queryCounter,
      activeShadows: this.activeShadows,
      sampleRate: this.config.sampleRate,
    };
  }
}
