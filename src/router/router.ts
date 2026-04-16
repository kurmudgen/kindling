import pino from 'pino';
import { getConfig } from '../config/config.js';
import { scoreValence } from './valence.js';
import { ConfidenceAggregator } from './confidence.js';
import { ConcurrencyGuard } from './concurrency.js';
import { SpeculativeBuffer } from '../buffer/speculative.js';
import { Tier1 } from '../tiers/tier1.js';
import { Tier2 } from '../tiers/tier2.js';
import { Tier3 } from '../tiers/tier3.js';
import { logEscalation, logRecoveryEvent, hashPrompt } from '../sleep/logger.js';
import { MetaConfidenceModel } from '../meta/meta-confidence.js';
import { runWithRecovery, circuitBreaker } from '../recovery/recovery.js';
import type { RecoveryEvent } from '../recovery/recovery.js';
import { recordVcrEntry, isRecordingActive } from '../tools/vcr.js';
import { ShadowEvaluator } from '../shadow/shadow.js';
import type { Tier, TierQuery, TierResponse, ValenceScore } from '../tiers/tier-interface.js';

const log = pino({ level: process.env.KINDLING_LOG_LEVEL ?? 'info' });

export interface QueryResult {
  text: string;
  tier: 1 | 2 | 3;
  escalated: boolean;
  escalationPath: number[];
  latencyMs: number;
  confidence: number;
  metaAction?: 'confirm' | 'suppress' | 'force';
  recoveryEvents?: RecoveryEvent[];
}

export class Router {
  private tiers: Map<number, Tier> = new Map();
  // meta-confidence is intentionally shared — it learns across queries
  private meta: MetaConfidenceModel;
  // concurrency guard tracks in-flight queries
  private guard: ConcurrencyGuard;
  // shadow evaluator — API ground-truth teacher (Phase 4)
  private shadow: ShadowEvaluator;
  private initialized = false;

  constructor() {
    this.meta = new MetaConfidenceModel();
    this.guard = new ConcurrencyGuard();
    this.shadow = new ShadowEvaluator();
  }

  /** Get current in-flight query count (for diagnostics) */
  inFlightCount(): number {
    return this.guard.inFlightCount();
  }

  /** Get shadow evaluator stats (for diagnostics / REPL display) */
  getShadowStats() {
    return this.shadow.getStats();
  }

  /** Get the shadow evaluator instance (for dream task direct shadow) */
  getShadowEvaluator(): ShadowEvaluator {
    return this.shadow;
  }

  async init(): Promise<void> {
    if (this.initialized) return;

    const tier1 = new Tier1();
    this.tiers.set(1, tier1);

    try {
      const tier2 = new Tier2();
      this.tiers.set(2, tier2);
    } catch (err) {
      log.warn({ err }, 'Tier 2 initialization failed, will rely on Tier 1 + Tier 3');
    }

    try {
      const tier3 = new Tier3();
      this.tiers.set(3, tier3);
    } catch (err) {
      log.warn({ err }, 'Tier 3 initialization failed (no API key?), API escalation unavailable');
    }

    // Warm up available tiers
    for (const [id, tier] of this.tiers) {
      try {
        if (await tier.isAvailable()) {
          await tier.warmup();
          log.info({ tier: id, name: tier.name }, 'Tier warmed up');
        } else {
          log.warn({ tier: id }, 'Tier not available');
        }
      } catch (err) {
        log.warn({ tier: id, err }, 'Tier warmup failed');
      }
    }

    this.initialized = true;
  }

  /**
   * Route a query through the tier system and return the result.
   * Returns a QueryResult with the response text, tier used, and routing metadata.
   */
  async queryDetailed(prompt: string, context: string[] = []): Promise<QueryResult> {
    if (!this.initialized) await this.init();

    // Acquire a generation number for this query — isolates concurrent queries
    const generation = this.guard.acquire();
    const startTime = performance.now();

    // Per-query state — no shared confidence or buffer
    const confidence = new ConfidenceAggregator();
    const buffer = new SpeculativeBuffer();
    void buffer; // buffer reserved for future per-token routing

    try {
      return await this.queryInternal(prompt, context, startTime, confidence, generation);
    } finally {
      this.guard.release(generation);
    }
  }

  private async queryInternal(
    prompt: string,
    context: string[],
    startTime: number,
    confidence: ConfidenceAggregator,
    _generation: number
  ): Promise<QueryResult> {
    const valence = scoreValence(prompt);

    log.debug({ valence }, 'Query valence scored');

    // PRIORITY 1.2: Valence-based pre-escalation gate
    // High-valence queries skip Tier 1 entirely
    let startTier = this.determineStartTier(valence);
    let currentTierId = startTier;
    const escalationPath: number[] = [currentTierId];

    const tierQuery: TierQuery = {
      prompt,
      context,
      maxTokens: 1024,
      valenceScore: valence,
    };

    // Generate from starting tier through the recovery cascade
    const recoveryEvents: RecoveryEvent[] = [];
    const firstGen = await this.generateFromTierWithRecovery(currentTierId, tierQuery);
    let response = firstGen.response;
    recoveryEvents.push(...firstGen.events);

    if (!response) {
      throw new Error('No tier available to handle query');
    }

    // If recovery downgraded the tier, reflect that in the path
    if (response.tier !== currentTierId) {
      escalationPath.push(response.tier);
      currentTierId = response.tier as 1 | 2 | 3;
    }

    // Check escalation signals from the response
    const decision = confidence.decide(
      currentTierId,
      response.escalationSignals,
      valence
    );

    // Meta-confidence correction layer
    const metaDecision = this.meta.evaluate(decision, response.escalationSignals, valence, currentTierId);
    let metaAction = metaDecision.action;

    let shouldEscalate = decision.shouldEscalate;
    if (metaAction === 'suppress' && shouldEscalate) {
      shouldEscalate = false;
      log.info({ reason: metaDecision.reason }, 'Meta-confidence suppressed escalation');
    } else if (metaAction === 'force' && !shouldEscalate && currentTierId < 3) {
      shouldEscalate = true;
      log.info({ reason: metaDecision.reason }, 'Meta-confidence forced escalation');
    }

    let escalated = false;

    if (shouldEscalate && decision.targetTier !== currentTierId) {
      escalated = true;
      const previousTier = currentTierId;
      currentTierId = decision.targetTier;
      escalationPath.push(currentTierId);

      log.info(
        {
          from: previousTier,
          to: currentTierId,
          reason: decision.reason,
          confidence: decision.confidence.toFixed(3),
          signals: response.escalationSignals,
        },
        'Escalating'
      );

      logEscalation({
        queryHash: hashPrompt(prompt),
        valenceScore: valence,
        tier1ConfidenceAtHandoff: decision.confidence,
        escalatedToTier: currentTierId,
        tokensBeforeEscalation: response.tokens.length,
        handoffSuccessful: true,
        totalLatencyMs: performance.now() - startTime,
      });

      // Generate from higher tier through recovery cascade
      const escalatedGen = await this.generateFromTierWithRecovery(currentTierId, tierQuery);
      recoveryEvents.push(...escalatedGen.events);
      if (escalatedGen.response) {
        response = escalatedGen.response;
      } else {
        // Escalation target unavailable, keep lower tier output
        currentTierId = previousTier;
        escalationPath.push(previousTier);
        log.warn('Escalation target unavailable, keeping lower tier output');
      }
    }

    const totalLatencyMs = performance.now() - startTime;

    // Record outcome for meta-confidence learning
    const text = response.tokens.join(' ');
    const coherence = this.estimateCoherence(text);
    this.meta.recordOutcome(
      response.escalationSignals,
      valence,
      escalated ? 'escalate' : 'stay',
      metaAction,
      startTier,
      response.tokens.length,
      coherence
    );

    log.info(
      {
        tier: currentTierId,
        confidence: response.confidence.toFixed(3),
        latencyMs: Math.round(totalLatencyMs),
        tokens: response.tokens.length,
        escalated,
        metaAction,
        signals: response.escalationSignals,
      },
      'Query complete'
    );

    // VCR recording — privacy-safe (hash only, no raw prompt)
    if (isRecordingActive()) {
      recordVcrEntry({
        timestamp: new Date().toISOString(),
        queryHash: hashPrompt(prompt),
        valence,
        signals: response.escalationSignals,
        startTier,
        finalTier: currentTierId,
        confidence: response.confidence,
        routerDecision: decision.shouldEscalate ? 'escalate' : 'stay',
        metaAction,
        escalated,
        responseTokens: response.tokens.length,
        recoveryEventCount: recoveryEvents.length,
      });
    }

    // Shadow evaluation — API ground-truth comparison (Phase 4)
    // Fire-and-forget: raw prompt held in memory only for the API call,
    // never persisted to disk. Training store records hash only.
    this.shadow.maybeShadow({
      prompt,
      context,
      signals: response.escalationSignals,
      valence,
      tierUsed: startTier,
      routerDecision: escalated ? 'escalate' : 'stay',
      metaAction,
      localTokenCount: response.tokens.length,
      localCoherence: coherence,
      localLatencyMs: totalLatencyMs,
    });

    return {
      text,
      tier: currentTierId as 1 | 2 | 3,
      escalated,
      escalationPath,
      latencyMs: totalLatencyMs,
      confidence: response.confidence,
      metaAction,
      recoveryEvents: recoveryEvents.length > 0 ? recoveryEvents : undefined,
    };
  }

  /** Convenience method — returns just the text */
  async query(prompt: string, context: string[] = []): Promise<string> {
    const result = await this.queryDetailed(prompt, context);
    return result.text;
  }

  /**
   * Determine which tier to start at based on valence score.
   *
   * High-complexity or high-stakes queries skip straight to Tier 2 or 3,
   * avoiding the latency of generating a full Tier 1 response that will
   * inevitably be thrown away.
   */
  private determineStartTier(valence: ValenceScore): 1 | 2 | 3 {
    // Critical: high stakes + high complexity → Tier 3
    if (valence.composite > 0.50 && valence.stakes > 0.5 && this.tiers.has(3)) {
      log.info(
        { composite: valence.composite.toFixed(2), stakes: valence.stakes.toFixed(2) },
        'Critical valence, starting at Tier 3'
      );
      return 3;
    }

    // High complexity or multi-part queries → Tier 2
    if (valence.composite > 0.35 && this.tiers.has(2)) {
      log.info(
        { composite: valence.composite.toFixed(2), complexity: valence.complexity.toFixed(2) },
        'High valence, starting at Tier 2'
      );
      return 2;
    }

    return 1;
  }

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

  /**
   * Generate from a tier, running through the 5-layer recovery cascade on failure.
   * Returns both the response and any recovery events that occurred.
   */
  private async generateFromTierWithRecovery(
    tierId: 1 | 2 | 3,
    query: TierQuery
  ): Promise<{ response: TierResponse | null; events: RecoveryEvent[] }> {
    const allEvents: RecoveryEvent[] = [];
    const tier = this.tiers.get(tierId);

    if (!tier || !(await tier.isAvailable())) {
      // Tier not available — try lower tier
      if (tierId > 1) {
        return this.generateFromTierWithRecovery((tierId - 1) as 1 | 2 | 3, query);
      }
      return { response: null, events: allEvents };
    }

    const config = getConfig();
    const tierConfig = this.getTierConfig(tierId, config);

    const ctx = {
      tierId,
      query,
      primaryGenerate: () => tier.generate(query),
      fallbackModelGenerate:
        tier.generateWithModel && tierConfig.fallbackModel
          ? () => tier.generateWithModel!(query, tierConfig.fallbackModel!)
          : undefined,
      apiFallbackGenerate:
        tier.generateWithApi && tierConfig.apiFallbackModel && process.env.ANTHROPIC_API_KEY
          ? () => tier.generateWithApi!(query)
          : undefined,
      tierDowngradeGenerate:
        tierId > 1
          ? async () => {
              const downResult = await this.generateFromTierWithRecovery(
                (tierId - 1) as 1 | 2 | 3,
                query
              );
              allEvents.push(...downResult.events);
              if (!downResult.response) {
                throw new Error(`Tier ${tierId - 1} downgrade also failed`);
              }
              return downResult.response;
            }
          : undefined,
    };

    const result = await runWithRecovery(ctx);
    allEvents.push(...result.events);

    // Log all recovery events to the escalation log for analyst review
    for (const ev of result.events) {
      logRecoveryEvent(ev);
    }

    return { response: result.response, events: allEvents };
  }

  private getTierConfig(
    tierId: 1 | 2 | 3,
    config: ReturnType<typeof getConfig>
  ): { fallbackModel?: string; apiFallbackModel?: string } {
    if (tierId === 1) return config.tier1;
    if (tierId === 2) return config.tier2;
    return config.tier3;
  }
}

// Convenience function matching the spec
const _router = new Router();

export async function query(prompt: string, context?: string[]): Promise<string> {
  return _router.query(prompt, context);
}
