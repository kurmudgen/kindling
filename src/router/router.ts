import pino from 'pino';
import { getConfig } from '../config/config.js';
import { scoreValence } from './valence.js';
import { ConfidenceAggregator } from './confidence.js';
import { SpeculativeBuffer } from '../buffer/speculative.js';
import { Tier1 } from '../tiers/tier1.js';
import { Tier2 } from '../tiers/tier2.js';
import { Tier3 } from '../tiers/tier3.js';
import { logEscalation, hashPrompt } from '../sleep/logger.js';
import type { Tier, TierQuery, TierResponse, ValenceScore } from '../tiers/tier-interface.js';

const log = pino({ level: process.env.KINDLING_LOG_LEVEL ?? 'info' });

export class Router {
  private tiers: Map<number, Tier> = new Map();
  private confidence: ConfidenceAggregator;
  private buffer: SpeculativeBuffer;
  private initialized = false;

  constructor() {
    this.confidence = new ConfidenceAggregator();
    this.buffer = new SpeculativeBuffer();
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

  async query(prompt: string, context: string[] = []): Promise<string> {
    if (!this.initialized) await this.init();

    const startTime = performance.now();
    const valence = scoreValence(prompt);
    this.buffer.reset();
    this.confidence.reset();

    log.debug({ valence }, 'Query valence scored');

    // Determine starting tier based on valence
    let currentTierId: 1 | 2 | 3 = 1;
    if (valence.composite > 0.7 && this.tiers.has(2)) {
      currentTierId = 2;
      log.info('High valence, starting at Tier 2');
    }

    const tierQuery: TierQuery = {
      prompt,
      context,
      maxTokens: 1024,
      valenceScore: valence,
    };

    // Phase 1 simplified flow: generate from starting tier, check confidence,
    // escalate if needed. Full speculative parallel execution in Phase 2.
    let response = await this.generateFromTier(currentTierId, tierQuery);

    if (!response) {
      throw new Error('No tier available to handle query');
    }

    // Populate buffer with Tier 1 tokens
    for (const token of response.tokens) {
      this.buffer.push(token);
      if (this.buffer.isFull()) {
        // Check if we should escalate
        const decision = this.confidence.decide(
          currentTierId,
          response.escalationSignals,
          valence
        );

        if (decision.shouldEscalate && decision.targetTier !== currentTierId) {
          const { rejectedTokens } = this.buffer.reject();
          log.info(
            { from: currentTierId, to: decision.targetTier, reason: decision.reason },
            'Escalating'
          );

          logEscalation({
            queryHash: hashPrompt(prompt),
            valenceScore: valence,
            tier1ConfidenceAtHandoff: decision.confidence,
            escalatedToTier: decision.targetTier,
            tokensBeforeEscalation: this.buffer.getFlushedTokens().length + rejectedTokens.length,
            handoffSuccessful: true,
            totalLatencyMs: performance.now() - startTime,
          });

          // Generate from higher tier
          const escalatedResponse = await this.generateFromTier(decision.targetTier, tierQuery);
          if (escalatedResponse) {
            response = escalatedResponse;
            currentTierId = decision.targetTier;
            // Replace buffer with escalated response
            this.buffer.overrideFlushed(response.tokens);
          } else {
            // Escalation failed, keep what we have
            this.buffer.confirm();
            log.warn('Escalation target unavailable, keeping lower tier output');
          }
          break;
        } else {
          // Confident enough, flush buffer
          this.buffer.confirm();
        }
      }
    }

    // If buffer has unflushed tokens, flush them
    if (this.buffer.snapshot().state === 'filling') {
      this.buffer.confirm();
    }

    // If we never populated the buffer (e.g., started at tier 2+), set output directly
    if (this.buffer.getFlushedTokens().length === 0 && response.tokens.length > 0) {
      this.buffer.overrideFlushed(response.tokens);
    }

    const totalLatencyMs = performance.now() - startTime;
    log.info(
      {
        tier: currentTierId,
        confidence: response.confidence,
        latencyMs: Math.round(totalLatencyMs),
        tokens: this.buffer.getFlushedTokens().length,
      },
      'Query complete'
    );

    return this.buffer.getOutput();
  }

  private async generateFromTier(
    tierId: 1 | 2 | 3,
    query: TierQuery
  ): Promise<TierResponse | null> {
    // Try requested tier, fall through to next available
    for (let id = tierId; id <= 3; id++) {
      const tier = this.tiers.get(id);
      if (tier && (await tier.isAvailable())) {
        try {
          return await tier.generate(query);
        } catch (err) {
          log.warn({ tier: id, err }, 'Tier generation failed, trying next');
        }
      }
    }
    return null;
  }
}

// Convenience function matching the spec
const _router = new Router();

export async function query(prompt: string, context?: string[]): Promise<string> {
  return _router.query(prompt, context);
}
