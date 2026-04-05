/**
 * Tier 3 — Deep tier router.
 *
 * Prefers local NVMe-streamed model (tier3-local.ts) when available.
 * Falls back to Anthropic API when local model is not pulled.
 */

import Anthropic from '@anthropic-ai/sdk';
import { getConfig } from '../config/config.js';
import { Tier3Local } from './tier3-local.js';
import { computeConfidence } from './signals.js';
import type { Tier, TierQuery, TierResponse, EscalationSignals } from './tier-interface.js';

const COST_PER_INPUT_TOKEN: Record<string, number> = {
  'claude-sonnet-4-6': 3.0 / 1_000_000,
  'claude-opus-4-6': 15.0 / 1_000_000,
};
const COST_PER_OUTPUT_TOKEN: Record<string, number> = {
  'claude-sonnet-4-6': 15.0 / 1_000_000,
  'claude-opus-4-6': 75.0 / 1_000_000,
};

export class Tier3 implements Tier {
  readonly id = 3 as const;
  readonly name = 'Deep (Local + API Fallback)';
  private localTier: Tier3Local | null = null;
  private apiClient: Anthropic | null = null;
  private apiModel: string;

  constructor() {
    const config = getConfig();
    this.apiModel = config.tier3.model;

    // Try to initialize local tier
    try {
      this.localTier = new Tier3Local();
    } catch {
      // Local tier not configured
    }

    // Initialize API fallback if key available
    if (process.env.ANTHROPIC_API_KEY) {
      this.apiClient = new Anthropic();
    }

    if (!this.localTier && !this.apiClient) {
      throw new Error(
        'Tier 3 requires either a local model or ANTHROPIC_API_KEY environment variable'
      );
    }
  }

  async isAvailable(): Promise<boolean> {
    if (this.localTier && (await this.localTier.isAvailable())) return true;
    return !!this.apiClient;
  }

  async warmup(): Promise<void> {
    if (this.localTier) {
      try {
        await this.localTier.warmup();
        return;
      } catch {
        // Local warmup failed, fall through to API
      }
    }
    // API tier has no warmup
  }

  async generate(query: TierQuery): Promise<TierResponse> {
    // Prefer local
    if (this.localTier && (await this.localTier.isAvailable())) {
      try {
        return await this.localTier.generate(query);
      } catch {
        // Fall through to API
      }
    }

    if (this.apiClient) {
      return this.generateAPI(query);
    }

    throw new Error('Tier 3: No backend available');
  }

  private async generateAPI(query: TierQuery): Promise<TierResponse> {
    const start = performance.now();
    const messages: Anthropic.MessageParam[] = [];

    if (query.context.length > 0) {
      messages.push({ role: 'user', content: query.context.join('\n') });
      messages.push({ role: 'assistant', content: 'Understood.' });
    }
    messages.push({ role: 'user', content: query.prompt });

    const response = await this.apiClient!.messages.create({
      model: this.apiModel,
      max_tokens: query.maxTokens,
      messages,
    });

    const latencyMs = performance.now() - start;
    const text = response.content
      .filter((b): b is Anthropic.TextBlock => b.type === 'text')
      .map(b => b.text)
      .join('');
    const tokens = text.split(/\s+/).filter(Boolean);

    const inputCostRate = COST_PER_INPUT_TOKEN[this.apiModel] ?? COST_PER_INPUT_TOKEN['claude-sonnet-4-6'];
    const outputCostRate = COST_PER_OUTPUT_TOKEN[this.apiModel] ?? COST_PER_OUTPUT_TOKEN['claude-sonnet-4-6'];
    const costEstimate =
      response.usage.input_tokens * inputCostRate +
      response.usage.output_tokens * outputCostRate;

    const escalationSignals: EscalationSignals = {
      tokenProbabilitySpread: 0.05,
      semanticVelocity: 0.05,
      surpriseScore: 0.02,
      attentionAnomalyScore: 0,
    };

    return {
      tokens,
      confidence: 0.95,
      tier: 3,
      escalationSignals,
      latencyMs,
      metadata: {
        source: 'api',
        model: this.apiModel,
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
        costEstimateUSD: costEstimate,
      },
    };
  }
}
