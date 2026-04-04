// TODO: Phase 2 — Replace API stand-in with local streaming inference.
// This tier currently uses the Anthropic API as a placeholder.
// Phase 2 will implement direct local model streaming with speculative decoding.

import Anthropic from '@anthropic-ai/sdk';
import { getConfig } from '../config/config.js';
import type { Tier, TierQuery, TierResponse, EscalationSignals } from './tier-interface.js';

// Rough per-token cost estimates (USD) for cost tracking
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
  readonly name = 'Deep (API Stand-in)';
  private client: Anthropic;
  private model: string;

  constructor() {
    if (!process.env.ANTHROPIC_API_KEY) {
      throw new Error('Tier 3 requires ANTHROPIC_API_KEY environment variable');
    }
    this.client = new Anthropic();
    this.model = getConfig().tier3.model;
  }

  async isAvailable(): Promise<boolean> {
    return !!process.env.ANTHROPIC_API_KEY;
  }

  async warmup(): Promise<void> {
    // API tier has no warmup — connection pooling handled by SDK
  }

  async generate(query: TierQuery): Promise<TierResponse> {
    const start = performance.now();
    const messages: Anthropic.MessageParam[] = [];

    if (query.context.length > 0) {
      messages.push({ role: 'user', content: query.context.join('\n') });
      messages.push({ role: 'assistant', content: 'Understood.' });
    }
    messages.push({ role: 'user', content: query.prompt });

    const response = await this.client.messages.create({
      model: this.model,
      max_tokens: query.maxTokens,
      messages,
    });

    const latencyMs = performance.now() - start;
    const text = response.content
      .filter((b): b is Anthropic.TextBlock => b.type === 'text')
      .map(b => b.text)
      .join('');
    const tokens = text.split(/\s+/).filter(Boolean);

    const inputCostRate = COST_PER_INPUT_TOKEN[this.model] ?? COST_PER_INPUT_TOKEN['claude-sonnet-4-6'];
    const outputCostRate = COST_PER_OUTPUT_TOKEN[this.model] ?? COST_PER_OUTPUT_TOKEN['claude-sonnet-4-6'];
    const costEstimate =
      response.usage.input_tokens * inputCostRate +
      response.usage.output_tokens * outputCostRate;

    const escalationSignals: EscalationSignals = {
      tokenProbabilitySpread: 0.05,
      semanticVelocity: this.estimateSemanticVelocity(tokens),
      surpriseScore: 0.02,
      attentionAnomalyScore: 0,
    };

    return {
      tokens,
      confidence: 0.95, // Deep tier is assumed highly confident
      tier: 3,
      escalationSignals,
      latencyMs,
      metadata: {
        source: 'api',
        model: this.model,
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
        costEstimateUSD: costEstimate,
      },
    };
  }

  private estimateSemanticVelocity(tokens: string[]): number {
    if (tokens.length === 0) return 0;
    const unique = new Set(tokens.map(t => t.toLowerCase()));
    return unique.size / tokens.length;
  }
}
