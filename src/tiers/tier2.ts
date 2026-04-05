import { Ollama } from 'ollama';
import Anthropic from '@anthropic-ai/sdk';
import { getConfig, getOllamaHost } from '../config/config.js';
import {
  computeSignalsFromLogprobs,
  computeSignalsFromHeuristics,
  computeConfidence,
} from './signals.js';
import type { TokenLogprobData } from './signals.js';
import type { Tier, TierQuery, TierResponse, EscalationSignals } from './tier-interface.js';

export class Tier2 implements Tier {
  readonly id = 2 as const;
  readonly name = 'Medium (Ollama + API Fallback)';
  private ollamaClient: Ollama;
  private anthropicClient: Anthropic | null = null;
  private model: string;
  private apiFallback: boolean;

  constructor() {
    const config = getConfig();
    this.ollamaClient = new Ollama({ host: getOllamaHost() });
    this.model = config.tier2.model;
    this.apiFallback = config.tier2.apiFallback;

    if (this.apiFallback && process.env.ANTHROPIC_API_KEY) {
      this.anthropicClient = new Anthropic();
    }
  }

  async isAvailable(): Promise<boolean> {
    if (await this.isOllamaModelAvailable()) return true;
    return this.apiFallback && this.anthropicClient !== null;
  }

  async warmup(): Promise<void> {
    try {
      await this.ollamaClient.generate({ model: this.model, prompt: '', keep_alive: '10m' });
    } catch {
      if (!this.apiFallback || !this.anthropicClient) {
        throw new Error(
          `Tier 2 model "${this.model}" unavailable and no API fallback configured. ` +
          `Run: ollama pull ${this.model}`
        );
      }
    }
  }

  async generate(query: TierQuery): Promise<TierResponse> {
    // Recovery layer handles fallbacks — this method just tries the primary model.
    // It throws on failure so the recovery cascade can take over.
    return this.generateOllama(query, this.model);
  }

  async generateWithModel(query: TierQuery, modelName: string): Promise<TierResponse> {
    return this.generateOllama(query, modelName);
  }

  async generateWithApi(query: TierQuery): Promise<TierResponse> {
    if (!this.anthropicClient) {
      throw new Error('Tier 2: API fallback not configured (no ANTHROPIC_API_KEY)');
    }
    return this.generateAPI(query);
  }

  private async isOllamaModelAvailable(): Promise<boolean> {
    try {
      const models = await this.ollamaClient.list();
      return models.models.some(m =>
        m.name === this.model || m.name.startsWith(this.model + '-')
      );
    } catch {
      return false;
    }
  }

  private async generateOllama(query: TierQuery, modelName: string): Promise<TierResponse> {
    const start = performance.now();
    const systemPrompt = query.context.length > 0
      ? `Previous context:\n${query.context.join('\n')}`
      : '';

    const response = await this.ollamaClient.generate({
      model: modelName,
      system: systemPrompt || undefined,
      prompt: query.prompt,
      options: {
        num_predict: query.maxTokens,
        num_thread: getConfig().tier2.maxCores,
      },
      logprobs: true,
      top_logprobs: 5,
    });

    const latencyMs = performance.now() - start;
    const tokens = response.response.split(/\s+/).filter(Boolean);

    let escalationSignals;
    if (response.logprobs && response.logprobs.length > 0) {
      const logprobData: TokenLogprobData[] = response.logprobs.map(lp => ({
        token: lp.token,
        logprob: lp.logprob,
        topLogprobs: lp.top_logprobs?.map(t => ({
          token: t.token,
          logprob: t.logprob,
        })),
      }));
      escalationSignals = computeSignalsFromLogprobs(logprobData, tokens);
    } else {
      escalationSignals = computeSignalsFromHeuristics(
        tokens,
        response.eval_count,
        response.eval_duration,
        15 // expected TPS for medium model
      );
    }

    const confidence = computeConfidence(escalationSignals);

    return {
      tokens,
      confidence,
      tier: 2,
      escalationSignals,
      latencyMs,
      metadata: {
        source: 'ollama',
        model: modelName,
        hasLogprobs: !!(response.logprobs && response.logprobs.length > 0),
      },
    };
  }

  private async generateAPI(query: TierQuery): Promise<TierResponse> {
    const start = performance.now();
    const messages: Anthropic.MessageParam[] = [];

    if (query.context.length > 0) {
      messages.push({ role: 'user', content: query.context.join('\n') });
      messages.push({ role: 'assistant', content: 'Understood. I have the context.' });
    }
    messages.push({ role: 'user', content: query.prompt });

    const response = await this.anthropicClient!.messages.create({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: query.maxTokens,
      messages,
    });

    const latencyMs = performance.now() - start;
    const text = response.content
      .filter((b): b is Anthropic.TextBlock => b.type === 'text')
      .map(b => b.text)
      .join('');
    const tokens = text.split(/\s+/).filter(Boolean);

    // API tiers don't expose logprobs — use conservative heuristic signals
    const escalationSignals: EscalationSignals = {
      tokenProbabilitySpread: 0.1,
      semanticVelocity: 0.1,
      surpriseScore: 0.05,
      attentionAnomalyScore: 0,
    };
    const confidence = computeConfidence(escalationSignals);

    return {
      tokens,
      confidence,
      tier: 2,
      escalationSignals,
      latencyMs,
      metadata: {
        source: 'api',
        model: 'claude-haiku-4-5-20251001',
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
      },
    };
  }
}
