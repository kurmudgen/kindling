import { Ollama } from 'ollama';
import Anthropic from '@anthropic-ai/sdk';
import { getConfig, getOllamaHost } from '../config/config.js';
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
    try {
      const models = await this.ollamaClient.list();
      const hasModel = models.models.some(m => m.name === this.model || m.name.startsWith(this.model + '-'));
      if (hasModel) return true;
    } catch {
      // Ollama not available
    }
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
    const ollamaAvailable = await this.isOllamaModelAvailable();

    if (ollamaAvailable) {
      return this.generateOllama(query);
    }

    if (this.anthropicClient) {
      return this.generateAPI(query);
    }

    throw new Error('Tier 2: No backend available (Ollama model missing, no API key)');
  }

  private async isOllamaModelAvailable(): Promise<boolean> {
    try {
      const models = await this.ollamaClient.list();
      return models.models.some(m => m.name === this.model || m.name.startsWith(this.model + '-'));
    } catch {
      return false;
    }
  }

  private async generateOllama(query: TierQuery): Promise<TierResponse> {
    const start = performance.now();
    const systemPrompt = query.context.length > 0
      ? `Previous context:\n${query.context.join('\n')}`
      : '';

    const response = await this.ollamaClient.generate({
      model: this.model,
      system: systemPrompt || undefined,
      prompt: query.prompt,
      options: {
        num_predict: query.maxTokens,
        num_thread: getConfig().tier2.maxCores,
      },
    });

    const latencyMs = performance.now() - start;
    const tokens = response.response.split(/\s+/).filter(Boolean);
    const escalationSignals = this.estimateSignals(tokens, response);
    const confidence = this.computeConfidence(escalationSignals);

    return {
      tokens,
      confidence,
      tier: 2,
      escalationSignals,
      latencyMs,
      metadata: { source: 'ollama', model: this.model },
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

    const escalationSignals: EscalationSignals = {
      tokenProbabilitySpread: 0.1, // API models generally confident
      semanticVelocity: this.estimateSemanticVelocity(tokens),
      surpriseScore: 0.05,
      attentionAnomalyScore: 0,
    };
    const confidence = this.computeConfidence(escalationSignals);

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

  private estimateSignals(
    tokens: string[],
    response: { eval_count?: number; eval_duration?: number }
  ): EscalationSignals {
    const evalCount = response.eval_count ?? tokens.length;
    const evalDuration = response.eval_duration ?? 1;
    const tps = evalCount / (evalDuration / 1e9);
    const expectedTps = 15; // medium model is slower
    const speedRatio = tps / expectedTps;

    return {
      tokenProbabilitySpread: Math.max(0, Math.min(1, 1 - speedRatio * 0.5)),
      semanticVelocity: this.estimateSemanticVelocity(tokens),
      surpriseScore: 0.1,
      attentionAnomalyScore: this.computeRepetitionRate(tokens),
    };
  }

  private estimateSemanticVelocity(tokens: string[]): number {
    if (tokens.length === 0) return 0;
    const unique = new Set(tokens.map(t => t.toLowerCase()));
    return unique.size / tokens.length;
  }

  private computeRepetitionRate(tokens: string[]): number {
    if (tokens.length < 4) return 0;
    let reps = 0;
    for (let i = 2; i < tokens.length; i++) {
      if (tokens[i] === tokens[i - 2]) reps++;
    }
    return reps / (tokens.length - 2);
  }

  private computeConfidence(signals: EscalationSignals): number {
    const config = getConfig().escalation;
    const weighted =
      signals.tokenProbabilitySpread * config.tokenProbabilitySpreadWeight +
      signals.semanticVelocity * config.semanticVelocityWeight +
      signals.surpriseScore * config.surpriseScoreWeight +
      signals.attentionAnomalyScore * config.attentionAnomalyWeight;
    return Math.max(0, Math.min(1, 1 - weighted));
  }
}
