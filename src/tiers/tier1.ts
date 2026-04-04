import { Ollama } from 'ollama';
import { getConfig, getOllamaHost } from '../config/config.js';
import type { Tier, TierQuery, TierResponse, EscalationSignals } from './tier-interface.js';

export class Tier1 implements Tier {
  readonly id = 1 as const;
  readonly name = 'Shallow (Ollama Local)';
  private client: Ollama;
  private model: string;

  constructor() {
    this.client = new Ollama({ host: getOllamaHost() });
    this.model = getConfig().tier1.model;
  }

  async isAvailable(): Promise<boolean> {
    try {
      await this.client.list();
      return true;
    } catch {
      return false;
    }
  }

  async warmup(): Promise<void> {
    if (!(await this.isAvailable())) {
      throw new Error(`Ollama not available at ${getOllamaHost()}. Is Ollama running?`);
    }
    // Load model into memory
    try {
      await this.client.generate({ model: this.model, prompt: '', keep_alive: '10m' });
    } catch {
      throw new Error(
        `Failed to warm up model "${this.model}". Is it pulled? Run: ollama pull ${this.model}`
      );
    }
  }

  async generate(query: TierQuery): Promise<TierResponse> {
    const start = performance.now();

    const systemPrompt = query.context.length > 0
      ? `Previous context:\n${query.context.join('\n')}`
      : '';

    const response = await this.client.generate({
      model: this.model,
      system: systemPrompt || undefined,
      prompt: query.prompt,
      options: {
        num_predict: query.maxTokens,
        num_thread: getConfig().tier1.maxCores,
      },
    });

    const latencyMs = performance.now() - start;
    const tokens = response.response.split(/\s+/).filter(Boolean);
    const escalationSignals = this.estimateEscalationSignals(response, tokens);
    const confidence = this.computeConfidence(escalationSignals);

    return {
      tokens,
      confidence,
      tier: 1,
      escalationSignals,
      latencyMs,
    };
  }

  private estimateEscalationSignals(
    response: { eval_count?: number; eval_duration?: number; total_duration?: number },
    tokens: string[]
  ): EscalationSignals {
    // Without direct logprobs from Ollama, we estimate signals from available metadata
    const evalCount = response.eval_count ?? tokens.length;
    const evalDuration = response.eval_duration ?? 1;
    const tokensPerSecond = evalCount / (evalDuration / 1e9);

    // Token probability spread: estimate from generation speed
    // Slower generation can indicate more uncertainty (more sampling retries)
    const expectedTps = 30; // baseline for small model
    const speedRatio = tokensPerSecond / expectedTps;
    const tokenProbabilitySpread = Math.max(0, Math.min(1, 1 - speedRatio * 0.5));

    // Semantic velocity: measure vocabulary diversity as a proxy
    const uniqueTokens = new Set(tokens.map(t => t.toLowerCase()));
    const semanticVelocity = tokens.length > 0 ? uniqueTokens.size / tokens.length : 0;

    // Surprise score: estimate from response length vs expected
    const lengthRatio = tokens.length / Math.max(1, (response.eval_count ?? tokens.length));
    const surpriseScore = Math.abs(1 - lengthRatio);

    // Attention anomaly: high repetition suggests attention issues
    const repetitionRate = this.computeRepetitionRate(tokens);
    const attentionAnomalyScore = repetitionRate;

    return {
      tokenProbabilitySpread,
      semanticVelocity,
      surpriseScore,
      attentionAnomalyScore,
    };
  }

  private computeRepetitionRate(tokens: string[]): number {
    if (tokens.length < 4) return 0;
    let repetitions = 0;
    for (let i = 2; i < tokens.length; i++) {
      if (tokens[i] === tokens[i - 2]) repetitions++;
    }
    return repetitions / (tokens.length - 2);
  }

  private computeConfidence(signals: EscalationSignals): number {
    const config = getConfig().escalation;
    const weighted =
      signals.tokenProbabilitySpread * config.tokenProbabilitySpreadWeight +
      signals.semanticVelocity * config.semanticVelocityWeight +
      signals.surpriseScore * config.surpriseScoreWeight +
      signals.attentionAnomalyScore * config.attentionAnomalyWeight;

    // Confidence is inverse of escalation signal strength
    return Math.max(0, Math.min(1, 1 - weighted));
  }
}
