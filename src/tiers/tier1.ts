import { Ollama } from 'ollama';
import { getConfig, getOllamaHost } from '../config/config.js';
import {
  computeSignalsFromLogprobs,
  computeSignalsFromHeuristics,
  computeConfidence,
} from './signals.js';
import type { TokenLogprobData } from './signals.js';
import type { Tier, TierQuery, TierResponse } from './tier-interface.js';

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
      logprobs: true,
      top_logprobs: 5,
    });

    const latencyMs = performance.now() - start;
    const tokens = response.response.split(/\s+/).filter(Boolean);

    // Use real logprobs if available, fall back to heuristics
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
        30 // expected TPS for small model
      );
    }

    const confidence = computeConfidence(escalationSignals);

    return {
      tokens,
      confidence,
      tier: 1,
      escalationSignals,
      latencyMs,
      metadata: {
        source: 'ollama',
        model: this.model,
        hasLogprobs: !!(response.logprobs && response.logprobs.length > 0),
        logprobCount: response.logprobs?.length ?? 0,
      },
    };
  }
}
