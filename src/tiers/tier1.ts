import { Ollama, type GenerateResponse } from 'ollama';
import { getConfig, getOllamaHost } from '../config/config.js';
import {
  computeSignalsFromLogprobs,
  computeSignalsFromHeuristics,
  computeConfidence,
} from './signals.js';
import type { TokenLogprobData } from './signals.js';
import type { Tier, TierQuery, TierResponse, StreamChunk } from './tier-interface.js';

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
    return this.generateWithModel(query, this.model);
  }

  /**
   * Stream tokens one at a time from Tier 1 (Ollama streaming API).
   *
   * Each chunk carries the token text and, when available, per-token logprob data
   * so the router can evaluate escalation signals incrementally. The final chunk
   * has isFinal=true and a complete finalResponse for full signal computation.
   *
   * Used by router.queryStream() for per-token routing with speculative buffering.
   */
  async *generateStream(query: TierQuery): AsyncGenerator<StreamChunk> {
    const start = performance.now();

    const systemPrompt = query.context.length > 0
      ? `Previous context:\n${query.context.join('\n')}`
      : '';

    const stream = await this.client.generate({
      model: this.model,
      system: systemPrompt || undefined,
      prompt: query.prompt,
      options: {
        num_predict: query.maxTokens,
        num_thread: getConfig().tier1.maxCores,
      },
      logprobs: true,
      top_logprobs: 5,
      stream: true,
    });

    const accumulatedTokens: string[] = [];
    const accumulatedLogprobs: Array<{ token: string; logprob: number; topLogprobs?: Array<{ token: string; logprob: number }> }> = [];
    let lastChunkData: GenerateResponse | null = null;

    for await (const chunk of stream) {
      lastChunkData = chunk;

      // Ollama streaming: each chunk.response is the new partial text
      const rawToken = chunk.response;
      if (!rawToken) continue;

      // Split on whitespace but preserve the token text as-is for accumulation
      accumulatedTokens.push(rawToken);

      // Logprobs may be included per streaming chunk
      const chunkLogprob = (chunk as any).logprobs?.[0];
      const logprob: number | undefined = chunkLogprob?.logprob;
      const topLogprobs = chunkLogprob?.top_logprobs?.map((t: any) => ({
        token: t.token,
        logprob: t.logprob,
      }));

      if (logprob !== undefined) {
        accumulatedLogprobs.push({ token: rawToken, logprob, topLogprobs });
      }

      // Emit non-final chunk
      if (!chunk.done) {
        yield {
          token: rawToken,
          logprob,
          topLogprobs,
          isFinal: false,
        };
      }
    }

    // Build the final TierResponse from accumulated data
    const latencyMs = performance.now() - start;
    const tokens = accumulatedTokens.flatMap(t => t.split(/\s+/).filter(Boolean));

    let escalationSignals;
    if (accumulatedLogprobs.length > 0) {
      escalationSignals = computeSignalsFromLogprobs(accumulatedLogprobs, tokens);
    } else {
      escalationSignals = computeSignalsFromHeuristics(
        tokens,
        lastChunkData?.eval_count,
        lastChunkData?.eval_duration,
        30
      );
    }

    const confidence = computeConfidence(escalationSignals);
    const finalResponse: TierResponse = {
      tokens,
      confidence,
      tier: 1,
      escalationSignals,
      latencyMs,
      metadata: {
        source: 'ollama-stream',
        model: this.model,
        hasLogprobs: accumulatedLogprobs.length > 0,
        logprobCount: accumulatedLogprobs.length,
      },
    };

    yield {
      token: '',
      logprob: undefined,
      topLogprobs: undefined,
      isFinal: true,
      finalResponse,
    };
  }

  async generateWithModel(query: TierQuery, modelName: string): Promise<TierResponse> {
    const start = performance.now();

    const systemPrompt = query.context.length > 0
      ? `Previous context:\n${query.context.join('\n')}`
      : '';

    const response = await this.client.generate({
      model: modelName,
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
        model: modelName,
        hasLogprobs: !!(response.logprobs && response.logprobs.length > 0),
        logprobCount: response.logprobs?.length ?? 0,
      },
    };
  }
}
