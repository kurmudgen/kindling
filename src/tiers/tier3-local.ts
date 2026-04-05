/**
 * Tier 3 Local — NVMe Layer Streaming
 *
 * Loads a large model via Ollama with layer-by-layer streaming from disk.
 * Uses Ollama's native model management which handles mmap-based layer
 * loading from disk (NVMe/SSD) automatically.
 *
 * GPU staging buffer: If VRAM is available, Ollama will use it as a fast
 * buffer for active layers. If not, it falls back to RAM-only with mmap
 * from disk. This is fully transparent to our code — Ollama handles the
 * hardware detection and layer placement.
 *
 * LRU layer caching: Ollama's keep_alive parameter controls how long
 * model layers stay in memory. We configure this from the hardware profile
 * to balance memory pressure vs cold-start latency.
 *
 * HARDWARE REQUIREMENTS:
 * - Minimum: NVMe/SSD with enough space for the model (varies by model)
 * - Recommended: 32GB+ RAM for 30B+ models in q4 quantization
 * - Optional: GPU with VRAM for layer acceleration (graceful degradation without)
 *
 * GRACEFUL DEGRADATION:
 * - No GPU → Ollama uses CPU + RAM mmap, slower but functional
 * - Limited RAM → Ollama streams layers from disk via mmap, very slow but works
 * - Model not pulled → falls back to Tier3 API
 */

import { Ollama } from 'ollama';
import { getConfig, getOllamaHost } from '../config/config.js';
import {
  computeSignalsFromLogprobs,
  computeSignalsFromHeuristics,
  computeConfidence,
} from './signals.js';
import type { TokenLogprobData } from './signals.js';
import type { Tier, TierQuery, TierResponse } from './tier-interface.js';

export class Tier3Local implements Tier {
  readonly id = 3 as const;
  readonly name = 'Deep (Local NVMe Streaming)';
  private client: Ollama;
  private model: string;
  private keepAlive: string;

  constructor() {
    const config = getConfig();
    this.client = new Ollama({ host: getOllamaHost() });
    this.model = config.tier3.localModel ?? 'qwen2.5:32b';
    // Keep model in memory based on profile — prosumer keeps longer
    this.keepAlive = config.tier3.keepAlive ?? '5m';
  }

  async isAvailable(): Promise<boolean> {
    try {
      const models = await this.client.list();
      return models.models.some(m =>
        m.name === this.model || m.name.startsWith(this.model + '-')
      );
    } catch {
      return false;
    }
  }

  async warmup(): Promise<void> {
    if (!(await this.isAvailable())) {
      throw new Error(
        `Tier 3 local model "${this.model}" not available. ` +
        `Run: ollama pull ${this.model}`
      );
    }
    // Trigger model load into memory / mmap
    try {
      await this.client.generate({
        model: this.model,
        prompt: '',
        keep_alive: this.keepAlive,
      });
    } catch {
      throw new Error(`Failed to warm Tier 3 local model "${this.model}"`);
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
      },
      keep_alive: this.keepAlive,
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
        8 // expected TPS for large model on CPU
      );
    }

    const confidence = computeConfidence(escalationSignals);

    return {
      tokens,
      confidence: Math.max(confidence, 0.85), // Deep local tier has high baseline confidence
      tier: 3,
      escalationSignals,
      latencyMs,
      metadata: {
        source: 'ollama-local',
        model: this.model,
        hasLogprobs: !!(response.logprobs && response.logprobs.length > 0),
        evalCount: response.eval_count,
        evalDuration: response.eval_duration,
        totalDuration: response.total_duration,
      },
    };
  }
}
