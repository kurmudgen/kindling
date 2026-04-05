/**
 * Cold Concept Prewarmer
 *
 * During sleep stage, the analyst identifies concept clusters worth pre-warming.
 * After sleep, the prewarmer runs short prompts through Tier 1 to warm the
 * model's context window with flagged concepts.
 *
 * APPROACH: Prompt prefix injection.
 * Ollama does not expose direct KV cache manipulation via its API, so we
 * implement pre-warming by generating short responses for each concept cluster.
 * This forces the model to load relevant token embeddings and attention patterns
 * into its active state. The keep_alive parameter ensures the warmed state
 * persists until the next real query arrives.
 *
 * LIMITATION: This warms the model's loaded state but not the KV cache for
 * specific conversations. Each new query starts a fresh KV cache. The benefit
 * is that model weights and embeddings are hot in RAM/CPU cache, reducing
 * first-token latency for related concepts.
 */

import { Ollama } from 'ollama';
import pino from 'pino';
import { getConfig, getOllamaHost } from '../config/config.js';

const log = pino({ level: process.env.KINDLING_LOG_LEVEL ?? 'info' });

export class ConceptPrewarmer {
  private client: Ollama;
  private model: string;

  constructor() {
    this.client = new Ollama({ host: getOllamaHost() });
    this.model = getConfig().tier1.model;
  }

  /**
   * Pre-warm the model with a list of concept clusters.
   * Each concept generates a short probe prompt to activate relevant embeddings.
   */
  async prewarm(concepts: string[]): Promise<void> {
    if (concepts.length === 0) return;

    log.info({ count: concepts.length }, 'Pre-warming concept clusters');

    for (const concept of concepts) {
      try {
        const probePrompt = this.buildProbePrompt(concept);
        await this.client.generate({
          model: this.model,
          prompt: probePrompt,
          options: {
            num_predict: 10, // minimal generation — just enough to activate pathways
            num_thread: getConfig().tier1.maxCores,
          },
          keep_alive: '10m',
        });
        log.debug({ concept }, 'Concept pre-warmed');
      } catch (err) {
        log.warn({ concept, err }, 'Failed to pre-warm concept');
      }
    }

    log.info('Concept pre-warming complete');
  }

  /**
   * Build a short probe prompt that activates the model's knowledge
   * pathways for a given concept without generating a full response.
   */
  private buildProbePrompt(concept: string): string {
    return `Briefly define: ${concept}`;
  }
}
