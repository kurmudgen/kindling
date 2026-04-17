/**
 * Swarm Coordinator — Phase 6
 *
 * Routes queries to the right specialist node based on domain detection,
 * hardware availability, and latency budget. Falls back through a chain:
 *
 *   1. Best specialist for detected domain (GPU, within latency budget)
 *   2. Next specialist in chain (larger/slower GPU)
 *   3. CPU node (RAM-resident large model, long latency budget)
 *   4. Router fallback (existing Kindling 3-tier routing)
 *   5. API (Router's Tier 3)
 *
 * Key insight for small-GPU machines:
 *   A 6GB GPU + 64GB RAM can run a 4B model at GPU speed AND a 30B model
 *   at CPU speed. The coordinator decides which queries can wait 2 minutes
 *   (CPU node) vs need sub-10s response (GPU node).
 *
 * Each "node" is just an Ollama endpoint — no Kindling server required
 * on the remote side. Any machine running Ollama can be a swarm node.
 */

import pino from 'pino';
import { Ollama } from 'ollama';
import { DomainDetector } from './domain.js';
import { NodeRegistry } from './registry.js';
import type { SwarmNode } from './registry.js';
import type { Domain } from './domain.js';
import { Router } from '../router/router.js';
import type { QueryResult } from '../router/router.js';

const log = pino({ level: process.env.KINDLING_LOG_LEVEL ?? 'info' });

export interface SwarmResult extends QueryResult {
  swarmNode?: string;       // which node served this
  swarmDomain?: Domain;     // detected domain
  swarmConfidence?: number; // domain detection confidence
  swarmFallback?: boolean;  // true if router fallback was used
}

export interface SwarmOptions {
  /** Max time to wait for a response in ms. Defaults to registry default. */
  latencyBudgetMs?: number;
  /** If true, allow CPU nodes (may take minutes). Default: true. */
  allowCpu?: boolean;
  /** Context lines for multi-turn conversations */
  context?: string[];
}

export class SwarmCoordinator {
  private detector: DomainDetector;
  private registry: NodeRegistry;
  private router: Router;
  private initialized = false;

  constructor() {
    this.detector = new DomainDetector();
    this.registry = new NodeRegistry();
    this.router = new Router();
  }

  async init(): Promise<void> {
    if (this.initialized) return;
    await this.router.init();
    this.initialized = true;
    log.info({ nodes: this.registry.getNodes().length }, 'SwarmCoordinator initialized');
  }

  /**
   * Route a query through the swarm.
   *
   * Domain is detected first, then the best node is selected.
   * Falls back to the local Router if no specialist is available or healthy.
   */
  async query(prompt: string, options: SwarmOptions = {}): Promise<SwarmResult> {
    if (!this.initialized) await this.init();

    const latencyBudget = options.latencyBudgetMs ?? this.registry.getDefaultLatencyBudgetMs();
    const allowCpu = options.allowCpu ?? true;
    const context = options.context ?? [];

    // Detect domain
    const domainResult = this.detector.detect(prompt);
    const { domain, confidence } = domainResult;

    log.debug({ domain, confidence: confidence.toFixed(2) }, 'Domain detected');

    // Get ordered fallback chain for this domain
    const chain = this.registry.getNodeChain(domain).filter(n => {
      if (!allowCpu && n.hardware === 'cpu') return false;
      if (n.maxLatencyMs > latencyBudget) return false;
      return true;
    });

    // Try each node in the chain
    for (const node of chain) {
      const healthy = await this.registry.isNodeHealthy(node);
      if (!healthy) {
        log.debug({ nodeId: node.id }, 'Node unhealthy — skipping');
        continue;
      }

      try {
        const result = await this.callNode(node, prompt, context, latencyBudget);
        return {
          ...result,
          swarmNode: node.id,
          swarmDomain: domain,
          swarmConfidence: confidence,
          swarmFallback: false,
        };
      } catch (err) {
        log.warn({ nodeId: node.id, err }, 'Node failed — trying next in chain');
      }
    }

    // All specialist nodes failed or were skipped — fall back to Router
    log.info(
      { domain, nodesAttempted: chain.length },
      'Swarm chain exhausted — falling back to Router'
    );

    const fallback = await this.router.queryDetailed(prompt, context);
    return {
      ...fallback,
      swarmDomain: domain,
      swarmConfidence: confidence,
      swarmFallback: true,
    };
  }

  /**
   * Stream tokens from the best available node.
   * Falls back to router.queryStream() if no specialist is available.
   */
  async *queryStream(prompt: string, options: SwarmOptions = {}): AsyncGenerator<string> {
    if (!this.initialized) await this.init();

    const latencyBudget = options.latencyBudgetMs ?? this.registry.getDefaultLatencyBudgetMs();
    const allowCpu = options.allowCpu ?? true;
    const context = options.context ?? [];

    const domainResult = this.detector.detect(prompt);
    const { domain } = domainResult;

    const chain = this.registry.getNodeChain(domain).filter(n => {
      if (!allowCpu && n.hardware === 'cpu') return false;
      if (n.maxLatencyMs > latencyBudget) return false;
      return true;
    });

    for (const node of chain) {
      const healthy = await this.registry.isNodeHealthy(node);
      if (!healthy) continue;

      try {
        yield* this.streamFromNode(node, prompt, context);
        return;
      } catch (err) {
        log.warn({ nodeId: node.id, err }, 'Node stream failed — trying next');
      }
    }

    // Fall back to router streaming
    yield* this.router.queryStream(prompt, context);
  }

  /** Diagnostics — node summary + router stats */
  getStatus(): {
    nodes: ReturnType<NodeRegistry['getSummary']>;
    routerInFlight: number;
  } {
    return {
      nodes: this.registry.getSummary(),
      routerInFlight: this.router.inFlightCount(),
    };
  }

  // ─── Private helpers ────────────────────────────────────────────────────────

  private async callNode(
    node: SwarmNode,
    prompt: string,
    context: string[],
    latencyBudgetMs: number
  ): Promise<QueryResult> {
    const start = performance.now();
    const client = new Ollama({ host: node.endpoint });

    const systemPrompt = context.length > 0
      ? `Previous context:\n${context.join('\n')}`
      : undefined;

    const timeout = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`Node ${node.id} timed out after ${latencyBudgetMs}ms`)), latencyBudgetMs)
    );

    const response = await Promise.race([
      client.generate({
        model: node.model,
        system: systemPrompt,
        prompt,
        options: { num_predict: 2048 },
      }),
      timeout,
    ]);

    const latencyMs = performance.now() - start;
    const text = response.response;

    log.info(
      { nodeId: node.id, model: node.model, latencyMs: Math.round(latencyMs) },
      'Swarm node response'
    );

    return {
      text,
      tier: node.hardware === 'gpu' ? 1 : 2,
      escalated: false,
      escalationPath: [1],
      latencyMs,
      confidence: 0.8,
    };
  }

  private async *streamFromNode(
    node: SwarmNode,
    prompt: string,
    context: string[]
  ): AsyncGenerator<string> {
    const client = new Ollama({ host: node.endpoint });

    const systemPrompt = context.length > 0
      ? `Previous context:\n${context.join('\n')}`
      : undefined;

    const stream = await client.generate({
      model: node.model,
      system: systemPrompt,
      prompt,
      options: { num_predict: 2048 },
      stream: true,
    });

    for await (const chunk of stream) {
      if (chunk.response) yield chunk.response;
    }
  }
}
