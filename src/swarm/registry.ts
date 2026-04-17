/**
 * Swarm Node Registry — Phase 6
 *
 * Manages a fleet of Ollama endpoints, each running a different model
 * optimized for a specific domain or hardware tier. Nodes can run on
 * the same machine (different models) or remote machines (multi-host swarm).
 *
 * Hardware types:
 *   gpu    — VRAM-resident, fast (< 30s typical)
 *   cpu    — RAM-resident, slow but can run much larger models
 *   remote — another machine's Ollama endpoint
 *
 * Config loaded from: config/swarm.json
 */

import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import pino from 'pino';
import type { Domain } from './domain.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CONFIG_FILE = resolve(__dirname, '../../config/swarm.json');

const log = pino({ level: process.env.KINDLING_LOG_LEVEL ?? 'info' });

export type Hardware = 'gpu' | 'cpu' | 'remote';

export interface SwarmNode {
  id: string;
  /** Ollama base URL, e.g. http://localhost:11434 */
  endpoint: string;
  model: string;
  /** Domains this node specializes in — ordered by preference */
  domains: Domain[];
  hardware: Hardware;
  /** Expected worst-case latency in ms — used for latency gating */
  maxLatencyMs: number;
  /** Human-readable description */
  description?: string;
}

export interface SwarmConfig {
  nodes: SwarmNode[];
  /** Default latency budget for routing decisions (ms) */
  defaultLatencyBudgetMs: number;
}

// ─── Default config — auto-detected from available Ollama models ─────────────

const DEFAULT_CONFIG: SwarmConfig = {
  defaultLatencyBudgetMs: 60_000,
  nodes: [
    {
      id: 'fast-general',
      endpoint: 'http://localhost:11434',
      model: 'qwen2.5:1.5b',
      domains: ['factual', 'general'],
      hardware: 'gpu',
      maxLatencyMs: 10_000,
      description: 'Ultra-fast small model for simple factual queries',
    },
    {
      id: 'code-small',
      endpoint: 'http://localhost:11434',
      model: 'qwen2.5-coder:7b',
      domains: ['code'],
      hardware: 'gpu',
      maxLatencyMs: 30_000,
      description: 'Code specialist — fast, 7B',
    },
    {
      id: 'code-large',
      endpoint: 'http://localhost:11434',
      model: 'deepseek-coder-v2:16b',
      domains: ['code'],
      hardware: 'gpu',
      maxLatencyMs: 90_000,
      description: 'Code specialist — deep, 16B',
    },
    {
      id: 'general-medium',
      endpoint: 'http://localhost:11434',
      model: 'qwen2.5:14b',
      domains: ['general', 'factual', 'creative'],
      hardware: 'gpu',
      maxLatencyMs: 60_000,
      description: 'General purpose medium model',
    },
    {
      id: 'reasoning-deep',
      endpoint: 'http://localhost:11434',
      model: 'deepseek-r1:70b',
      domains: ['reasoning', 'math'],
      hardware: 'cpu',
      maxLatencyMs: 600_000,
      description: 'Deep reasoning — 70B, CPU/RAM offloaded, slow but powerful',
    },
    {
      id: 'general-large',
      endpoint: 'http://localhost:11434',
      model: 'qwen2.5:32b',
      domains: ['general', 'reasoning', 'creative'],
      hardware: 'cpu',
      maxLatencyMs: 300_000,
      description: 'Large general — 32B, partial CPU offload',
    },
  ],
};

// ─── Registry ────────────────────────────────────────────────────────────────

export class NodeRegistry {
  private config: SwarmConfig;
  private healthCache: Map<string, { healthy: boolean; checkedAt: number }> = new Map();
  private readonly HEALTH_CACHE_TTL_MS = 30_000;

  constructor() {
    this.config = this.loadConfig();
    log.info({ nodeCount: this.config.nodes.length }, 'Swarm node registry initialized');
  }

  private loadConfig(): SwarmConfig {
    if (existsSync(CONFIG_FILE)) {
      try {
        const raw = JSON.parse(readFileSync(CONFIG_FILE, 'utf-8')) as SwarmConfig;
        log.info({ file: CONFIG_FILE, nodes: raw.nodes.length }, 'Loaded swarm config');
        return raw;
      } catch (err) {
        log.warn({ err }, 'Failed to parse swarm.json — using defaults');
      }
    }
    log.info('No swarm.json found — using built-in default node config');
    return DEFAULT_CONFIG;
  }

  /** All registered nodes */
  getNodes(): SwarmNode[] {
    return this.config.nodes;
  }

  /** Default latency budget from config */
  getDefaultLatencyBudgetMs(): number {
    return this.config.defaultLatencyBudgetMs;
  }

  /**
   * Find the best node for a given domain and latency budget.
   *
   * Selection priority:
   * 1. Must be a domain match (or general if no specialist exists)
   * 2. Must fit within latencyBudgetMs
   * 3. Within candidates: prefer gpu > cpu, then smaller maxLatencyMs
   */
  selectNode(domain: Domain, latencyBudgetMs: number): SwarmNode | null {
    const candidates = this.config.nodes.filter(n =>
      n.domains.includes(domain) && n.maxLatencyMs <= latencyBudgetMs
    );

    if (candidates.length === 0) {
      // No specialist — fall back to general nodes within budget
      const fallback = this.config.nodes.filter(n =>
        n.domains.includes('general') && n.maxLatencyMs <= latencyBudgetMs
      );
      if (fallback.length === 0) return null;
      return this.pickBest(fallback);
    }

    return this.pickBest(candidates);
  }

  /**
   * Find all nodes suitable for a domain, ordered by preference.
   * Used for fallback chains.
   */
  getNodeChain(domain: Domain): SwarmNode[] {
    const specialists = this.config.nodes.filter(n => n.domains.includes(domain));
    const generals = this.config.nodes.filter(n =>
      n.domains.includes('general') && !specialists.includes(n)
    );
    return [...specialists, ...generals].sort((a, b) => {
      // GPU before CPU, then by latency
      if (a.hardware === 'gpu' && b.hardware !== 'gpu') return -1;
      if (a.hardware !== 'gpu' && b.hardware === 'gpu') return 1;
      return a.maxLatencyMs - b.maxLatencyMs;
    });
  }

  /** Check if an Ollama endpoint is reachable (cached for 30s) */
  async isNodeHealthy(node: SwarmNode): Promise<boolean> {
    const cached = this.healthCache.get(node.id);
    if (cached && Date.now() - cached.checkedAt < this.HEALTH_CACHE_TTL_MS) {
      return cached.healthy;
    }

    try {
      const res = await fetch(`${node.endpoint}/api/tags`, {
        signal: AbortSignal.timeout(3000),
      });
      const healthy = res.ok;
      this.healthCache.set(node.id, { healthy, checkedAt: Date.now() });
      return healthy;
    } catch {
      this.healthCache.set(node.id, { healthy: false, checkedAt: Date.now() });
      return false;
    }
  }

  /** Get a summary of all nodes for diagnostics */
  getSummary(): Array<{ id: string; model: string; domains: string[]; hardware: string; maxLatencyMs: number }> {
    return this.config.nodes.map(n => ({
      id: n.id,
      model: n.model,
      domains: n.domains,
      hardware: n.hardware,
      maxLatencyMs: n.maxLatencyMs,
    }));
  }

  private pickBest(nodes: SwarmNode[]): SwarmNode {
    return nodes.sort((a, b) => {
      if (a.hardware === 'gpu' && b.hardware !== 'gpu') return -1;
      if (a.hardware !== 'gpu' && b.hardware === 'gpu') return 1;
      return a.maxLatencyMs - b.maxLatencyMs;
    })[0];
  }
}
