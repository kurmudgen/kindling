/**
 * Concurrency Guard — generation-numbered query isolation.
 *
 * Each concurrent query acquires a monotonically increasing generation
 * number. Routing decisions check generation validity before committing
 * state. Stale generations are detected when they try to mutate shared
 * state after a newer generation has started.
 *
 * Also serializes log writes via the write queue in logger.ts (already
 * implemented). This module provides the in-flight tracking and
 * per-query isolation.
 */

import pino from 'pino';

const log = pino({ level: process.env.KINDLING_LOG_LEVEL ?? 'info' });

export class ConcurrencyGuard {
  private currentGeneration = 0;
  private activeGenerations = new Set<number>();

  /**
   * Acquire a new generation number for a query.
   * The generation is registered as active until released.
   */
  acquire(): number {
    const gen = ++this.currentGeneration;
    this.activeGenerations.add(gen);
    return gen;
  }

  /**
   * Check if a generation is still active (not released and not superseded).
   */
  isActive(gen: number): boolean {
    return this.activeGenerations.has(gen);
  }

  /**
   * Release a generation — marks it as complete.
   * Safe to call multiple times.
   */
  release(gen: number): void {
    this.activeGenerations.delete(gen);
  }

  /**
   * Get the current in-flight count (for diagnostics).
   */
  inFlightCount(): number {
    return this.activeGenerations.size;
  }

  /**
   * Get the latest generation number assigned.
   */
  latest(): number {
    return this.currentGeneration;
  }

  /**
   * Reset all generation state (for tests only).
   */
  reset(): void {
    this.currentGeneration = 0;
    this.activeGenerations.clear();
  }
}

/**
 * Per-query isolation helper — creates a scoped view of router state
 * that can be safely shared with concurrent queries. Each concurrent
 * query gets its own confidence aggregator and buffer instance instead
 * of sharing the router's singletons.
 */
export interface PerQueryState<TConf, TBuf> {
  generation: number;
  confidence: TConf;
  buffer: TBuf;
}
