/**
 * Meta-Confidence Model — Phase 3 Centerpiece
 *
 * A correction layer that learns when the confidence router's escalation
 * decisions are likely to be wrong. Watches the router's track record via
 * a sliding window of recent decisions and their outcomes.
 *
 * NOT a neural network. Uses weighted history with per-signal correction
 * factors. The ML version is Phase 4.
 *
 * Data per decision:
 * - valenceScore at decision time
 * - four signal values at decision time
 * - the routing decision (escalate / stay)
 * - outcome quality (approximate from response coherence/length)
 *
 * Interventions:
 * - SUPPRESS: router says escalate, meta says router is wrong → stay
 * - FORCE: router says stay, meta says router is wrong → escalate
 * - CONFIRM: meta agrees with router → no change
 */

import { appendFile, mkdir } from 'node:fs/promises';
import { existsSync, readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import pino from 'pino';
import type { EscalationSignals, ValenceScore } from '../tiers/tier-interface.js';
import type { RoutingDecision } from '../router/confidence.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const LOG_DIR = resolve(__dirname, '../../logs');
const META_LOG = resolve(LOG_DIR, 'meta.jsonl');

const log = pino({ level: process.env.KINDLING_LOG_LEVEL ?? 'info' });

export interface MetaDecision {
  action: 'confirm' | 'suppress' | 'force';
  reason: string;
  correctionStrength: number; // 0-1, how confident meta is in its correction
}

interface DecisionRecord {
  timestamp: number;
  signals: EscalationSignals;
  valence: ValenceScore;
  routerDecision: 'escalate' | 'stay';
  metaAction: 'confirm' | 'suppress' | 'force';
  currentTier: number;
  responseTokens: number;
  responseCoherence: number;
}

interface MetaLogEntry {
  timestamp: string;
  action: 'confirm' | 'suppress' | 'force';
  reason: string;
  correctionStrength: number;
  routerDecision: 'escalate' | 'stay';
  signals: EscalationSignals;
  valence: ValenceScore;
  currentTier: number;
}

const WINDOW_SIZE = 50; // sliding window of recent decisions

export class MetaConfidenceModel {
  private history: DecisionRecord[] = [];
  // Per-signal correction factors: positive means signal tends to over-escalate,
  // negative means it under-escalates
  private signalCorrections = {
    tokenProbabilitySpread: 0,
    semanticVelocity: 0,
    surpriseScore: 0,
    attentionAnomalyScore: 0,
  };
  private totalDecisions = 0;
  private totalInterventions = 0;

  constructor() {
    this.loadHistory();
  }

  /**
   * Evaluate the router's escalation decision and decide whether to
   * confirm, suppress, or force escalation.
   */
  evaluate(
    routerDecision: RoutingDecision,
    signals: EscalationSignals,
    valence: ValenceScore,
    currentTier: number
  ): MetaDecision {
    this.totalDecisions++;

    // If we have insufficient history, confirm everything
    if (this.history.length < 5) {
      const decision: MetaDecision = {
        action: 'confirm',
        reason: 'insufficient history for meta-correction',
        correctionStrength: 0,
      };
      this.logIntervention(decision, routerDecision, signals, valence, currentTier);
      return decision;
    }

    if (routerDecision.shouldEscalate) {
      return this.evaluateEscalation(routerDecision, signals, valence, currentTier);
    } else {
      return this.evaluateStay(routerDecision, signals, valence, currentTier);
    }
  }

  /**
   * Record the outcome of a routing decision for future learning.
   * Called after the response is generated with quality metrics.
   */
  recordOutcome(
    signals: EscalationSignals,
    valence: ValenceScore,
    routerDecision: 'escalate' | 'stay',
    metaAction: 'confirm' | 'suppress' | 'force',
    currentTier: number,
    responseTokens: number,
    responseCoherence: number
  ): void {
    const record: DecisionRecord = {
      timestamp: Date.now(),
      signals,
      valence,
      routerDecision,
      metaAction,
      currentTier,
      responseTokens,
      responseCoherence,
    };

    this.history.push(record);

    // Trim to window size
    if (this.history.length > WINDOW_SIZE) {
      this.history.shift();
    }

    // Update correction factors
    this.updateCorrections();
  }

  getStats(): { totalDecisions: number; totalInterventions: number; corrections: Record<string, number> } {
    return {
      totalDecisions: this.totalDecisions,
      totalInterventions: this.totalInterventions,
      corrections: { ...this.signalCorrections },
    };
  }

  /**
   * Router says escalate — should we suppress?
   *
   * Look at recent escalations with similar signal profiles. If many of
   * them produced high-quality responses at the LOWER tier (meaning
   * escalation was unnecessary), suppress this one.
   */
  private evaluateEscalation(
    routerDecision: RoutingDecision,
    signals: EscalationSignals,
    valence: ValenceScore,
    currentTier: number
  ): MetaDecision {
    // Find recent escalations from this tier
    const recentEscalations = this.history.filter(
      r => r.routerDecision === 'escalate' && r.currentTier === currentTier
    );

    if (recentEscalations.length < 3) {
      const decision: MetaDecision = {
        action: 'confirm',
        reason: 'insufficient escalation history from this tier',
        correctionStrength: 0,
      };
      this.logIntervention(decision, routerDecision, signals, valence, currentTier);
      return decision;
    }

    // Check if escalations from this tier with similar signals produced
    // low-quality responses (meaning escalation WAS needed) or
    // high-quality responses (meaning it WASN'T needed)
    const similarEscalations = recentEscalations.filter(r =>
      Math.abs(r.signals.tokenProbabilitySpread - signals.tokenProbabilitySpread) < 0.15
    );

    if (similarEscalations.length >= 2) {
      const avgCoherence = similarEscalations.reduce((s, r) => s + r.responseCoherence, 0)
        / similarEscalations.length;

      // If similar escalations produced high coherence at the lower tier,
      // the router may be over-escalating for this signal profile
      if (avgCoherence > 0.85 && this.signalCorrections.tokenProbabilitySpread > 0.1) {
        this.totalInterventions++;
        const decision: MetaDecision = {
          action: 'suppress',
          reason: `similar signal profiles produced high-quality output at Tier ${currentTier} (avg coherence ${avgCoherence.toFixed(2)})`,
          correctionStrength: Math.min(0.8, this.signalCorrections.tokenProbabilitySpread),
        };
        this.logIntervention(decision, routerDecision, signals, valence, currentTier);
        return decision;
      }
    }

    const decision: MetaDecision = {
      action: 'confirm',
      reason: 'escalation consistent with history',
      correctionStrength: 0,
    };
    this.logIntervention(decision, routerDecision, signals, valence, currentTier);
    return decision;
  }

  /**
   * Router says stay — should we force escalation?
   *
   * Look at recent stays with similar signal profiles. If many produced
   * low-quality responses (short, incoherent), we should have escalated.
   */
  private evaluateStay(
    routerDecision: RoutingDecision,
    signals: EscalationSignals,
    valence: ValenceScore,
    currentTier: number
  ): MetaDecision {
    const recentStays = this.history.filter(
      r => r.routerDecision === 'stay' && r.currentTier === currentTier
    );

    if (recentStays.length < 3) {
      const decision: MetaDecision = {
        action: 'confirm',
        reason: 'insufficient stay history from this tier',
        correctionStrength: 0,
      };
      this.logIntervention(decision, routerDecision, signals, valence, currentTier);
      return decision;
    }

    // Find stays with similar signal profiles that produced poor output
    const similarStays = recentStays.filter(r =>
      Math.abs(r.signals.tokenProbabilitySpread - signals.tokenProbabilitySpread) < 0.15
    );

    if (similarStays.length >= 2) {
      const avgCoherence = similarStays.reduce((s, r) => s + r.responseCoherence, 0)
        / similarStays.length;
      const avgTokens = similarStays.reduce((s, r) => s + r.responseTokens, 0)
        / similarStays.length;

      // If similar stays produced poor output, force escalation
      if (avgCoherence < 0.7 && avgTokens < 50 && valence.complexity > 0.2) {
        this.totalInterventions++;
        const decision: MetaDecision = {
          action: 'force',
          reason: `similar signal profiles produced poor output at Tier ${currentTier} (avg coherence ${avgCoherence.toFixed(2)}, avg ${Math.round(avgTokens)} tokens)`,
          correctionStrength: 0.6,
        };
        this.logIntervention(decision, routerDecision, signals, valence, currentTier);
        return decision;
      }
    }

    const decision: MetaDecision = {
      action: 'confirm',
      reason: 'stay decision consistent with history',
      correctionStrength: 0,
    };
    this.logIntervention(decision, routerDecision, signals, valence, currentTier);
    return decision;
  }

  /**
   * Update per-signal correction factors based on recent history.
   *
   * For each signal, look at cases where the router escalated vs stayed
   * and compare outcome quality. If escalations with high signal X produced
   * the same quality as stays with high signal X, then signal X is
   * over-weighted (positive correction = signal tends to cause false escalations).
   */
  private updateCorrections(): void {
    if (this.history.length < 10) return;

    const recent = this.history.slice(-20);
    const escalations = recent.filter(r => r.routerDecision === 'escalate');
    const stays = recent.filter(r => r.routerDecision === 'stay');

    if (escalations.length < 3 || stays.length < 3) return;

    const avgEscCoherence = escalations.reduce((s, r) => s + r.responseCoherence, 0) / escalations.length;
    const avgStayCoherence = stays.reduce((s, r) => s + r.responseCoherence, 0) / stays.length;

    // If escalations don't produce meaningfully better output than stays,
    // the signals are over-triggering
    const qualityGap = avgEscCoherence - avgStayCoherence;

    for (const signal of ['tokenProbabilitySpread', 'semanticVelocity', 'surpriseScore', 'attentionAnomalyScore'] as const) {
      const escalatedSignalAvg = escalations.reduce((s, r) => s + r.signals[signal], 0) / escalations.length;
      const staySignalAvg = stays.reduce((s, r) => s + r.signals[signal], 0) / stays.length;
      const signalGap = escalatedSignalAvg - staySignalAvg;

      // If signal is high in escalations but quality gap is small,
      // signal is over-weighted
      if (signalGap > 0.1 && qualityGap < 0.05) {
        this.signalCorrections[signal] = Math.min(1, this.signalCorrections[signal] + 0.05);
      } else if (qualityGap > 0.1) {
        // Escalations produce notably better output — signal is correctly weighted
        this.signalCorrections[signal] = Math.max(-1, this.signalCorrections[signal] - 0.02);
      }
    }
  }

  private logIntervention(
    decision: MetaDecision,
    routerDecision: RoutingDecision,
    signals: EscalationSignals,
    valence: ValenceScore,
    currentTier: number
  ): void {
    if (decision.action !== 'confirm') {
      log.info(
        { action: decision.action, reason: decision.reason, strength: decision.correctionStrength },
        `Meta-confidence: ${decision.action}`
      );
    }

    const entry: MetaLogEntry = {
      timestamp: new Date().toISOString(),
      action: decision.action,
      reason: decision.reason,
      correctionStrength: decision.correctionStrength,
      routerDecision: routerDecision.shouldEscalate ? 'escalate' : 'stay',
      signals,
      valence,
      currentTier,
    };

    (async () => {
      if (!existsSync(LOG_DIR)) {
        await mkdir(LOG_DIR, { recursive: true });
      }
      await appendFile(META_LOG, JSON.stringify(entry) + '\n', 'utf-8');
    })().catch(() => {});
  }

  private loadHistory(): void {
    // Load recent meta log entries as history seed
    if (!existsSync(META_LOG)) return;
    try {
      const lines = readFileSync(META_LOG, 'utf-8').trim().split('\n').slice(-WINDOW_SIZE);
      // We don't have full DecisionRecords in the log, but we can seed
      // the correction factors from intervention patterns
      let suppressions = 0;
      let forces = 0;
      for (const line of lines) {
        try {
          const entry = JSON.parse(line) as MetaLogEntry;
          if (entry.action === 'suppress') suppressions++;
          if (entry.action === 'force') forces++;
        } catch { /* skip malformed lines */ }
      }
      // If we've been suppressing a lot, signals are probably over-weighted
      if (suppressions > lines.length * 0.3) {
        this.signalCorrections.tokenProbabilitySpread = 0.2;
      }
    } catch { /* no history, start fresh */ }
  }
}
