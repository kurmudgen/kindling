import { getConfig } from '../config/config.js';
import type { EscalationSignals, ValenceScore } from '../tiers/tier-interface.js';

export interface RoutingDecision {
  targetTier: 1 | 2 | 3;
  confidence: number;
  reason: string;
  shouldEscalate: boolean;
}

export class ConfidenceAggregator {
  private deescalationCounter = 0;

  computeConfidence(signals: EscalationSignals): number {
    const config = getConfig().escalation;
    const weighted =
      signals.tokenProbabilitySpread * config.tokenProbabilitySpreadWeight +
      signals.semanticVelocity * config.semanticVelocityWeight +
      signals.surpriseScore * config.surpriseScoreWeight +
      signals.attentionAnomalyScore * config.attentionAnomalyWeight;

    return Math.max(0, Math.min(1, 1 - weighted));
  }

  decide(
    currentTier: 1 | 2 | 3,
    signals: EscalationSignals,
    valence: ValenceScore
  ): RoutingDecision {
    const config = getConfig().escalation;
    const confidence = this.computeConfidence(signals);

    // Valence biases: high-stakes/urgent queries lower the effective confidence
    // making escalation more likely
    const valenceAdjustedConfidence = confidence - valence.composite * 0.2;

    // Escalation check
    if (valenceAdjustedConfidence < (1 - config.escalationThreshold) && currentTier < 3) {
      this.deescalationCounter = 0;
      const nextTier = (currentTier + 1) as 1 | 2 | 3;
      return {
        targetTier: nextTier,
        confidence: valenceAdjustedConfidence,
        reason: this.buildEscalationReason(signals, valence),
        shouldEscalate: true,
      };
    }

    // De-escalation check: signals must stay below threshold for N consecutive tokens
    if (valenceAdjustedConfidence > (1 - config.deescalationThreshold) && currentTier > 1) {
      this.deescalationCounter++;
      if (this.deescalationCounter >= config.deescalationTokenWindow) {
        this.deescalationCounter = 0;
        const prevTier = (currentTier - 1) as 1 | 2 | 3;
        return {
          targetTier: prevTier,
          confidence: valenceAdjustedConfidence,
          reason: `Signals stable for ${config.deescalationTokenWindow} tokens, de-escalating`,
          shouldEscalate: false,
        };
      }
    } else {
      this.deescalationCounter = 0;
    }

    return {
      targetTier: currentTier,
      confidence: valenceAdjustedConfidence,
      reason: 'Holding current tier',
      shouldEscalate: false,
    };
  }

  reset(): void {
    this.deescalationCounter = 0;
  }

  private buildEscalationReason(signals: EscalationSignals, valence: ValenceScore): string {
    const reasons: string[] = [];
    if (signals.tokenProbabilitySpread > 0.6) reasons.push('high token uncertainty');
    if (signals.semanticVelocity > 0.7) reasons.push('rapid meaning shifts');
    if (signals.surpriseScore > 0.5) reasons.push('unexpected content');
    if (signals.attentionAnomalyScore > 0.4) reasons.push('attention anomalies');
    if (valence.composite > 0.5) reasons.push(`high valence (${valence.composite.toFixed(2)})`);
    return reasons.length > 0 ? reasons.join(', ') : 'composite threshold exceeded';
  }
}
