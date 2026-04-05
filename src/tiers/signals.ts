import { getConfig } from '../config/config.js';
import type { EscalationSignals } from './tier-interface.js';

/** Per-token logprob data from Ollama */
export interface TokenLogprobData {
  token: string;
  logprob: number;
  topLogprobs?: Array<{ token: string; logprob: number }>;
}

/**
 * Compute escalation signals from actual logprob data.
 * This is the core signal computation used by all Ollama-backed tiers.
 */
export function computeSignalsFromLogprobs(
  logprobs: TokenLogprobData[],
  tokens: string[]
): EscalationSignals {
  return {
    tokenProbabilitySpread: computeTokenProbabilitySpread(logprobs),
    semanticVelocity: computeSemanticVelocity(logprobs),
    surpriseScore: computeSurpriseScore(logprobs),
    attentionAnomalyScore: computeAttentionAnomaly(tokens),
  };
}

/**
 * Compute escalation signals from heuristics when logprobs aren't available.
 * Used as fallback for API-backed tiers that don't expose logprobs.
 */
export function computeSignalsFromHeuristics(
  tokens: string[],
  evalCount?: number,
  evalDuration?: number,
  expectedTps?: number
): EscalationSignals {
  const actualTps = evalCount && evalDuration
    ? evalCount / (evalDuration / 1e9)
    : 0;
  const speedRatio = expectedTps ? actualTps / expectedTps : 1;

  return {
    tokenProbabilitySpread: Math.max(0, Math.min(1, 1 - speedRatio * 0.5)),
    semanticVelocity: computeVocabDiversity(tokens),
    surpriseScore: 0.1,
    attentionAnomalyScore: computeAttentionAnomaly(tokens),
  };
}

/**
 * Token probability spread: measures how uncertain the model is across tokens.
 *
 * High spread = model was unsure about many tokens = should escalate.
 * Computed from logprob entropy: low logprob (e.g., -5) means low probability.
 *
 * When top_logprobs available, we use the entropy of the top-k distribution.
 * When only the chosen token logprob is available, we use mean logprob.
 */
function computeTokenProbabilitySpread(logprobs: TokenLogprobData[]): number {
  if (logprobs.length === 0) return 0.5;

  // If we have top_logprobs, compute average entropy across tokens
  const hasTopLogprobs = logprobs.some(lp => lp.topLogprobs && lp.topLogprobs.length > 1);

  if (hasTopLogprobs) {
    let totalEntropy = 0;
    let count = 0;
    for (const lp of logprobs) {
      if (lp.topLogprobs && lp.topLogprobs.length > 1) {
        // Convert logprobs to probabilities and compute entropy
        const probs = lp.topLogprobs.map(t => Math.exp(t.logprob));
        const sumProbs = probs.reduce((a, b) => a + b, 0);
        const normalizedProbs = probs.map(p => p / sumProbs);
        const entropy = -normalizedProbs.reduce((sum, p) =>
          p > 0 ? sum + p * Math.log2(p) : sum, 0);
        // Normalize entropy: log2(k) is max entropy for k choices
        const maxEntropy = Math.log2(lp.topLogprobs.length);
        totalEntropy += maxEntropy > 0 ? entropy / maxEntropy : 0;
        count++;
      }
    }
    return count > 0 ? totalEntropy / count : 0.5;
  }

  // Fallback: use mean logprob directly
  // logprob of 0 = 100% confident, logprob of -10 = very uncertain
  const meanLogprob = logprobs.reduce((sum, lp) => sum + lp.logprob, 0) / logprobs.length;
  // Map logprob range [-10, 0] to spread [1, 0]
  return Math.max(0, Math.min(1, -meanLogprob / 10));
}

/**
 * Semantic velocity: rate of meaning change across the response.
 *
 * Measured by tracking how much the logprob pattern changes over sliding windows.
 * High velocity = model is jumping between topics = may be struggling.
 *
 * Unlike the Phase 1 version (which used vocabulary diversity and was inverted),
 * this measures variance in per-token confidence, which correlates with the model
 * struggling through unfamiliar territory.
 */
function computeSemanticVelocity(logprobs: TokenLogprobData[]): number {
  if (logprobs.length < 4) return 0;

  // Compute variance in logprob values across a sliding window
  const windowSize = Math.min(8, Math.floor(logprobs.length / 2));
  const windowMeans: number[] = [];

  for (let i = 0; i <= logprobs.length - windowSize; i++) {
    const windowLogprobs = logprobs.slice(i, i + windowSize).map(lp => lp.logprob);
    const mean = windowLogprobs.reduce((a, b) => a + b, 0) / windowLogprobs.length;
    windowMeans.push(mean);
  }

  if (windowMeans.length < 2) return 0;

  // Compute variance of window means (how much the "confidence landscape" shifts)
  const overallMean = windowMeans.reduce((a, b) => a + b, 0) / windowMeans.length;
  const variance = windowMeans.reduce((sum, m) =>
    sum + (m - overallMean) ** 2, 0) / windowMeans.length;

  // Normalize: variance of 0 = stable, high variance = shifting
  // Empirically, logprob variance > 4 is very unstable
  return Math.max(0, Math.min(1, Math.sqrt(variance) / 2));
}

/**
 * Surprise score: how many tokens were genuinely unexpected.
 *
 * A token is "surprising" if its logprob is below a threshold (model assigned
 * it < ~5% probability). High surprise rate = model is generating content
 * it's not confident about.
 */
function computeSurpriseScore(logprobs: TokenLogprobData[]): number {
  if (logprobs.length === 0) return 0;

  // logprob < -3.0 means probability < ~5% (e^-3 ≈ 0.05)
  const surpriseThreshold = -3.0;
  const surprisingTokens = logprobs.filter(lp => lp.logprob < surpriseThreshold).length;
  return surprisingTokens / logprobs.length;
}

/**
 * Attention anomaly: detects repetitive or degenerate output patterns.
 *
 * Combines skip-bigram repetition with n-gram repetition detection.
 * High score = model attention is looping.
 */
function computeAttentionAnomaly(tokens: string[]): number {
  if (tokens.length < 6) return 0;

  // Skip-2 repetition (existing approach, still valid for degenerate detection)
  let skipReps = 0;
  for (let i = 2; i < tokens.length; i++) {
    if (tokens[i].toLowerCase() === tokens[i - 2].toLowerCase()) skipReps++;
  }
  const skipRate = skipReps / (tokens.length - 2);

  // Trigram repetition: how many 3-grams appear more than once
  const trigrams = new Map<string, number>();
  for (let i = 0; i < tokens.length - 2; i++) {
    const tri = `${tokens[i]} ${tokens[i + 1]} ${tokens[i + 2]}`.toLowerCase();
    trigrams.set(tri, (trigrams.get(tri) ?? 0) + 1);
  }
  const repeatedTrigrams = [...trigrams.values()].filter(c => c > 1).length;
  const trigramRate = trigrams.size > 0 ? repeatedTrigrams / trigrams.size : 0;

  // Weighted combination
  return Math.min(1, skipRate * 0.4 + trigramRate * 0.6);
}

/** Simple vocabulary diversity — used as fallback for API tiers */
function computeVocabDiversity(tokens: string[]): number {
  if (tokens.length === 0) return 0;
  const unique = new Set(tokens.map(t => t.toLowerCase()));
  return unique.size / tokens.length;
}

/**
 * Compute overall confidence from escalation signals using config weights.
 * Confidence is inverse of weighted signal strength.
 */
export function computeConfidence(signals: EscalationSignals): number {
  const config = getConfig().escalation;
  const weighted =
    signals.tokenProbabilitySpread * config.tokenProbabilitySpreadWeight +
    signals.semanticVelocity * config.semanticVelocityWeight +
    signals.surpriseScore * config.surpriseScoreWeight +
    signals.attentionAnomalyScore * config.attentionAnomalyWeight;

  return Math.max(0, Math.min(1, 1 - weighted));
}
