export interface TierResponse {
  tokens: string[];
  confidence: number;
  tier: 1 | 2 | 3;
  escalationSignals: EscalationSignals;
  latencyMs: number;
  metadata?: Record<string, unknown>;
}

export interface EscalationSignals {
  tokenProbabilitySpread: number;
  semanticVelocity: number;
  surpriseScore: number;
  attentionAnomalyScore: number;
}

export interface TierQuery {
  prompt: string;
  context: string[];
  maxTokens: number;
  valenceScore: ValenceScore;
}

export interface ValenceScore {
  urgency: number;
  complexity: number;
  stakes: number;
  composite: number;
}

/**
 * A single streaming token chunk from a tier.
 *
 * isFinal=true on the last chunk, which also carries the complete finalResponse
 * so callers can compute full escalation signals from the entire response.
 */
export interface StreamChunk {
  token: string;
  /** Raw logprob for this token — available when Ollama provides them per-chunk */
  logprob?: number;
  /** Top alternative tokens at this position */
  topLogprobs?: Array<{ token: string; logprob: number }>;
  isFinal: boolean;
  /** Only populated on the last chunk — full TierResponse for signal computation */
  finalResponse?: TierResponse;
}

export interface Tier {
  id: 1 | 2 | 3;
  name: string;
  generate(query: TierQuery): Promise<TierResponse>;
  isAvailable(): Promise<boolean>;
  warmup(): Promise<void>;
  /** Optional: generate using a specific model name (for recovery fallbacks) */
  generateWithModel?(query: TierQuery, modelName: string): Promise<TierResponse>;
  /** Optional: generate using API fallback (for recovery cascade) */
  generateWithApi?(query: TierQuery): Promise<TierResponse>;
  /** Optional: stream tokens one at a time (Phase 5 — per-token routing) */
  generateStream?(query: TierQuery): AsyncGenerator<StreamChunk>;
}
