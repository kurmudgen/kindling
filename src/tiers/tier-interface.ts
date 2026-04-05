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
}
