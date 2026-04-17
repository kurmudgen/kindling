/**
 * Domain Detector — Phase 6 Swarm
 *
 * Fast heuristic classifier that detects the domain of a query
 * (code, math, reasoning, creative, factual, general) so the
 * SwarmCoordinator can route to the right specialist node.
 *
 * No ML, no Ollama calls — pure string scoring under 1ms.
 */

export type Domain = 'code' | 'math' | 'reasoning' | 'creative' | 'factual' | 'general';

export interface DomainScore {
  domain: Domain;
  confidence: number; // 0–1
  scores: Record<Domain, number>; // raw scores for diagnostics
}

// ─── Pattern sets ────────────────────────────────────────────────────────────

const PATTERNS: Record<Domain, Array<RegExp | string>> = {
  code: [
    /```[\s\S]*?```/,           // code blocks
    /\b(function|class|interface|import|export|const|let|var|def|async|await)\b/i,
    /\b(implement|write|code|program|script|debug|refactor|lint|compile)\b/i,
    /\b(typescript|javascript|python|rust|go|java|c\+\+|sql|bash|shell)\b/i,
    /\b(api|endpoint|rest|graphql|grpc|http|tcp|socket)\b/i,
    /\b(algorithm|data structure|tree|graph|hash|queue|stack|heap)\b/i,
    /\b(docker|kubernetes|ci\/cd|pipeline|deployment|microservice)\b/i,
    /[(){};=<>!].*[(){};=<>!]/,  // code-like punctuation density
  ],

  math: [
    /\b(proof|theorem|lemma|corollary|axiom|derive|equation)\b/i,
    /\b(integral|derivative|gradient|matrix|vector|eigenvalue|tensor)\b/i,
    /\b(probability|statistics|distribution|variance|entropy|bayes)\b/i,
    /\b(calculate|compute|solve|evaluate|simplify|minimize|maximize)\b/i,
    /\b(convergence|divergence|complexity|big.?o|time complexity)\b/i,
    /[\d]+\s*[+\-*/^=]\s*[\d]+/,  // arithmetic expressions
    /\b(sin|cos|tan|log|exp|sqrt|sigma|pi|infinity)\b/i,
  ],

  reasoning: [
    /\b(analyze|analyse|compare|contrast|evaluate|critique|assess)\b/i,
    /\b(trade.?offs?|pros and cons|advantages|disadvantages|implications)\b/i,
    /\b(why|because|therefore|thus|hence|consequently|reasoning)\b/i,
    /\b(architect|design|strategy|framework|approach|methodology)\b/i,
    /\b(consensus|distributed|concurrent|parallel|fault.?toleran)\b/i,
    /\b(security|threat|vulnerability|attack|defense|mitigation)\b/i,
    /\b(production|scale|enterprise|critical|compliance|audit)\b/i,
    /\b(latency|throughput|bandwidth|scalability|performance|overhead)\b/i,
    /\b(philosophical|ethical|moral|principl|theory|argument)\b/i,
  ],

  creative: [
    /\b(write|generate|create|compose|craft|draft|story|poem|essay)\b/i,
    /\b(fiction|narrative|character|plot|setting|dialogue|scene)\b/i,
    /\b(imagine|invent|brainstorm|creative|original|novel idea)\b/i,
    /\b(song|lyrics|script|screenplay|blog|article|marketing)\b/i,
    /\b(tone|style|voice|audience|persuasive|engaging)\b/i,
  ],

  factual: [
    /^(what|who|when|where|which|how many|how much|is|are|was|were|does|do|did)\b/i,
    /\b(define|definition|explain|describe|list|name|what is)\b/i,
    /\b(history|origin|founded|invented|discovered|born|died)\b/i,
    /\b(capital|population|language|currency|country|continent)\b/i,
  ],

  general: [],
};

// Weight by how discriminating each pattern is
const DOMAIN_WEIGHTS: Record<Domain, number> = {
  code:      1.0,
  math:      1.1,
  reasoning: 0.9,  // was 0.8 — bumped since reasoning vocab overlaps with code (tcp/udp etc)
  creative:  1.2,
  factual:   0.9,
  general:   0.0,
};

// ─── Detector ────────────────────────────────────────────────────────────────

export class DomainDetector {
  detect(prompt: string): DomainScore {
    const lower = prompt.toLowerCase();
    const scores: Record<Domain, number> = {
      code: 0, math: 0, reasoning: 0, creative: 0, factual: 0, general: 0,
    };

    for (const [domain, patterns] of Object.entries(PATTERNS) as [Domain, typeof PATTERNS[Domain]][]) {
      if (patterns.length === 0) continue;
      let hits = 0;
      for (const pattern of patterns) {
        if (pattern instanceof RegExp) {
          if (pattern.test(prompt)) hits++;
        } else {
          if (lower.includes(pattern.toLowerCase())) hits++;
        }
      }
      scores[domain] = (hits / patterns.length) * DOMAIN_WEIGHTS[domain];
    }

    // Factual: if query is short and starts with a question word, boost it
    const words = prompt.trim().split(/\s+/);
    if (words.length <= 10 && /^(what|who|when|where|which|how)/i.test(prompt)) {
      scores.factual += 0.4;
    }

    // Find winner
    let topDomain: Domain = 'general';
    let topScore = 0;
    for (const [d, s] of Object.entries(scores) as [Domain, number][]) {
      if (d === 'general') continue;
      if (s > topScore) { topScore = s; topDomain = d; }
    }

    // Confidence: how much the winner beats second place
    const sorted = Object.values(scores).sort((a, b) => b - a);
    const margin = sorted[0] - (sorted[1] ?? 0);
    const confidence = topScore === 0 ? 0 : Math.min(1, 0.3 + margin * 2);

    return {
      domain: topScore > 0.1 ? topDomain : 'general',
      confidence,
      scores,
    };
  }
}
