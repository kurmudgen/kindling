import type { ValenceScore } from '../tiers/tier-interface.js';

const URGENCY_KEYWORDS = [
  'urgent', 'asap', 'immediately', 'critical', 'emergency',
  'right now', 'time-sensitive', 'deadline', 'hurry', 'rush',
];

const COMPLEXITY_MARKERS = [
  'explain', 'analyze', 'compare', 'design', 'architect', 'implement',
  'refactor', 'optimize', 'debug', 'trace', 'evaluate', 'synthesize',
  'derive', 'prove', 'formalize', 'decompose',
];

const HIGH_COMPLEXITY_MARKERS = [
  'architect', 'design a distributed', 'design a comprehensive',
  'implement a production', 'analyze and compare',
  'tradeoffs', 'trade-offs', 'implications',
];

const STAKES_INDICATORS = [
  'production', 'security', 'medical', 'legal', 'financial', 'important',
  'compliance', 'regulatory', 'patient', 'safety', 'audit', 'sensitive',
  'confidential', 'hipaa', 'gdpr', 'pci', 'liability',
];

function countMatches(text: string, keywords: string[]): number {
  const lower = text.toLowerCase();
  return keywords.filter(k => lower.includes(k)).length;
}

export function scoreValence(prompt: string): ValenceScore {
  const lower = prompt.toLowerCase();

  // Urgency: keyword density
  const urgencyHits = countMatches(lower, URGENCY_KEYWORDS);
  const urgency = Math.min(1, urgencyHits / 2);

  // Complexity: multi-signal
  const complexityHits = countMatches(lower, COMPLEXITY_MARKERS);
  const highComplexityHits = countMatches(lower, HIGH_COMPLEXITY_MARKERS);
  const lengthFactor = Math.min(1, prompt.length / 300); // shorter divisor = more sensitive
  const questionMarks = (prompt.match(/\?/g) || []).length;
  const ambiguityFactor = Math.min(1, questionMarks / 3);

  // Count clauses/sentences as complexity proxy (commas, periods, semicolons)
  const clauseCount = (prompt.match(/[,;.]/g) || []).length;
  const clauseFactor = Math.min(1, clauseCount / 4);

  // Multi-part instructions ("then", "and then", "also", "including")
  const multiPartHits = countMatches(lower, ['then', 'also', 'including', 'furthermore', 'additionally', 'as well as']);
  const multiPartFactor = Math.min(1, multiPartHits / 2);

  const complexity = Math.min(1,
    (complexityHits / 2) * 0.30 +
    highComplexityHits * 0.25 +
    lengthFactor * 0.15 +
    ambiguityFactor * 0.10 +
    clauseFactor * 0.10 +
    multiPartFactor * 0.10
  );

  // Stakes: keyword density
  const stakesHits = countMatches(lower, STAKES_INDICATORS);
  const stakes = Math.min(1, stakesHits / 2);

  // Weighted composite
  const composite = urgency * 0.20 + complexity * 0.50 + stakes * 0.30;

  return { urgency, complexity, stakes, composite };
}
