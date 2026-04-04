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

  // Complexity: keyword density + length + question marks
  const complexityHits = countMatches(lower, COMPLEXITY_MARKERS);
  const lengthFactor = Math.min(1, prompt.length / 500);
  const questionMarks = (prompt.match(/\?/g) || []).length;
  const ambiguityFactor = Math.min(1, questionMarks / 3);
  const complexity = Math.min(1, (complexityHits / 3 + lengthFactor + ambiguityFactor) / 3 * 2);

  // Stakes: keyword density
  const stakesHits = countMatches(lower, STAKES_INDICATORS);
  const stakes = Math.min(1, stakesHits / 2);

  // Weighted composite — complexity weighted highest since it most directly
  // predicts whether a small model can handle the query
  const composite = urgency * 0.25 + complexity * 0.50 + stakes * 0.25;

  return { urgency, complexity, stakes, composite };
}
