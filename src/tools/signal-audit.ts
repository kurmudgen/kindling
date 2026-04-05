/**
 * Signal Audit Tool — Milestone 2
 *
 * Runs 6 test queries through Tier 1 only and prints raw escalation signals,
 * confidence scores, and whether escalation would fire.
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { loadConfig, getConfig } from '../config/config.js';
import { Tier1 } from '../tiers/tier1.js';
import { scoreValence } from '../router/valence.js';
import { ConfidenceAggregator } from '../router/confidence.js';
import type { TierQuery, EscalationSignals } from '../tiers/tier-interface.js';

const QUERIES = [
  { label: 'SIMPLE-1', q: 'What is 2 + 2?' },
  { label: 'SIMPLE-2', q: 'Name a color.' },
  { label: 'MEDIUM-1', q: 'Explain the difference between TCP and UDP.' },
  { label: 'MEDIUM-2', q: 'Design a simple caching strategy for a REST API.' },
  { label: 'HARD-1', q: 'Architect a production-grade real-time collaborative document editor with conflict resolution, supporting 10,000 concurrent users.' },
  { label: 'HARD-2', q: 'Design a comprehensive security audit framework for a financial services application handling PII and payment data under GDPR and PCI-DSS compliance requirements.' },
];

function fmtSig(s: EscalationSignals): string {
  return `spread=${s.tokenProbabilitySpread.toFixed(3)} velocity=${s.semanticVelocity.toFixed(3)} surprise=${s.surpriseScore.toFixed(3)} anomaly=${s.attentionAnomalyScore.toFixed(3)}`;
}

async function main() {
  loadConfig();
  const config = getConfig();
  const tier1 = new Tier1();

  console.log('Warming up Tier 1...');
  await tier1.warmup();

  const agg = new ConfidenceAggregator();
  const threshold = 1 - config.escalation.escalationThreshold;

  console.log(`\nEscalation threshold: confidence < ${threshold.toFixed(3)} triggers escalation`);
  console.log(`Weights: spread=${config.escalation.tokenProbabilitySpreadWeight} velocity=${config.escalation.semanticVelocityWeight} surprise=${config.escalation.surpriseScoreWeight} anomaly=${config.escalation.attentionAnomalyWeight}`);
  console.log('='.repeat(90));

  const results: Array<{ label: string; confidence: number; wouldEscalate: boolean; signals: EscalationSignals }> = [];

  for (const { label, q } of QUERIES) {
    const valence = scoreValence(q);
    const tierQuery: TierQuery = {
      prompt: q,
      context: [],
      maxTokens: 512,
      valenceScore: valence,
    };

    const response = await tier1.generate(tierQuery);
    const decision = agg.decide(1, response.escalationSignals, valence);
    agg.reset();

    results.push({
      label,
      confidence: response.confidence,
      wouldEscalate: decision.shouldEscalate,
      signals: response.escalationSignals,
    });

    const escLabel = decision.shouldEscalate ? 'YES ESCALATE' : 'stay T1';
    console.log(`\n[${label}] "${q.slice(0, 60)}${q.length > 60 ? '...' : ''}"`);
    console.log(`  Signals:    ${fmtSig(response.escalationSignals)}`);
    console.log(`  Confidence: ${response.confidence.toFixed(3)}  |  Escalate: ${escLabel}`);
    console.log(`  Valence:    composite=${valence.composite.toFixed(2)} complexity=${valence.complexity.toFixed(2)} stakes=${valence.stakes.toFixed(2)}`);
    console.log(`  Tokens:     ${response.tokens.length}  |  Logprobs: ${(response.metadata as any)?.hasLogprobs ? 'YES' : 'NO'}`);
  }

  // Summary analysis
  console.log('\n' + '='.repeat(90));
  console.log('SIGNAL DIFFERENTIATION ANALYSIS\n');

  const simples = results.filter(r => r.label.startsWith('SIMPLE'));
  const hards = results.filter(r => r.label.startsWith('HARD'));
  const mediums = results.filter(r => r.label.startsWith('MEDIUM'));

  const avgConf = (rs: typeof results) => rs.reduce((s, r) => s + r.confidence, 0) / rs.length;
  const avgSpread = (rs: typeof results) => rs.reduce((s, r) => s + r.signals.tokenProbabilitySpread, 0) / rs.length;

  console.log(`Simple avg confidence:  ${avgConf(simples).toFixed(3)}  avg spread: ${avgSpread(simples).toFixed(3)}`);
  console.log(`Medium avg confidence:  ${avgConf(mediums).toFixed(3)}  avg spread: ${avgSpread(mediums).toFixed(3)}`);
  console.log(`Hard   avg confidence:  ${avgConf(hards).toFixed(3)}  avg spread: ${avgSpread(hards).toFixed(3)}`);

  const simpleConfOk = simples.every(r => r.confidence > 0.75);
  const hardConfOk = hards.every(r => r.confidence < 0.80);
  const spreadDiff = avgSpread(hards) - avgSpread(simples);

  console.log(`\nGATE CHECK:`);
  console.log(`  Simple confidence > 0.75: ${simpleConfOk ? 'PASS' : 'FAIL'} (${simples.map(r => r.confidence.toFixed(3)).join(', ')})`);
  console.log(`  Hard confidence < 0.80:   ${hardConfOk ? 'PASS' : 'FAIL'} (${hards.map(r => r.confidence.toFixed(3)).join(', ')})`);
  console.log(`  Spread differentiation:   ${spreadDiff > 0.05 ? 'PASS' : 'FAIL'} (diff=${spreadDiff.toFixed(3)})`);

  const allPass = simpleConfOk && hardConfOk && spreadDiff > 0.05;
  console.log(`\n  MILESTONE 2 GATE: ${allPass ? 'PASS' : 'NEEDS ATTENTION'}`);
}

main().catch(err => {
  console.error('Signal audit failed:', err);
  process.exit(1);
});
