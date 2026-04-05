/**
 * Warmup Bench Tool — Milestone 5b
 *
 * Measures the latency effect of concept pre-warming on Tier 1.
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { loadConfig, getConfig } from '../config/config.js';
import { Tier1 } from '../tiers/tier1.js';
import { ConceptPrewarmer } from '../sleep/prewarmer.js';
import type { TierQuery } from '../tiers/tier-interface.js';

const concept = process.argv[2] ?? 'JWT authentication';

function buildQuery(concept: string, variant: number): string {
  const variants = [
    `What is ${concept}?`,
    `Explain how ${concept} works.`,
    `What are the security implications of ${concept}?`,
  ];
  return variants[variant % variants.length];
}

async function main() {
  loadConfig();
  const tier1 = new Tier1();
  console.log(`Warming up Tier 1 model...`);
  await tier1.warmup();

  const valence = { urgency: 0, complexity: 0.2, stakes: 0, composite: 0.1 };

  console.log(`\nConcept: "${concept}"`);
  console.log('='.repeat(60));

  // Pre-warming latencies
  console.log('\n--- BEFORE WARMING ---');
  const beforeLatencies: number[] = [];
  for (let i = 0; i < 3; i++) {
    const q: TierQuery = {
      prompt: buildQuery(concept, i),
      context: [],
      maxTokens: 128,
      valenceScore: valence,
    };
    const result = await tier1.generate(q);
    beforeLatencies.push(result.latencyMs);
    console.log(`  Query ${i + 1}: ${Math.round(result.latencyMs)}ms (${result.tokens.length} tokens)`);
  }

  // Run warming
  console.log('\n--- WARMING CONCEPT ---');
  const prewarmer = new ConceptPrewarmer();
  const warmStart = performance.now();
  await prewarmer.prewarm([concept]);
  const warmMs = performance.now() - warmStart;
  console.log(`  Warming took: ${Math.round(warmMs)}ms`);

  // Post-warming latencies
  console.log('\n--- AFTER WARMING ---');
  const afterLatencies: number[] = [];
  for (let i = 0; i < 3; i++) {
    const q: TierQuery = {
      prompt: buildQuery(concept, i),
      context: [],
      maxTokens: 128,
      valenceScore: valence,
    };
    const result = await tier1.generate(q);
    afterLatencies.push(result.latencyMs);
    console.log(`  Query ${i + 1}: ${Math.round(result.latencyMs)}ms (${result.tokens.length} tokens)`);
  }

  // Summary
  const avgBefore = beforeLatencies.reduce((a, b) => a + b, 0) / beforeLatencies.length;
  const avgAfter = afterLatencies.reduce((a, b) => a + b, 0) / afterLatencies.length;
  const diff = avgAfter - avgBefore;
  const pctChange = (diff / avgBefore) * 100;

  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY');
  console.log(`  Avg before: ${Math.round(avgBefore)}ms`);
  console.log(`  Avg after:  ${Math.round(avgAfter)}ms`);
  console.log(`  Change:     ${diff > 0 ? '+' : ''}${Math.round(diff)}ms (${pctChange > 0 ? '+' : ''}${pctChange.toFixed(1)}%)`);
  console.log(`  Warming:    ${Math.round(warmMs)}ms`);

  if (pctChange < -5) {
    console.log(`\n  RESULT: Warming improved latency by ${Math.abs(pctChange).toFixed(1)}%`);
  } else if (pctChange > 5) {
    console.log(`\n  RESULT: Warming had no benefit (${pctChange.toFixed(1)}% slower — likely noise)`);
  } else {
    console.log(`\n  RESULT: No significant difference (within noise margin)`);
  }
}

main().catch(err => {
  console.error('Warmup bench failed:', err);
  process.exit(1);
});
