/**
 * Training Data Collection Tool — Phase 4
 *
 * Runs only the hard + mixed benchmark queries (Tier 2/3 territory) to
 * quickly accumulate positive-class training examples for the ML classifier.
 * Simple/medium queries already have plenty of Tier-1 (negative) examples.
 *
 * Usage:
 *   npx tsx src/tools/collect-training-data.ts
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import pino from 'pino';
import { loadConfig } from '../config/config.js';
import { Router } from '../router/router.js';

const log = pino({ level: 'info' });

const HARD_QUERIES = [
  'Architect a production-grade real-time collaborative document editor with conflict resolution, supporting 10,000 concurrent users.',
  'Design a comprehensive security audit framework for a financial services application handling PII and payment data under GDPR and PCI-DSS compliance requirements.',
  'Analyze and compare the theoretical foundations of transformer attention mechanisms versus state space models for sequence modeling, with implications for hardware optimization.',
  'Design a distributed consensus protocol for a multi-region database that maintains strong consistency while minimizing cross-region latency.',
  'Architect a machine learning pipeline for production fraud detection that handles concept drift, requires explainability, and processes 1M transactions per second.',
  'Design a comprehensive disaster recovery strategy for a critical healthcare system that must maintain 99.999% uptime across multiple failure domains.',
  'Analyze the security implications of implementing a zero-trust architecture in a legacy enterprise environment with 500+ microservices.',
  'Design a real-time stream processing system that handles backpressure, exactly-once semantics, and late-arriving data at petabyte scale.',
  'Architect a privacy-preserving federated learning system for medical imaging that complies with HIPAA while maintaining model accuracy.',
  'Design a comprehensive observability platform that correlates metrics, traces, and logs across a polyglot microservices architecture with 200+ services.',
];

const MIXED_QUERIES = [
  'What is a load balancer? Then explain how to design one that handles 1M requests per second with geographic failover.',
  'Define recursion. Now implement a production-grade recursive descent parser with error recovery for a SQL-like language.',
  'What is encryption? Explain it simply, then design a key management system for a financial institution.',
  'Name three sorting algorithms. Then analyze the mathematical proof of the lower bound for comparison-based sorting.',
  'What is an API? Then architect a comprehensive API gateway with rate limiting, auth, caching, and circuit breaking for a critical production system.',
];

function estimateCoherence(text: string): number {
  if (!text || text.length === 0) return 0;
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length === 0) return 0;
  const hasSentences = /[.!?]/.test(text) ? 0.3 : 0;
  const uniqueRatio = new Set(words.map(w => w.toLowerCase())).size / words.length;
  const diversityScore = Math.min(0.4, uniqueRatio * 0.5);
  const lengthScore = Math.min(0.3, words.length / 100);
  return hasSentences + diversityScore + lengthScore;
}

async function runQuery(router: Router, query: string, category: string, idx: number, total: number) {
  const start = performance.now();
  try {
    const result = await router.queryDetailed(query);
    const latency = Math.round(performance.now() - start);
    const coherence = estimateCoherence(result.text);
    log.info(
      {
        tier: result.tier,
        escalated: result.escalated,
        confidence: result.confidence.toFixed(3),
        latencyMs: latency,
        tokens: result.text.split(/\s+/).length,
        metaAction: result.metaAction,
      },
      `[${idx}/${total}] [${category}] ${query.slice(0, 55)}...`
    );
    return { success: true, tier: result.tier };
  } catch (err) {
    log.error({ err }, `[${idx}/${total}] [${category}] FAILED: ${query.slice(0, 55)}`);
    return { success: false, tier: 0 };
  }
}

async function main() {
  console.log('=== Training Data Collection — Hard + Mixed Queries ===\n');
  console.log('These queries route to Tier 2/3, producing positive-class training examples.\n');

  loadConfig();
  const router = new Router();
  await router.init();

  const allQueries = [
    ...HARD_QUERIES.map(q => ({ query: q, category: 'hard' })),
    ...MIXED_QUERIES.map(q => ({ query: q, category: 'mixed' })),
  ];

  const total = allQueries.length;
  let tier2Plus = 0;
  let failed = 0;

  for (let i = 0; i < allQueries.length; i++) {
    const { query, category } = allQueries[i];
    const result = await runQuery(router, query, category, i + 1, total);
    if (!result.success) failed++;
    if (result.tier > 1) tier2Plus++;
    // Small pause between queries to let shadow evals complete
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  const shadowStats = router.getShadowStats?.() ?? {};
  console.log('\n=== Collection Complete ===');
  console.log(`Queries: ${total} | Failed: ${failed} | Tier2+: ${tier2Plus}`);
  if (Object.keys(shadowStats).length > 0) {
    console.log(`Shadow: ${JSON.stringify(shadowStats)}`);
  }
  console.log('\nCheck training data: wc -l logs/shadow/training.jsonl');
  console.log('Train: python scripts/train-classifier.py');
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
