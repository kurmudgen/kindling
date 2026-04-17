/**
 * Continuous Training Data Collection Loop — Phase 6
 *
 * Runs hard + mixed + simple queries in rotating batches, pausing between
 * rounds to let shadow evals complete. Runs indefinitely until SIGINT.
 *
 * Designed for overnight data collection — safe, rate-limited, cheap.
 * Shadow eval fires every 5th query → ~6 Haiku calls per 30-query round.
 * Estimated cost: ~$0.01/hour at default pace.
 *
 * Usage:
 *   npx tsx src/tools/collect-loop.ts
 *   KINDLING_LOOP_PAUSE_MS=120000 npx tsx src/tools/collect-loop.ts
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { writeFileSync, readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import pino from 'pino';
import { loadConfig } from '../config/config.js';
import { Router } from '../router/router.js';
import { getTrainingStats } from '../shadow/training-store.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const LOOP_STATE_FILE = resolve(__dirname, '../../logs/collect-loop-state.json');

const log = pino({ level: 'info' });

// Pause between rounds — default 3 minutes (lets Ollama cool, shadow evals flush)
const PAUSE_MS = parseInt(process.env.KINDLING_LOOP_PAUSE_MS ?? '180000', 10);

// ─── Query corpus ───────────────────────────────────────────────────────────

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
  'Explain the mathematical proof behind Byzantine fault tolerance in distributed systems and design a practical BFT consensus algorithm.',
  'Design a multi-tenant SaaS platform with strict data isolation, per-tenant model customization, and SOC2 compliance across 10,000 tenants.',
  'Architect a compiler backend for a new programming language targeting WebAssembly with SIMD optimizations and garbage collection.',
  'Design a real-time recommendation engine serving personalized content to 50M users with sub-50ms latency using collaborative filtering and contextual bandits.',
  'Analyze the trade-offs between CRDT-based and OT-based conflict resolution for distributed collaborative applications at scale.',
];

const MIXED_QUERIES = [
  'What is a load balancer? Then explain how to design one that handles 1M requests per second with geographic failover.',
  'Define recursion. Now implement a production-grade recursive descent parser with error recovery for a SQL-like language.',
  'What is encryption? Explain it simply, then design a key management system for a financial institution.',
  'Name three sorting algorithms. Then analyze the mathematical proof of the lower bound for comparison-based sorting.',
  'What is an API? Then architect a comprehensive API gateway with rate limiting, auth, caching, and circuit breaking.',
  'What is a database index? Then explain how to design a multi-tenant index strategy for a petabyte-scale data warehouse.',
  'Define gradient descent. Then design an adaptive learning rate scheduler for training a 70B parameter language model.',
  'What is a mutex? Then design a lock-free concurrent data structure for high-frequency trading order books.',
];

const SIMPLE_QUERIES = [
  'What is the capital of France?',
  'What does HTTP stand for?',
  'What is the difference between RAM and ROM?',
  'Name the planets in our solar system.',
  'What is a hash function?',
  'What does CSS stand for?',
  'What is the boiling point of water?',
  'What is a linked list?',
  'Who wrote Romeo and Juliet?',
  'What is the speed of light?',
];

const CODE_QUERIES = [
  'Write a Python function that implements a binary search tree with insert, search, and delete operations.',
  'Implement a TypeScript generic LRU cache with O(1) get and put operations.',
  'Write a Rust function that safely parses a CSV file with quoted fields and escaped characters.',
  'Implement a concurrent worker pool in Go that gracefully handles panics and context cancellation.',
  'Write a React hook that debounces an async function and handles race conditions between calls.',
  'Implement a SQL query builder in TypeScript that prevents injection and supports nested conditions.',
  'Write a Python decorator that implements exponential backoff with jitter for API calls.',
  'Implement a topological sort with cycle detection in JavaScript using DFS.',
];

// ─── State tracking ──────────────────────────────────────────────────────────

interface LoopState {
  startedAt: string;
  roundsCompleted: number;
  queriesTotal: number;
  failuresTotal: number;
  examplesAtStart: number;
  lastRoundAt: string;
}

function loadState(examplesAtStart: number): LoopState {
  try {
    if (existsSync(LOOP_STATE_FILE)) {
      return JSON.parse(readFileSync(LOOP_STATE_FILE, 'utf-8')) as LoopState;
    }
  } catch { /* ignore */ }
  return {
    startedAt: new Date().toISOString(),
    roundsCompleted: 0,
    queriesTotal: 0,
    failuresTotal: 0,
    examplesAtStart,
    lastRoundAt: new Date().toISOString(),
  };
}

function saveState(state: LoopState): void {
  try {
    writeFileSync(LOOP_STATE_FILE, JSON.stringify(state, null, 2), 'utf-8');
  } catch { /* ignore */ }
}

// ─── Query runner ────────────────────────────────────────────────────────────

function estimateCoherence(text: string): number {
  if (!text || text.length === 0) return 0;
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length === 0) return 0;
  const hasSentences = /[.!?]/.test(text) ? 0.3 : 0;
  const uniqueRatio = new Set(words.map(w => w.toLowerCase())).size / words.length;
  return hasSentences + Math.min(0.4, uniqueRatio * 0.5) + Math.min(0.3, words.length / 100);
}

async function runQuery(
  router: Router,
  query: string,
  category: string,
  idx: number,
  total: number
): Promise<{ success: boolean; tier: number }> {
  try {
    const start = performance.now();
    const result = await router.queryDetailed(query);
    const latency = Math.round(performance.now() - start);
    log.info(
      { tier: result.tier, escalated: result.escalated, latencyMs: latency, metaAction: result.metaAction },
      `[${idx}/${total}] [${category}] ${query.slice(0, 55)}...`
    );
    return { success: true, tier: result.tier };
  } catch (err) {
    log.warn({ err }, `[${idx}/${total}] [${category}] FAILED: ${query.slice(0, 55)}`);
    return { success: false, tier: 0 };
  }
}

// ─── Round builder ────────────────────────────────────────────────────────────

/** Build a balanced round: 4 hard, 2 mixed, 3 simple, 3 code — rotated by round number */
function buildRound(roundNum: number): Array<{ query: string; category: string }> {
  const offset = roundNum % 4; // rotate starting position each round
  const pick = <T>(arr: T[], count: number, start: number): T[] => {
    const result: T[] = [];
    for (let i = 0; i < count; i++) result.push(arr[(start + i) % arr.length]);
    return result;
  };

  return [
    ...pick(HARD_QUERIES, 5, offset * 5).map(q => ({ query: q, category: 'hard' })),
    ...pick(MIXED_QUERIES, 3, offset * 3).map(q => ({ query: q, category: 'mixed' })),
    ...pick(SIMPLE_QUERIES, 3, offset * 3).map(q => ({ query: q, category: 'simple' })),
    ...pick(CODE_QUERIES, 3, offset * 3).map(q => ({ query: q, category: 'code' })),
  ];
}

// ─── Main loop ───────────────────────────────────────────────────────────────

async function main() {
  console.log('=== Kindling Training Data Collection Loop ===');
  console.log(`Pause between rounds: ${PAUSE_MS / 1000}s`);
  console.log('Press Ctrl+C to stop gracefully.\n');

  loadConfig();

  const initialStats = await getTrainingStats();
  const state = loadState(initialStats.totalExamples);
  console.log(`Examples at start: ${initialStats.totalExamples} | Rounds so far: ${state.roundsCompleted}\n`);

  const router = new Router();
  await router.init();

  let running = true;
  process.on('SIGINT', () => {
    console.log('\n\nSIGINT received — finishing current round then stopping...');
    running = false;
  });

  while (running) {
    const round = state.roundsCompleted + 1;
    const queries = buildRound(round);
    console.log(`\n--- Round ${round} — ${queries.length} queries ---`);

    let roundFailed = 0;
    for (let i = 0; i < queries.length; i++) {
      if (!running && i > 0) break; // mid-round interrupt after first query
      const { query, category } = queries[i];
      const result = await runQuery(router, query, category, i + 1, queries.length);
      if (!result.success) roundFailed++;
      // 1s between queries — lets shadow eval API calls overlap naturally
      if (i < queries.length - 1) await new Promise(r => setTimeout(r, 1000));
    }

    state.roundsCompleted++;
    state.queriesTotal += queries.length;
    state.failuresTotal += roundFailed;
    state.lastRoundAt = new Date().toISOString();

    const stats = await getTrainingStats();
    const newExamples = stats.totalExamples - state.examplesAtStart;
    console.log(`\nRound ${round} done — failures: ${roundFailed}/${queries.length}`);
    console.log(`Training examples: ${stats.totalExamples} total (+${newExamples} since loop start)`);
    saveState(state);

    if (!running) break;

    console.log(`Pausing ${PAUSE_MS / 1000}s before next round...`);
    await new Promise(r => setTimeout(r, PAUSE_MS));
  }

  const finalStats = await getTrainingStats();
  console.log(`\n=== Loop stopped after ${state.roundsCompleted} rounds ===`);
  console.log(`Total queries: ${state.queriesTotal} | Failures: ${state.failuresTotal}`);
  console.log(`Training examples: ${finalStats.totalExamples} (started at ${state.examplesAtStart})`);
  console.log(`Net new: +${finalStats.totalExamples - state.examplesAtStart}`);
  process.exit(0);
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
