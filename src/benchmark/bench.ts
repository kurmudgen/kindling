import { writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
dotenv.config({ override: true });

import pino from 'pino';
import { loadConfig } from '../config/config.js';
import { Router } from '../router/router.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const BENCH_DIR = resolve(__dirname, '../../logs/benchmark');

const log = pino({ level: 'info' });

interface BenchmarkResult {
  query: string;
  category: string;
  latencyMs: number;
  responseLength: number;
  coherenceScore: number;
  success: boolean;
  error?: string;
}

// Test queries organized by expected tier
const SIMPLE_QUERIES = [
  'What is 2 + 2?',
  'Name a color.',
  'What day comes after Monday?',
  'Say hello.',
  'What is the capital of France?',
  'Is water wet?',
  'Name a fruit.',
  'What is 10 times 5?',
  'Spell the word cat.',
  'What color is the sky?',
  'Name an animal.',
  'What is the opposite of hot?',
  'Count to 5.',
  'What is a dog?',
  'Name a month.',
  'What shape has 4 sides?',
  'Is fire hot or cold?',
  'Name a planet.',
  'What comes after 7?',
  'What is the first letter of the alphabet?',
];

const MEDIUM_QUERIES = [
  'Explain the difference between TCP and UDP.',
  'Compare Python and JavaScript for web development.',
  'Analyze the pros and cons of microservices architecture.',
  'Design a simple caching strategy for a REST API.',
  'Explain how garbage collection works in modern languages.',
  'Compare SQL and NoSQL databases for a social media app.',
  'Explain the CAP theorem in distributed systems.',
  'Analyze the tradeoffs between REST and GraphQL.',
  'Design a rate limiting algorithm for an API gateway.',
  'Explain how TLS handshake works step by step.',
  'Compare container orchestration approaches: Kubernetes vs Nomad.',
  'Analyze the benefits and risks of event-driven architecture.',
  'Explain the difference between authentication and authorization.',
  'Design a simple pub/sub message queue.',
  'Compare monorepo vs polyrepo strategies.',
  'Explain how DNS resolution works from browser to server.',
  'Analyze the tradeoffs of eventual consistency.',
  'Design an API versioning strategy.',
  'Explain how WebSockets differ from HTTP long polling.',
  'Compare load balancing algorithms for web servers.',
];

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

  // Simple heuristics: sentence structure, vocabulary diversity, length adequacy
  const hasSentences = /[.!?]/.test(text) ? 0.3 : 0;
  const uniqueRatio = new Set(words.map(w => w.toLowerCase())).size / words.length;
  const diversityScore = Math.min(0.4, uniqueRatio * 0.5);
  const lengthScore = Math.min(0.3, words.length / 100);

  return Math.min(1, hasSentences + diversityScore + lengthScore);
}

async function runBenchmark(): Promise<void> {
  loadConfig();
  const router = new Router();
  await router.init();

  const results: BenchmarkResult[] = [];
  const categories = [
    { name: 'simple', queries: SIMPLE_QUERIES },
    { name: 'medium', queries: MEDIUM_QUERIES },
    { name: 'hard', queries: HARD_QUERIES },
    { name: 'mixed', queries: MIXED_QUERIES },
  ];

  for (const { name, queries } of categories) {
    log.info(`\n--- ${name.toUpperCase()} QUERIES (${queries.length}) ---`);

    for (const q of queries) {
      const start = performance.now();
      try {
        const response = await router.query(q);
        const latencyMs = performance.now() - start;
        const result: BenchmarkResult = {
          query: q.slice(0, 80),
          category: name,
          latencyMs: Math.round(latencyMs),
          responseLength: response.split(/\s+/).length,
          coherenceScore: estimateCoherence(response),
          success: true,
        };
        results.push(result);
        log.info(
          { latency: result.latencyMs, tokens: result.responseLength, coherence: result.coherenceScore.toFixed(2) },
          `[${name}] ${q.slice(0, 50)}...`
        );
      } catch (err) {
        const latencyMs = performance.now() - start;
        results.push({
          query: q.slice(0, 80),
          category: name,
          latencyMs: Math.round(latencyMs),
          responseLength: 0,
          coherenceScore: 0,
          success: false,
          error: String(err),
        });
        log.error({ err }, `[${name}] FAILED: ${q.slice(0, 50)}...`);
      }
    }
  }

  // Summary
  const summary = {
    timestamp: new Date().toISOString(),
    totalQueries: results.length,
    successful: results.filter(r => r.success).length,
    failed: results.filter(r => !r.success).length,
    byCategory: {} as Record<string, {
      count: number;
      avgLatencyMs: number;
      avgTokens: number;
      avgCoherence: number;
      successRate: number;
    }>,
    results,
  };

  for (const cat of ['simple', 'medium', 'hard', 'mixed']) {
    const catResults = results.filter(r => r.category === cat);
    const successful = catResults.filter(r => r.success);
    summary.byCategory[cat] = {
      count: catResults.length,
      avgLatencyMs: Math.round(successful.reduce((s, r) => s + r.latencyMs, 0) / (successful.length || 1)),
      avgTokens: Math.round(successful.reduce((s, r) => s + r.responseLength, 0) / (successful.length || 1)),
      avgCoherence: +(successful.reduce((s, r) => s + r.coherenceScore, 0) / (successful.length || 1)).toFixed(3),
      successRate: +(successful.length / (catResults.length || 1)).toFixed(3),
    };
  }

  // Save
  if (!existsSync(BENCH_DIR)) mkdirSync(BENCH_DIR, { recursive: true });
  const filename = new Date().toISOString().slice(0, 10) + '.json';
  writeFileSync(resolve(BENCH_DIR, filename), JSON.stringify(summary, null, 2), 'utf-8');

  // Console summary
  console.log('\n=== BENCHMARK SUMMARY ===');
  console.log(`Total: ${summary.totalQueries} | Pass: ${summary.successful} | Fail: ${summary.failed}`);
  console.table(summary.byCategory);
  console.log(`Results saved to logs/benchmark/${filename}`);
}

runBenchmark().catch(err => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
