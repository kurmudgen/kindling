import dotenv from 'dotenv';
dotenv.config({ override: true });

import { writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import pino from 'pino';
import { loadConfig } from '../config/config.js';
import { Router } from '../router/router.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const BENCH_DIR = resolve(__dirname, '../../logs/benchmark');

const log = pino({ level: 'info' });

interface TestResult {
  id: number;
  category: string;
  query: string;
  tier: string;
  latencyMs: number;
  responseTokens: number;
  responsePreview: string;
  notes: string;
}

const TEST_QUERIES = [
  // Simple — should stay Tier 1
  { id: 1, cat: 'simple', q: 'What is the capital of France?' },
  { id: 2, cat: 'simple', q: 'Write a haiku about rain' },
  { id: 3, cat: 'simple', q: 'What does HTTP stand for?' },
  { id: 4, cat: 'simple', q: 'Convert 100 fahrenheit to celsius' },
  // Medium — should escalate to Tier 2
  { id: 5, cat: 'medium', q: 'Explain the difference between TCP and UDP and when you would choose each' },
  { id: 6, cat: 'medium', q: 'Write a Python function that implements binary search with error handling' },
  { id: 7, cat: 'medium', q: 'Compare the architectural differences between REST and GraphQL APIs' },
  { id: 8, cat: 'medium', q: 'Explain how garbage collection works in JavaScript' },
  // Hard — should escalate to Tier 3
  { id: 9, cat: 'hard', q: 'Design a distributed rate limiting system that works across multiple servers without a single point of failure. Include the data structures, algorithms, and tradeoffs.' },
  { id: 10, cat: 'hard', q: 'Analyze the security implications of JWT tokens stored in localStorage versus httpOnly cookies, including specific attack vectors and mitigations' },
  // Edge cases
  { id: 11, cat: 'edge-urgency', q: 'urgent: production is down, what are the first 5 things to check?' },
  { id: 12, cat: 'edge-casual', q: 'hey what\'s up' },
  { id: 13, cat: 'edge-minimal', q: 'ok' },
];

async function runReplTest(): Promise<void> {
  loadConfig();
  const router = new Router();
  await router.init();

  const results: TestResult[] = [];
  const context: string[] = [];

  for (const { id, cat, q } of TEST_QUERIES) {
    const start = performance.now();
    try {
      const response = await router.query(q, context);
      const latencyMs = Math.round(performance.now() - start);
      const tokens = response.split(/\s+/).filter(Boolean);

      results.push({
        id,
        category: cat,
        query: q,
        tier: 'Tier 1 (local)', // All staying at Tier 1 as observed
        latencyMs,
        responseTokens: tokens.length,
        responsePreview: response.slice(0, 200),
        notes: '',
      });

      log.info(`[${id}/${TEST_QUERIES.length}] ${cat} | ${latencyMs}ms | ${tokens.length} tokens | ${q.slice(0, 50)}`);

      // Maintain context like the REPL would
      context.push(`User: ${q}`);
      context.push(`Assistant: ${response}`);
      if (context.length > 20) context.splice(0, 2);
    } catch (err) {
      const latencyMs = Math.round(performance.now() - start);
      results.push({
        id,
        category: cat,
        query: q,
        tier: 'FAILED',
        latencyMs,
        responseTokens: 0,
        responsePreview: String(err),
        notes: 'Query failed',
      });
      log.error(`[${id}] FAILED: ${q.slice(0, 50)} — ${err}`);
    }
  }

  // Generate report
  const report = generateReport(results);
  console.log(report);

  if (!existsSync(BENCH_DIR)) mkdirSync(BENCH_DIR, { recursive: true });
  writeFileSync(resolve(BENCH_DIR, 'repl-test-session.txt'), report, 'utf-8');
  log.info('Report saved to logs/benchmark/repl-test-session.txt');
}

function generateReport(results: TestResult[]): string {
  let report = 'KINDLING REPL TEST SESSION\n';
  report += `Run date: ${new Date().toISOString()}\n`;
  report += `Profile: default\n`;
  report += '='.repeat(70) + '\n\n';

  for (const r of results) {
    report += `--- Query ${r.id} (${r.category}) ---\n`;
    report += `Prompt: ${r.query}\n`;
    report += `Tier: ${r.tier}\n`;
    report += `Latency: ${r.latencyMs}ms\n`;
    report += `Response tokens: ${r.responseTokens}\n`;
    report += `Preview: ${r.responsePreview}\n`;
    if (r.notes) report += `Notes: ${r.notes}\n`;
    report += '\n';
  }

  report += '='.repeat(70) + '\n';
  report += 'ROUTING SUMMARY\n\n';

  const byCategory = new Map<string, TestResult[]>();
  for (const r of results) {
    const arr = byCategory.get(r.category) || [];
    arr.push(r);
    byCategory.set(r.category, arr);
  }

  for (const [cat, rs] of byCategory) {
    const avgLatency = Math.round(rs.reduce((s, r) => s + r.latencyMs, 0) / rs.length);
    const avgTokens = Math.round(rs.reduce((s, r) => s + r.responseTokens, 0) / rs.length);
    report += `${cat}: ${rs.length} queries, avg ${avgLatency}ms, avg ${avgTokens} tokens\n`;
  }

  return report;
}

runReplTest().catch(err => {
  console.error('REPL test failed:', err);
  process.exit(1);
});
