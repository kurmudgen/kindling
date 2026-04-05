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
  expectedTier: string;
  query: string;
  actualTier: number;
  escalated: boolean;
  escalationPath: number[];
  confidence: number;
  latencyMs: number;
  responseTokens: number;
  responsePreview: string;
  tierCorrect: boolean;
}

const TEST_QUERIES = [
  { id: 1, cat: 'simple', expected: 'Tier 1', q: 'What is the capital of France?' },
  { id: 2, cat: 'simple', expected: 'Tier 1', q: 'Write a haiku about rain' },
  { id: 3, cat: 'simple', expected: 'Tier 1', q: 'What does HTTP stand for?' },
  { id: 4, cat: 'simple', expected: 'Tier 1', q: 'Convert 100 fahrenheit to celsius' },
  { id: 5, cat: 'medium', expected: 'Tier 2', q: 'Explain the difference between TCP and UDP and when you would choose each' },
  { id: 6, cat: 'medium', expected: 'Tier 2', q: 'Write a Python function that implements binary search with error handling' },
  { id: 7, cat: 'medium', expected: 'Tier 2', q: 'Compare the architectural differences between REST and GraphQL APIs' },
  { id: 8, cat: 'medium', expected: 'Tier 2', q: 'Explain how garbage collection works in JavaScript' },
  { id: 9, cat: 'hard', expected: 'Tier 3', q: 'Design a distributed rate limiting system that works across multiple servers without a single point of failure. Include the data structures, algorithms, and tradeoffs.' },
  { id: 10, cat: 'hard', expected: 'Tier 3', q: 'Analyze the security implications of JWT tokens stored in localStorage versus httpOnly cookies, including specific attack vectors and mitigations' },
  { id: 11, cat: 'edge-urgency', expected: 'Tier 2+', q: 'urgent: production is down, what are the first 5 things to check?' },
  { id: 12, cat: 'edge-casual', expected: 'Tier 1', q: 'hey what\'s up' },
  { id: 13, cat: 'edge-minimal', expected: 'Tier 1', q: 'ok' },
];

async function runReplTest(): Promise<void> {
  loadConfig();
  const router = new Router();
  await router.init();

  const results: TestResult[] = [];
  const context: string[] = [];

  for (const { id, cat, expected, q } of TEST_QUERIES) {
    try {
      const qr = await router.queryDetailed(q, context);
      const tokens = qr.text.split(/\s+/).filter(Boolean);

      // Determine if the tier choice was "correct" based on expectations
      const tierCorrect = evaluateTierCorrectness(cat, expected, qr.tier);

      results.push({
        id,
        category: cat,
        expectedTier: expected,
        query: q,
        actualTier: qr.tier,
        escalated: qr.escalated,
        escalationPath: qr.escalationPath,
        confidence: +qr.confidence.toFixed(3),
        latencyMs: Math.round(qr.latencyMs),
        responseTokens: tokens.length,
        responsePreview: qr.text.slice(0, 200),
        tierCorrect,
      });

      const marker = tierCorrect ? 'OK' : 'MISMATCH';
      log.info(
        `[${id}/${TEST_QUERIES.length}] ${marker} | ${cat} | Tier ${qr.tier} (expected ${expected}) | ${Math.round(qr.latencyMs)}ms | conf=${qr.confidence.toFixed(3)} | ${q.slice(0, 40)}`
      );

      context.push(`User: ${q}`);
      context.push(`Assistant: ${qr.text}`);
      if (context.length > 20) context.splice(0, 2);
    } catch (err) {
      results.push({
        id,
        category: cat,
        expectedTier: expected,
        query: q,
        actualTier: 0 as any,
        escalated: false,
        escalationPath: [],
        confidence: 0,
        latencyMs: 0,
        responseTokens: 0,
        responsePreview: String(err),
        tierCorrect: false,
      });
      log.error(`[${id}] FAILED: ${q.slice(0, 50)} — ${err}`);
    }
  }

  const report = generateReport(results);
  console.log(report);

  if (!existsSync(BENCH_DIR)) mkdirSync(BENCH_DIR, { recursive: true });
  writeFileSync(resolve(BENCH_DIR, 'repl-test-session-phase2.txt'), report, 'utf-8');
  log.info('Report saved to logs/benchmark/repl-test-session-phase2.txt');
}

function evaluateTierCorrectness(cat: string, expected: string, actual: number): boolean {
  if (cat === 'simple' || cat === 'edge-casual' || cat === 'edge-minimal') {
    return actual === 1;
  }
  if (cat === 'medium') {
    return actual >= 2; // Tier 2 or higher is correct
  }
  if (cat === 'hard') {
    return actual >= 2; // Tier 2 or 3 is correct (Tier 3 may not be available)
  }
  if (cat === 'edge-urgency') {
    return actual >= 2; // Should escalate
  }
  return true;
}

function generateReport(results: TestResult[]): string {
  let report = 'KINDLING REPL TEST SESSION — PHASE 2\n';
  report += `Run date: ${new Date().toISOString()}\n`;
  report += `Profile: default\n`;
  report += '='.repeat(70) + '\n\n';

  const correct = results.filter(r => r.tierCorrect).length;
  report += `ROUTING ACCURACY: ${correct}/${results.length} (${((correct / results.length) * 100).toFixed(0)}%)\n\n`;

  for (const r of results) {
    const marker = r.tierCorrect ? 'OK' : 'MISMATCH';
    report += `--- Query ${r.id} (${r.category}) [${marker}] ---\n`;
    report += `Prompt: ${r.query}\n`;
    report += `Expected: ${r.expectedTier} | Actual: Tier ${r.actualTier}\n`;
    report += `Escalated: ${r.escalated} | Path: ${r.escalationPath.join(' → ')}\n`;
    report += `Confidence: ${r.confidence} | Latency: ${r.latencyMs}ms\n`;
    report += `Response tokens: ${r.responseTokens}\n`;
    report += `Preview: ${r.responsePreview}\n\n`;
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
    const tiers = rs.map(r => `T${r.actualTier}`).join(', ');
    const accuracy = rs.filter(r => r.tierCorrect).length;
    report += `${cat}: ${rs.length} queries | avg ${avgLatency}ms | avg ${avgTokens} tokens | tiers: [${tiers}] | accuracy: ${accuracy}/${rs.length}\n`;
  }

  return report;
}

runReplTest().catch(err => {
  console.error('REPL test failed:', err);
  process.exit(1);
});
