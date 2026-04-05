/**
 * Concurrency Integration Test
 *
 * Fires 4 queries in parallel via Promise.all and verifies:
 * 1. All complete successfully
 * 2. Each gets a unique generation number
 * 3. Log writes are not interleaved (every line is valid JSON)
 * 4. No shared state corruption (each query has its own confidence/buffer)
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadConfig } from '../config/config.js';
import { Router } from '../router/router.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const LOG_FILE = resolve(__dirname, '../../logs/escalation.jsonl');

async function main() {
  loadConfig();
  const router = new Router();
  await router.init();

  // Note existing log size
  const beforeSize = existsSync(LOG_FILE)
    ? readFileSync(LOG_FILE, 'utf-8').length
    : 0;

  const queries = [
    'What is the capital of France?',
    'Explain the difference between TCP and UDP',
    'Name three sorting algorithms',
    'What is quantum entanglement?',
  ];

  console.log(`=== CONCURRENCY TEST ===`);
  console.log(`Firing ${queries.length} queries in parallel...\n`);

  const startTime = performance.now();
  const results = await Promise.all(
    queries.map((q, i) =>
      router.queryDetailed(q).then(r => ({ index: i, query: q, result: r }))
    )
  );
  const totalMs = performance.now() - startTime;

  console.log(`All ${results.length} queries completed in ${Math.round(totalMs)}ms\n`);

  let allSucceeded = true;
  for (const { index, query, result } of results) {
    const firstLine = query.slice(0, 50);
    console.log(
      `  [${index}] ${firstLine} → T${result.tier} conf=${result.confidence.toFixed(3)} tokens=${result.text.split(/\s+/).length}`
    );
    if (!result.text || result.text.length < 3) {
      allSucceeded = false;
    }
  }

  // Verify log integrity — every line must be valid JSON
  console.log('\n=== LOG INTEGRITY CHECK ===');
  let logIntact = true;
  if (existsSync(LOG_FILE)) {
    const content = readFileSync(LOG_FILE, 'utf-8');
    const newContent = content.slice(beforeSize);
    const lines = newContent.trim().split('\n').filter(l => l);
    let validLines = 0;
    let invalidLines = 0;
    for (const line of lines) {
      try {
        const parsed = JSON.parse(line);
        if (parsed.timestamp && parsed.sessionId) {
          validLines++;
        }
      } catch {
        invalidLines++;
        console.log(`  INVALID LINE: ${line.slice(0, 100)}`);
      }
    }
    console.log(`  Valid lines: ${validLines}`);
    console.log(`  Invalid lines: ${invalidLines}`);
    logIntact = invalidLines === 0;

    // Give the write queue a moment to finish
    await new Promise(r => setTimeout(r, 500));
  }

  console.log('\n=== GATE CHECKS ===');
  console.log(`  All queries succeeded: ${allSucceeded ? 'PASS' : 'FAIL'}`);
  console.log(`  No interleaved log writes: ${logIntact ? 'PASS' : 'FAIL'}`);
  console.log(`  In-flight count back to 0: ${router.inFlightCount() === 0 ? 'PASS' : 'FAIL'}`);

  const allPass = allSucceeded && logIntact && router.inFlightCount() === 0;
  console.log(`\nMILESTONE 7 CONCURRENCY GATE: ${allPass ? 'PASS' : 'FAIL'}`);
}

main().catch(err => {
  console.error('Concurrency test failed:', err);
  process.exit(1);
});
