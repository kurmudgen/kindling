/**
 * Dream Task Verification
 *
 * Runs 10 queries, sets a very short idle threshold, and verifies
 * the dream task fires automatically without /sleep.
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { writeFileSync, readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CONFIG_DIR = resolve(__dirname, '../../config');
const LOGS_DIR = resolve(__dirname, '../../logs');
const DREAM_LOG = resolve(LOGS_DIR, 'dream.jsonl');

async function main() {
  // Mutate default config to enable fast dream trigger
  const defaultPath = resolve(CONFIG_DIR, 'default.json');
  const originalConfig = readFileSync(defaultPath, 'utf-8');

  // Note dream log state before test
  const dreamLogBefore = existsSync(DREAM_LOG) ? readFileSync(DREAM_LOG, 'utf-8') : '';

  try {
    const config = JSON.parse(originalConfig);
    config.sleep.idleThresholdMinutes = 0.1; // 6 seconds
    config.sleep.autoSleepMinQueries = 5;
    writeFileSync(defaultPath, JSON.stringify(config, null, 2), 'utf-8');

    const { loadConfig } = await import('../config/config.js');
    loadConfig();

    const { Router } = await import('../router/router.js');
    const { SleepAnalyst } = await import('../sleep/analyst.js');

    const router = new Router();
    await router.init();

    if (!process.env.ANTHROPIC_API_KEY) {
      console.log('SKIP: No ANTHROPIC_API_KEY — dream task requires API access');
      console.log('MILESTONE 6 DREAM TASK GATE: SKIPPED (no API key)');
      return;
    }

    const analyst = new SleepAnalyst();
    let notificationReceived = false;
    analyst.onNotification((msg) => {
      console.log(`  NOTIFICATION: ${msg}`);
      notificationReceived = true;
    });
    // Use short check interval to see it fire quickly
    analyst.startIdleMonitor(2000);

    console.log('=== DREAM TASK TEST ===');
    console.log(`idleThresholdMinutes: 0.1 (6s)`);
    console.log(`autoSleepMinQueries: 5\n`);

    const queries = [
      'What is 2 + 2?',
      'What color is the sky?',
      'Name a fruit.',
      'Hello',
      'What does HTTP stand for?',
      'Is water wet?',
    ];

    for (const q of queries) {
      analyst.recordActivity();
      await router.queryDetailed(q);
    }

    console.log(`\nRan ${queries.length} queries. Now idling and waiting for dream task...`);

    // Wait up to 60 seconds for the dream task to fire
    const maxWait = 60_000;
    const start = Date.now();
    let fired = false;

    while (Date.now() - start < maxWait) {
      if (notificationReceived) {
        fired = true;
        break;
      }
      if (existsSync(DREAM_LOG)) {
        const current = readFileSync(DREAM_LOG, 'utf-8');
        if (current !== dreamLogBefore) {
          // Check if new entries appeared
          const newEntries = current
            .slice(dreamLogBefore.length)
            .trim()
            .split('\n')
            .filter(l => l);
          const hasComplete = newEntries.some(l => {
            try {
              return JSON.parse(l).event === 'complete';
            } catch {
              return false;
            }
          });
          if (hasComplete) {
            fired = true;
            break;
          }
        }
      }
      await new Promise(r => setTimeout(r, 1000));
    }

    analyst.stopIdleMonitor();

    console.log(`\nDream task fired: ${fired ? 'YES' : 'NO'}`);
    console.log(`Notification received: ${notificationReceived ? 'YES' : 'NO'}`);

    // Read dream log entries from this run
    if (existsSync(DREAM_LOG)) {
      const current = readFileSync(DREAM_LOG, 'utf-8');
      const newContent = current.slice(dreamLogBefore.length).trim();
      if (newContent) {
        console.log('\nDream log entries from this run:');
        for (const line of newContent.split('\n')) {
          if (line) console.log('  ' + line);
        }
      }
    }

    const gatePass = fired;
    console.log(`\nMILESTONE 6 DREAM TASK GATE: ${gatePass ? 'PASS' : 'FAIL'}`);
  } finally {
    writeFileSync(defaultPath, originalConfig, 'utf-8');
    console.log('\n(restored default.json)');
  }
}

main().catch(err => {
  console.error('Dream task test failed:', err);
  process.exit(1);
});
