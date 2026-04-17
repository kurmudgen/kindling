import dotenv from 'dotenv';
dotenv.config({ override: true });

import pino from 'pino';
import { loadConfig } from './config/config.js';
import { Router } from './router/router.js';
import { SleepAnalyst } from './sleep/analyst.js';
import { createInterface } from 'node:readline';

const log = pino({ level: process.env.KINDLING_LOG_LEVEL ?? 'info' });

async function main(): Promise<void> {
  const config = loadConfig();
  log.info({ profile: config.profile }, 'Kindling starting');

  const router = new Router();
  await router.init();

  // Start sleep analyst if API key available
  let analyst: SleepAnalyst | null = null;
  if (process.env.ANTHROPIC_API_KEY) {
    analyst = new SleepAnalyst();
    analyst.onNotification((msg) => {
      // Surface dream task completion to the user
      console.log(`\n${msg}\n`);
      process.stdout.write('> ');
    });
    // Phase 5A: hot-reload the ML classifier after each successful retrain
    analyst.setRetrainCallback(() => router.reloadMLClassifier());
    analyst.startIdleMonitor();
    if (process.env.KINDLING_DREAM !== 'false') {
      log.info('Dream task monitor started (auto sleep on idle)');
    }
  }

  log.info('Kindling ready. Enter prompts (Ctrl+C to exit).\n');

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const context: string[] = [];

  rl.on('line', async (line) => {
    const prompt = line.trim();
    if (!prompt) return;

    if (prompt === '/sleep') {
      if (analyst) {
        log.info('Triggering manual sleep analysis...');
        const result = await analyst.analyze();
        if (result) {
          console.log('\nSleep Analysis:');
          console.log(JSON.stringify(result, null, 2));
        } else {
          console.log('No data to analyze.');
        }
      } else {
        console.log('Sleep analyst not available (no API key).');
      }
      return;
    }

    if (prompt === '/clear') {
      context.length = 0;
      console.log('Context cleared.');
      return;
    }

    analyst?.recordActivity();

    try {
      const result = await router.queryDetailed(prompt, context);
      const tierLabel = `[Tier ${result.tier}${result.escalated ? ' (escalated)' : ''}]`;
      console.log(`\n${tierLabel} ${result.text}\n`);
      context.push(`User: ${prompt}`);
      context.push(`Assistant: ${result.text}`);
      // Keep context window manageable
      if (context.length > 20) {
        context.splice(0, 2);
      }
    } catch (err) {
      log.error({ err }, 'Query failed');
      console.error('Error processing query. Check logs for details.');
    }
  });

  rl.on('close', () => {
    analyst?.stopIdleMonitor();
    log.info('Kindling shutting down');
    process.exit(0);
  });
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
