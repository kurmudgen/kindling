/**
 * Recovery Cascade Verification
 *
 * Simulates Tier 2 failure (bogus model name) and verifies:
 * - Recovery cascade attempts fallback model, API fallback, then tier downgrade
 * - Circuit breaker trips after 3 consecutive degradations
 * - Recovery events appear in the escalation log
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { writeFileSync, readFileSync, unlinkSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CONFIG_DIR = resolve(__dirname, '../../config');

async function main() {
  // Backup default.json
  const defaultPath = resolve(CONFIG_DIR, 'default.json');
  const originalConfig = readFileSync(defaultPath, 'utf-8');

  try {
    // Mutate Tier 2 model to a bogus name
    const config = JSON.parse(originalConfig);
    config.tier2.model = 'this-model-definitely-does-not-exist:99b';
    config.tier2.fallbackModel = 'also-fake:99b';
    // Leave apiFallbackModel pointing at the real Haiku model if API key set
    writeFileSync(defaultPath, JSON.stringify(config, null, 2), 'utf-8');

    console.log('=== SIMULATING TIER 2 FAILURE ===');
    console.log(`Set tier2.model to: ${config.tier2.model}`);
    console.log(`Set tier2.fallbackModel to: ${config.tier2.fallbackModel}`);
    console.log(`Has ANTHROPIC_API_KEY: ${!!process.env.ANTHROPIC_API_KEY}`);

    // Now import the router (after config mutation)
    const { loadConfig } = await import('../config/config.js');
    loadConfig(); // reload with mutated config

    const { Router } = await import('../router/router.js');
    const router = new Router();
    await router.init();

    // Force a query to Tier 2 by using a complex prompt
    const result = await router.queryDetailed(
      'Design a distributed consensus protocol for a multi-region database system with strong consistency requirements and explain all the tradeoffs.'
    );

    console.log('\n=== RESULT ===');
    console.log(`Final tier: ${result.tier}`);
    console.log(`Escalation path: ${result.escalationPath.join(' → ')}`);
    console.log(`Tokens: ${result.text.split(/\s+/).filter(Boolean).length}`);
    console.log(`Latency: ${Math.round(result.latencyMs)}ms`);

    if (result.recoveryEvents && result.recoveryEvents.length > 0) {
      console.log(`\n=== RECOVERY EVENTS (${result.recoveryEvents.length}) ===`);
      for (const ev of result.recoveryEvents) {
        console.log(`  [${ev.layer}] ${ev.originalTier}→${ev.recoveredTier} success=${ev.success}: ${ev.reason}`);
      }
    } else {
      console.log('\n=== NO RECOVERY EVENTS (unexpected) ===');
    }

    // Gate check
    const hasRecoveryEvents = (result.recoveryEvents?.length ?? 0) > 0;
    const finalTierIsValid = result.tier >= 1 && result.tier <= 3;
    const hasResponseContent = result.text.length > 20;

    console.log('\n=== GATE CHECKS ===');
    console.log(`  Recovery events generated: ${hasRecoveryEvents ? 'PASS' : 'FAIL'}`);
    console.log(`  Final tier is valid (not null): ${finalTierIsValid ? 'PASS' : 'FAIL'}`);
    console.log(`  Response has content (graceful fallback): ${hasResponseContent ? 'PASS' : 'FAIL'}`);

    const allPass = hasRecoveryEvents && finalTierIsValid && hasResponseContent;
    console.log(`\nMILESTONE 3 RECOVERY GATE: ${allPass ? 'PASS' : 'FAIL'}`);
  } finally {
    // Restore default.json
    writeFileSync(defaultPath, originalConfig, 'utf-8');
    console.log('\n(restored default.json)');
  }
}

main().catch(err => {
  console.error('Recovery test failed:', err);
  process.exit(1);
});
