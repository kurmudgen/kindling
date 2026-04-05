/**
 * Sleep Analyst Test — Milestone 3
 *
 * Runs 10 queries to generate escalation events, then triggers sleep analysis.
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { loadConfig } from '../config/config.js';
import { Router } from '../router/router.js';
import { SleepAnalyst } from '../sleep/analyst.js';

const QUERIES = [
  'What is 2 + 2',
  'Design a microservices authentication system with JWT refresh token rotation',
  'Hello',
  'Explain quantum entanglement in detail including the EPR paradox and Bell inequalities',
  'What color is the sky',
  'Write a secure password hashing implementation in TypeScript with salt, pepper, and timing attack protection',
  'ok thanks',
  'Analyze the tradeoffs between eventual consistency and strong consistency in distributed databases',
  'What time is it',
  'Design a real-time collaborative editing system like Google Docs including conflict resolution',
];

async function main() {
  loadConfig();
  const router = new Router();
  await router.init();

  console.log('=== Running 10 test queries to generate escalation log ===\n');

  for (let i = 0; i < QUERIES.length; i++) {
    const q = QUERIES[i];
    console.log(`[${i + 1}/10] ${q.slice(0, 60)}...`);
    try {
      const result = await router.queryDetailed(q);
      console.log(`  → Tier ${result.tier} | conf=${result.confidence.toFixed(3)} | escalated=${result.escalated} | ${result.text.split(/\s+/).length} tokens\n`);
    } catch (err) {
      console.log(`  → FAILED: ${err}\n`);
    }
  }

  console.log('=== Running Sleep Analysis ===\n');

  if (!process.env.ANTHROPIC_API_KEY) {
    console.log('BLOCKER: No ANTHROPIC_API_KEY available. Cannot run sleep analyst.');
    console.log('Gate: BLOCKED');
    process.exit(1);
  }

  try {
    const analyst = new SleepAnalyst();
    const result = await analyst.analyze();

    if (!result) {
      console.log('Analyst returned null — no escalation events in current session.');
      console.log('This may happen if no queries triggered escalation.');
      process.exit(1);
    }

    console.log('Sleep Analysis Result:');
    console.log(JSON.stringify(result, null, 2));

    // Gate checks
    console.log('\n=== GATE CHECKS ===');

    const conceptsOk = result.concepts_to_prewarm && result.concepts_to_prewarm.length > 0;
    console.log(`concepts_to_prewarm non-empty: ${conceptsOk ? 'PASS' : 'FAIL'} (${result.concepts_to_prewarm?.length ?? 0} items)`);
    if (conceptsOk) console.log(`  Values: ${result.concepts_to_prewarm.join(', ')}`);

    const adjustmentsOk = result.routing_adjustments && result.routing_adjustments.length > 0;
    console.log(`routing_adjustments non-empty: ${adjustmentsOk ? 'PASS' : 'FAIL'} (${result.routing_adjustments?.length ?? 0} items)`);

    const summaryOk = result.session_summary && result.session_summary.length > 50;
    console.log(`session_summary specific: ${summaryOk ? 'PASS' : 'FAIL'} (${result.session_summary?.length ?? 0} chars)`);

    const notesOk = result.confidence_model_notes && result.confidence_model_notes.length > 0;
    console.log(`confidence_model_notes present: ${notesOk ? 'PASS' : 'FAIL'} (${result.confidence_model_notes?.length ?? 0} items)`);

    const allPass = conceptsOk && adjustmentsOk && summaryOk && notesOk;
    console.log(`\nMILESTONE 3 GATE: ${allPass ? 'PASS' : 'NEEDS ATTENTION'}`);
  } catch (err) {
    console.error('Sleep analysis failed:', err);
    process.exit(1);
  }
}

main().catch(err => {
  console.error('Sleep test failed:', err);
  process.exit(1);
});
