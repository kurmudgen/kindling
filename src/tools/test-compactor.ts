/**
 * Log compaction verification.
 * Reads the real escalation.jsonl, runs it through the compactor,
 * and reports token reduction metrics.
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { compactLog } from '../sleep/compactor.js';
import type { EscalationEvent } from '../sleep/logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const LOG_FILE = resolve(__dirname, '../../logs/escalation.jsonl');

function main() {
  if (!existsSync(LOG_FILE)) {
    console.log('No escalation.jsonl found. Run some queries first.');
    process.exit(1);
  }

  const raw = readFileSync(LOG_FILE, 'utf-8').trim();
  const lines = raw.split('\n');
  const events: EscalationEvent[] = lines
    .map(l => {
      try {
        return JSON.parse(l);
      } catch {
        return null;
      }
    })
    .filter((e): e is EscalationEvent => e !== null && !('type' in e));

  // Group by sessionId
  const bySession = new Map<string, EscalationEvent[]>();
  for (const e of events) {
    const arr = bySession.get(e.sessionId) ?? [];
    arr.push(e);
    bySession.set(e.sessionId, arr);
  }

  console.log(`Total events: ${events.length}`);
  console.log(`Sessions: ${bySession.size}\n`);

  for (const [sessionId, sessionEvents] of bySession) {
    if (sessionEvents.length === 0) continue;

    const compacted = compactLog({
      sessionId,
      events: sessionEvents,
      sessionStartTime: sessionEvents[0].timestamp,
      sessionEndTime: sessionEvents[sessionEvents.length - 1].timestamp,
    });

    console.log(`=== Session ${sessionId.slice(0, 8)} (${sessionEvents.length} events) ===`);
    console.log(`  Raw chars:       ${compacted.tokenBudgetEstimate.rawCharCount}`);
    console.log(`  Compacted chars: ${compacted.tokenBudgetEstimate.compactedCharCount}`);
    console.log(`  Raw tokens (est):       ${compacted.tokenBudgetEstimate.rawTokensEst}`);
    console.log(`  Compacted tokens (est): ${compacted.tokenBudgetEstimate.compactedTokensEst}`);
    console.log(`  Reduction: ${compacted.tokenBudgetEstimate.reductionPct.toFixed(1)}%`);
    console.log(`  Summary header:`);
    console.log(
      compacted.summaryHeader
        .split('\n')
        .map(l => '    ' + l)
        .join('\n')
    );

    // Count repetition groups
    const repGroups = compacted.compactedEvents.filter(g => 'rep' in g);
    if (repGroups.length > 0) {
      console.log(`  Repetition groups: ${repGroups.length}`);
      for (const g of repGroups as Array<{ rep: true; n: number; tier: number; avgC: number; vRange: [number, number] }>) {
        console.log(
          `    [${g.n}x → T${g.tier}] valence ${g.vRange[0].toFixed(2)}–${g.vRange[1].toFixed(2)}, avg conf ${g.avgC.toFixed(2)}`
        );
      }
    }
    console.log();
  }

  // Build a synthetic 50-event log to verify the "under 2000 tokens" gate
  console.log('=== SYNTHETIC 50-EVENT LOG TEST ===');
  const synthetic: EscalationEvent[] = [];
  for (let i = 0; i < 50; i++) {
    synthetic.push({
      timestamp: new Date(Date.now() - (50 - i) * 60_000).toISOString(),
      sessionId: 'synthetic',
      queryHash: `hash_${i}`,
      valenceScore: {
        urgency: 0,
        complexity: 0.2 + (i % 5) * 0.05,
        stakes: 0,
        composite: 0.1 + (i % 5) * 0.025,
      },
      tier1ConfidenceAtHandoff: 0.68 + (i % 10) * 0.01,
      escalatedToTier: (2 + (i % 2)) as 2 | 3,
      tokensBeforeEscalation: 400 + (i % 100),
      handoffSuccessful: true,
      totalLatencyMs: 20000 + (i * 100),
    });
  }

  const syntheticCompacted = compactLog({
    sessionId: 'synthetic',
    events: synthetic,
    nonEscalatedQueryCount: 5,
  });

  console.log(`  Raw tokens: ${syntheticCompacted.tokenBudgetEstimate.rawTokensEst}`);
  console.log(`  Compacted tokens: ${syntheticCompacted.tokenBudgetEstimate.compactedTokensEst}`);
  console.log(`  Reduction: ${syntheticCompacted.tokenBudgetEstimate.reductionPct.toFixed(1)}%`);

  const under2000 = syntheticCompacted.tokenBudgetEstimate.compactedTokensEst < 2000;
  console.log(`\n  50-event log compacts to under 2000 tokens: ${under2000 ? 'PASS' : 'FAIL'}`);
  console.log(`\nMILESTONE 4 LOG COMPACTION GATE: ${under2000 ? 'PASS' : 'FAIL'}`);
}

main();
