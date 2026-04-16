/**
 * Shadow Evaluator Verification Tool — Phase 4 Gate Test
 *
 * Tests the shadow evaluation pipeline end-to-end:
 * 1. Records a synthetic training example to the store
 * 2. Reads it back and verifies fields
 * 3. Tests shadow summary aggregation
 * 4. Optionally runs a live shadow evaluation against the API
 *
 * Usage:
 *   tsx src/tools/test-shadow.ts          # offline tests only
 *   tsx src/tools/test-shadow.ts --live   # includes real API call
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync, unlinkSync } from 'node:fs';
import {
  recordTrainingExample,
  loadTrainingExamples,
  getTrainingStats,
  getTrainingFilePath,
} from '../shadow/training-store.js';
import type { TrainingExample } from '../shadow/training-store.js';
import type { EscalationSignals, ValenceScore } from '../tiers/tier-interface.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const isLive = process.argv.includes('--live');

function syntheticExample(overrides?: Partial<TrainingExample>): TrainingExample {
  return {
    timestamp: new Date().toISOString(),
    sessionId: 'test-session-001',
    queryHash: 'abc123def456',
    signals: {
      tokenProbabilitySpread: 0.45,
      semanticVelocity: 0.30,
      surpriseScore: 0.15,
      attentionAnomalyScore: 0.08,
    },
    valence: {
      urgency: 0.1,
      complexity: 0.5,
      stakes: 0.2,
      composite: 0.33,
    },
    tierUsed: 1,
    routerDecision: 'stay',
    metaAction: 'confirm',
    localTokenCount: 45,
    localCoherence: 0.72,
    localLatencyMs: 340,
    apiTokenCount: 120,
    apiCoherence: 0.91,
    apiLatencyMs: 1200,
    apiModel: 'claude-haiku-4-5-20251001',
    escalationNeeded: true,
    qualityDelta: 0.19,
    ...overrides,
  };
}

async function runTests() {
  console.log('=== Shadow Evaluator Verification (Phase 4) ===\n');

  // Clean up any previous test data
  const trainingPath = getTrainingFilePath();
  const backupPath = trainingPath + '.backup';
  if (existsSync(trainingPath)) {
    // Backup existing training data
    const { copyFileSync } = await import('node:fs');
    copyFileSync(trainingPath, backupPath);
    unlinkSync(trainingPath);
    console.log('Backed up existing training data\n');
  }

  let passed = 0;
  let failed = 0;

  try {
    // TEST 1: Record and read back training examples
    console.log('TEST 1: Training store write/read cycle');
    const example1 = syntheticExample({ escalationNeeded: true, qualityDelta: 0.19 });
    const example2 = syntheticExample({
      escalationNeeded: false,
      qualityDelta: -0.05,
      tierUsed: 2,
      localCoherence: 0.88,
      apiCoherence: 0.83,
      signals: {
        tokenProbabilitySpread: 0.15,
        semanticVelocity: 0.10,
        surpriseScore: 0.05,
        attentionAnomalyScore: 0.02,
      },
    });
    const example3 = syntheticExample({
      escalationNeeded: true,
      qualityDelta: 0.25,
      tierUsed: 1,
      localCoherence: 0.55,
      apiCoherence: 0.80,
    });

    recordTrainingExample(example1);
    recordTrainingExample(example2);
    recordTrainingExample(example3);

    // Wait for async writes to flush
    await new Promise(r => setTimeout(r, 200));

    const examples = await loadTrainingExamples();
    if (examples.length === 3) {
      console.log(`  PASS — wrote 3 examples, read back ${examples.length}`);
      passed++;
    } else {
      console.log(`  FAIL — expected 3 examples, got ${examples.length}`);
      failed++;
    }

    // TEST 2: Training stats
    console.log('\nTEST 2: Training store stats');
    const stats = await getTrainingStats();
    if (stats.exists && stats.totalExamples === 3 && stats.fileSizeBytes > 0) {
      console.log(`  PASS — ${stats.totalExamples} examples, ${stats.fileSizeBytes} bytes`);
      passed++;
    } else {
      console.log(`  FAIL — stats: ${JSON.stringify(stats)}`);
      failed++;
    }

    // TEST 3: Field integrity
    console.log('\nTEST 3: Field integrity on read-back');
    const readBack = examples[0];
    const checks = [
      readBack.signals.tokenProbabilitySpread === 0.45,
      readBack.valence.composite === 0.33,
      readBack.tierUsed === 1,
      readBack.escalationNeeded === true,
      readBack.qualityDelta === 0.19,
      readBack.apiModel === 'claude-haiku-4-5-20251001',
    ];
    if (checks.every(c => c)) {
      console.log('  PASS — all fields preserved correctly');
      passed++;
    } else {
      console.log(`  FAIL — field mismatch: ${checks.map((c, i) => `check${i}=${c}`).join(', ')}`);
      failed++;
    }

    // TEST 4: Escalation-needed label distribution
    console.log('\nTEST 4: Label distribution');
    const neededCount = examples.filter(e => e.escalationNeeded).length;
    const notNeededCount = examples.filter(e => !e.escalationNeeded).length;
    if (neededCount === 2 && notNeededCount === 1) {
      console.log(`  PASS — needed=${neededCount}, not-needed=${notNeededCount}`);
      passed++;
    } else {
      console.log(`  FAIL — needed=${neededCount}, not-needed=${notNeededCount} (expected 2, 1)`);
      failed++;
    }

    // TEST 5: Shadow evaluator instantiation (no API call)
    console.log('\nTEST 5: Shadow evaluator instantiation');
    try {
      const { ShadowEvaluator } = await import('../shadow/shadow.js');
      const shadow = new ShadowEvaluator({ sampleRate: 0 }); // disabled
      const evalStats = shadow.getStats();
      if (!evalStats.enabled && evalStats.sampleRate === 0) {
        console.log('  PASS — instantiated with sampleRate=0, correctly disabled');
        passed++;
      } else {
        console.log(`  FAIL — expected disabled, got: ${JSON.stringify(evalStats)}`);
        failed++;
      }
    } catch (err) {
      console.log(`  FAIL — instantiation error: ${(err as Error).message}`);
      failed++;
    }

    // TEST 6 (optional): Live shadow evaluation
    if (isLive) {
      console.log('\nTEST 6: Live shadow evaluation (API call)');
      if (!process.env.ANTHROPIC_API_KEY) {
        console.log('  SKIP — no ANTHROPIC_API_KEY set');
      } else {
        try {
          const { ShadowEvaluator } = await import('../shadow/shadow.js');
          const shadow = new ShadowEvaluator({
            sampleRate: 1,
            apiModel: 'claude-haiku-4-5-20251001',
            maxTokens: 256,
          });

          await shadow.shadowDirect({
            prompt: 'What is 2+2?',
            context: [],
            signals: {
              tokenProbabilitySpread: 0.10,
              semanticVelocity: 0.05,
              surpriseScore: 0.02,
              attentionAnomalyScore: 0.01,
            },
            valence: { urgency: 0, complexity: 0.1, stakes: 0, composite: 0.05 },
            tierUsed: 1,
            routerDecision: 'stay',
            metaAction: 'confirm',
            localTokenCount: 5,
            localCoherence: 0.65,
            localLatencyMs: 150,
          });

          // Wait for write
          await new Promise(r => setTimeout(r, 200));

          const afterLive = await loadTrainingExamples();
          if (afterLive.length === 4) {
            const liveExample = afterLive[3];
            console.log(`  PASS — live shadow recorded`);
            console.log(`    API coherence: ${liveExample.apiCoherence.toFixed(2)}`);
            console.log(`    Quality delta: ${liveExample.qualityDelta.toFixed(2)}`);
            console.log(`    Escalation needed: ${liveExample.escalationNeeded}`);
            passed++;
          } else {
            console.log(`  FAIL — expected 4 examples after live, got ${afterLive.length}`);
            failed++;
          }
        } catch (err) {
          console.log(`  FAIL — live shadow error: ${(err as Error).message}`);
          failed++;
        }
      }
    }

    console.log(`\n=== Results: ${passed} PASS, ${failed} FAIL ===`);
  } finally {
    // Restore backup if it exists
    if (existsSync(backupPath)) {
      const { copyFileSync } = await import('node:fs');
      copyFileSync(backupPath, trainingPath);
      unlinkSync(backupPath);
      console.log('Restored original training data');
    } else if (existsSync(trainingPath)) {
      // Clean up test data
      unlinkSync(trainingPath);
    }
  }

  process.exit(failed > 0 ? 1 : 0);
}

runTests().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
