/**
 * ML Classifier Verification Tool — Phase 4 Gate Test
 *
 * Tests the ML classifier inference engine with synthetic model weights.
 * Does NOT require Python or a trained model — creates a test model
 * in-memory and verifies the math is correct.
 *
 * Usage:
 *   tsx src/tools/test-ml-classifier.ts
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { writeFileSync, mkdirSync, existsSync, unlinkSync, copyFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  MLClassifier,
  extractFeatures,
  getModelFilePath,
  getModelDir,
} from '../meta/ml-classifier.js';
import type { EscalationSignals, ValenceScore } from '../tiers/tier-interface.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ─── Synthetic model for testing ────────────────────────────────

const SYNTHETIC_LOGREG = {
  type: 'logistic_regression' as const,
  version: 1,
  features: [
    'tokenProbabilitySpread',
    'semanticVelocity',
    'surpriseScore',
    'attentionAnomalyScore',
    'valenceUrgency',
    'valenceComplexity',
    'valenceStakes',
    'valenceComposite',
    'spreadXcomplexity',
    'surpriseXstakes',
  ],
  // Weights designed so high signals + high complexity → escalation needed
  weights: [
    2.0,   // tokenProbabilitySpread — strongest signal
    1.0,   // semanticVelocity
    1.5,   // surpriseScore
    0.5,   // attentionAnomalyScore
    0.3,   // valenceUrgency
    1.2,   // valenceComplexity
    0.8,   // valenceStakes
    0.5,   // valenceComposite
    1.0,   // spreadXcomplexity interaction
    0.7,   // surpriseXstakes interaction
  ],
  intercept: -3.0,  // bias toward "not needed" — escalation must be earned
  threshold: 0.5,
  trainedAt: '2026-04-16T00:00:00.000Z',
  trainingExamples: 100,
  accuracy: 0.85,
};

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

async function runTests() {
  console.log('=== ML Classifier Verification (Phase 4) ===\n');

  const modelPath = getModelFilePath();
  const modelDir = getModelDir();
  const backupPath = modelPath + '.backup';

  // Backup existing model if any
  if (existsSync(modelPath)) {
    copyFileSync(modelPath, backupPath);
    console.log('Backed up existing model\n');
  }

  let passed = 0;
  let failed = 0;

  try {
    // TEST 1: Feature extraction
    console.log('TEST 1: Feature extraction');
    const signals: EscalationSignals = {
      tokenProbabilitySpread: 0.45,
      semanticVelocity: 0.30,
      surpriseScore: 0.20,
      attentionAnomalyScore: 0.10,
    };
    const valence: ValenceScore = {
      urgency: 0.1,
      complexity: 0.6,
      stakes: 0.3,
      composite: 0.38,
    };
    const features = extractFeatures(signals, valence);
    if (
      features.length === 10 &&
      features[0] === 0.45 &&         // tokenProbabilitySpread
      features[4] === 0.1 &&          // valenceUrgency
      Math.abs(features[8] - 0.45 * 0.6) < 0.001 && // spreadXcomplexity
      Math.abs(features[9] - 0.20 * 0.3) < 0.001    // surpriseXstakes
    ) {
      console.log(`  PASS — ${features.length} features, interaction terms correct`);
      passed++;
    } else {
      console.log(`  FAIL — features: ${JSON.stringify(features)}`);
      failed++;
    }

    // TEST 2: Write synthetic model and load it
    console.log('\nTEST 2: Model load from disk');
    if (!existsSync(modelDir)) mkdirSync(modelDir, { recursive: true });
    writeFileSync(modelPath, JSON.stringify(SYNTHETIC_LOGREG, null, 2), 'utf-8');

    const classifier = new MLClassifier();
    if (classifier.isLoaded()) {
      const info = classifier.getModelInfo();
      console.log(`  PASS — loaded ${info?.type} with ${info?.trainingExamples} examples`);
      passed++;
    } else {
      console.log('  FAIL — model not loaded');
      failed++;
    }

    // TEST 3: Prediction on trivial query (low signals → should NOT need escalation)
    console.log('\nTEST 3: Trivial query prediction (should NOT escalate)');
    const trivialSignals: EscalationSignals = {
      tokenProbabilitySpread: 0.10,
      semanticVelocity: 0.05,
      surpriseScore: 0.02,
      attentionAnomalyScore: 0.01,
    };
    const trivialValence: ValenceScore = {
      urgency: 0,
      complexity: 0.1,
      stakes: 0,
      composite: 0.05,
    };
    const trivialResult = classifier.predict(trivialSignals, trivialValence);
    if (trivialResult && !trivialResult.escalationNeeded && trivialResult.probability < 0.5) {
      console.log(`  PASS — not needed (p=${trivialResult.probability.toFixed(3)}, confidence=${trivialResult.confidence.toFixed(3)})`);
      passed++;
    } else {
      console.log(`  FAIL — result: ${JSON.stringify(trivialResult)}`);
      failed++;
    }

    // TEST 4: Prediction on complex query (high signals → SHOULD need escalation)
    console.log('\nTEST 4: Complex query prediction (should escalate)');
    const complexSignals: EscalationSignals = {
      tokenProbabilitySpread: 0.80,
      semanticVelocity: 0.70,
      surpriseScore: 0.60,
      attentionAnomalyScore: 0.40,
    };
    const complexValence: ValenceScore = {
      urgency: 0.5,
      complexity: 0.8,
      stakes: 0.6,
      composite: 0.65,
    };
    const complexResult = classifier.predict(complexSignals, complexValence);
    if (complexResult && complexResult.escalationNeeded && complexResult.probability > 0.5) {
      console.log(`  PASS — needed (p=${complexResult.probability.toFixed(3)}, confidence=${complexResult.confidence.toFixed(3)})`);
      passed++;
    } else {
      console.log(`  FAIL — result: ${JSON.stringify(complexResult)}`);
      failed++;
    }

    // TEST 5: Math verification — manual dot product should match
    console.log('\nTEST 5: Manual math verification');
    const testFeatures = extractFeatures(complexSignals, complexValence);
    let manualLogit = SYNTHETIC_LOGREG.intercept;
    for (let i = 0; i < testFeatures.length; i++) {
      manualLogit += testFeatures[i] * SYNTHETIC_LOGREG.weights[i];
    }
    const manualProbability = sigmoid(manualLogit);
    const classifierProbability = complexResult?.probability ?? 0;

    if (Math.abs(manualProbability - classifierProbability) < 0.001) {
      console.log(`  PASS — manual=${manualProbability.toFixed(4)}, classifier=${classifierProbability.toFixed(4)} (match)`);
      passed++;
    } else {
      console.log(`  FAIL — manual=${manualProbability.toFixed(4)}, classifier=${classifierProbability.toFixed(4)} (mismatch)`);
      failed++;
    }

    // TEST 6: Classifier without model returns null
    console.log('\nTEST 6: No model → returns null');
    unlinkSync(modelPath);
    const emptyClassifier = new MLClassifier();
    const nullResult = emptyClassifier.predict(trivialSignals, trivialValence);
    if (nullResult === null && !emptyClassifier.isLoaded()) {
      console.log('  PASS — returns null when no model');
      passed++;
    } else {
      console.log(`  FAIL — expected null, got: ${JSON.stringify(nullResult)}`);
      failed++;
    }

    // TEST 7: Reload after model appears
    console.log('\nTEST 7: Hot reload after training');
    writeFileSync(modelPath, JSON.stringify(SYNTHETIC_LOGREG, null, 2), 'utf-8');
    const reloaded = emptyClassifier.reload();
    const afterReload = emptyClassifier.predict(complexSignals, complexValence);
    if (reloaded && afterReload && afterReload.escalationNeeded) {
      console.log('  PASS — hot reload succeeded, predictions working');
      passed++;
    } else {
      console.log(`  FAIL — reload=${reloaded}, prediction=${JSON.stringify(afterReload)}`);
      failed++;
    }

    console.log(`\n=== Results: ${passed} PASS, ${failed} FAIL ===`);
  } finally {
    // Restore backup
    if (existsSync(backupPath)) {
      copyFileSync(backupPath, modelPath);
      unlinkSync(backupPath);
      console.log('Restored original model');
    } else if (existsSync(modelPath)) {
      unlinkSync(modelPath);
    }
  }

  process.exit(failed > 0 ? 1 : 0);
}

runTests().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
