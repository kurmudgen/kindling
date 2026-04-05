/**
 * Weight Merge Verification — Milestone 4d
 *
 * Writes a synthetic learned.json and verifies that config.ts merges it
 * into the live escalation weights. This proves the end-to-end pipeline
 * works without requiring a live sleep analyst run.
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { writeFileSync, unlinkSync, existsSync, readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadConfig, getConfig } from '../config/config.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const LEARNED_PATH = resolve(__dirname, '../../config/learned.json');

function main() {
  // Make sure there's no existing learned.json
  const hadLearned = existsSync(LEARNED_PATH);
  let backup: string | null = null;
  if (hadLearned) {
    backup = readFileSync(LEARNED_PATH, 'utf-8');
  }

  try {
    // Step 1: Load config WITHOUT learned.json
    if (hadLearned) unlinkSync(LEARNED_PATH);
    const beforeConfig = loadConfig();
    const defaultSpreadWeight = beforeConfig.escalation.tokenProbabilitySpreadWeight;
    const defaultThreshold = beforeConfig.escalation.escalationThreshold;
    console.log('=== BEFORE learned.json ===');
    console.log(`  tokenProbabilitySpreadWeight: ${defaultSpreadWeight}`);
    console.log(`  escalationThreshold: ${defaultThreshold}`);

    // Step 2: Write synthetic learned.json
    const synthetic = {
      escalation: {
        tokenProbabilitySpreadWeight: 0.50,
        escalationThreshold: 0.35,
      },
    };
    writeFileSync(LEARNED_PATH, JSON.stringify(synthetic, null, 2), 'utf-8');

    // Step 3: Reload config (forces re-merge)
    const afterConfig = loadConfig();
    const mergedSpreadWeight = afterConfig.escalation.tokenProbabilitySpreadWeight;
    const mergedThreshold = afterConfig.escalation.escalationThreshold;
    console.log('\n=== AFTER learned.json merged ===');
    console.log(`  tokenProbabilitySpreadWeight: ${mergedSpreadWeight}`);
    console.log(`  escalationThreshold: ${mergedThreshold}`);

    // Step 4: Verify merge took effect
    const spreadMerged = mergedSpreadWeight === 0.50;
    const thresholdMerged = mergedThreshold === 0.35;
    // Verify untouched keys preserved
    const velocityPreserved = afterConfig.escalation.semanticVelocityWeight === beforeConfig.escalation.semanticVelocityWeight;

    console.log('\n=== VERIFICATION ===');
    console.log(`  tokenProbabilitySpreadWeight merged: ${spreadMerged ? 'PASS' : 'FAIL'}`);
    console.log(`  escalationThreshold merged: ${thresholdMerged ? 'PASS' : 'FAIL'}`);
    console.log(`  Untouched weights preserved: ${velocityPreserved ? 'PASS' : 'FAIL'}`);

    // Step 5: Verify KINDLING_IGNORE_LEARNED bypass
    process.env.KINDLING_IGNORE_LEARNED = 'true';
    const ignoredConfig = loadConfig();
    const ignoreSpread = ignoredConfig.escalation.tokenProbabilitySpreadWeight;
    const ignoreBypass = ignoreSpread === defaultSpreadWeight;
    console.log(`  KINDLING_IGNORE_LEARNED bypasses merge: ${ignoreBypass ? 'PASS' : 'FAIL'}`);
    delete process.env.KINDLING_IGNORE_LEARNED;

    const allPass = spreadMerged && thresholdMerged && velocityPreserved && ignoreBypass;
    console.log(`\nMILESTONE 4 WEIGHT MERGE GATE: ${allPass ? 'PASS' : 'FAIL'}`);
  } finally {
    // Cleanup
    if (existsSync(LEARNED_PATH)) unlinkSync(LEARNED_PATH);
    if (backup) {
      writeFileSync(LEARNED_PATH, backup, 'utf-8');
    }
  }
}

main();
