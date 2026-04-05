/**
 * Memory Decay Verification Test
 * Creates a synthetic learned.json with aged entries and verifies the
 * effective weights decay correctly toward defaults over session age.
 */
import { writeFileSync, unlinkSync, existsSync, readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadConfig, readLearnedStore, applyDecay } from '../config/config.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const LEARNED_PATH = resolve(__dirname, '../../config/learned.json');

function main() {
  // Backup existing
  const hadLearned = existsSync(LEARNED_PATH);
  const backup = hadLearned ? readFileSync(LEARNED_PATH, 'utf-8') : null;

  try {
    console.log('=== DECAY FORMULA UNIT TEST ===');
    const defaultWeight = 0.35;
    const suggestedWeight = 0.50;
    const halfLife = 10;

    for (const age of [0, 5, 10, 20, 40]) {
      const effective = applyDecay(defaultWeight, suggestedWeight, age, halfLife);
      const contribution = ((effective - defaultWeight) / (suggestedWeight - defaultWeight)) * 100;
      console.log(`  age=${age}: effective=${effective.toFixed(4)} (${contribution.toFixed(1)}% of raw adjustment)`);
    }

    // Verify: age 0 should be 100%, age 10 should be 50%, age 20 should be 25%
    const age0 = applyDecay(0.35, 0.50, 0, 10);
    const age10 = applyDecay(0.35, 0.50, 10, 10);
    const age20 = applyDecay(0.35, 0.50, 20, 10);

    const age0Ok = Math.abs(age0 - 0.50) < 0.001;
    const age10Ok = Math.abs(age10 - 0.425) < 0.001; // 0.35 + 0.15 * 0.5
    const age20Ok = Math.abs(age20 - 0.3875) < 0.001; // 0.35 + 0.15 * 0.25

    console.log(`\n  age=0 full effect: ${age0Ok ? 'PASS' : 'FAIL'}`);
    console.log(`  age=10 half effect: ${age10Ok ? 'PASS' : 'FAIL'}`);
    console.log(`  age=20 quarter effect: ${age20Ok ? 'PASS' : 'FAIL'}`);

    console.log('\n=== END-TO-END DECAY IN loadConfig ===');

    // Write a synthetic learned.json with an aged adjustment
    const synthetic = {
      sessionCount: 15,
      adjustments: [
        {
          signal: 'tokenProbabilitySpreadWeight',
          suggestedWeight: 0.50,
          appliedAt: '2026-01-01T00:00:00.000Z',
          sessionAge: 10, // half-life reached — should contribute 50%
        },
        {
          signal: 'semanticVelocityWeight',
          suggestedWeight: 0.40,
          appliedAt: '2026-03-01T00:00:00.000Z',
          sessionAge: 0, // fresh — full effect
        },
      ],
    };
    writeFileSync(LEARNED_PATH, JSON.stringify(synthetic, null, 2), 'utf-8');

    const config = loadConfig();
    console.log('  tokenProbabilitySpreadWeight (age=10, raw=0.50, default=0.35):');
    console.log(`    effective=${config.escalation.tokenProbabilitySpreadWeight.toFixed(4)} (expected ~0.425)`);
    console.log('  semanticVelocityWeight (age=0, raw=0.40, default=0.25):');
    console.log(`    effective=${config.escalation.semanticVelocityWeight.toFixed(4)} (expected 0.4000)`);

    const decayed = Math.abs(config.escalation.tokenProbabilitySpreadWeight - 0.425) < 0.001;
    const fresh = Math.abs(config.escalation.semanticVelocityWeight - 0.40) < 0.001;
    console.log(`\n  Aged adjustment decays correctly: ${decayed ? 'PASS' : 'FAIL'}`);
    console.log(`  Fresh adjustment fully applied: ${fresh ? 'PASS' : 'FAIL'}`);

    const allPass = age0Ok && age10Ok && age20Ok && decayed && fresh;
    console.log(`\nMILESTONE 2 MEMORY DECAY GATE: ${allPass ? 'PASS' : 'FAIL'}`);
  } finally {
    // Restore
    if (backup) {
      writeFileSync(LEARNED_PATH, backup, 'utf-8');
    } else if (existsSync(LEARNED_PATH)) {
      unlinkSync(LEARNED_PATH);
    }
  }
}

main();
