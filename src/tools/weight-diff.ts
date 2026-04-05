/**
 * Weight Diff Tool — Milestone 4c
 *
 * Shows default weights, learned weights, and the diff.
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readdirSync } from 'node:fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CONFIG_DIR = resolve(__dirname, '../../config');
const SLEEP_DIR = resolve(__dirname, '../../logs/sleep');

function main() {
  const profileName = process.env.KINDLING_PROFILE ?? 'default';
  const profilePath = resolve(CONFIG_DIR, `${profileName}.json`);
  const learnedPath = resolve(CONFIG_DIR, 'learned.json');

  const profile = JSON.parse(readFileSync(profilePath, 'utf-8'));
  const defaults = profile.escalation as Record<string, number>;

  console.log('=== DEFAULT WEIGHTS ===');
  for (const [k, v] of Object.entries(defaults)) {
    console.log(`  ${k}: ${v}`);
  }

  if (!existsSync(learnedPath)) {
    console.log('\n=== LEARNED WEIGHTS ===');
    console.log('  (no learned.json — no sleep sessions have run yet)');
    console.log('\n=== DIFF ===');
    console.log('  (no changes)');
  } else {
    const learned = JSON.parse(readFileSync(learnedPath, 'utf-8'));
    const learnedEscalation = (learned.escalation ?? {}) as Record<string, number>;

    console.log('\n=== LEARNED WEIGHTS ===');
    if (Object.keys(learnedEscalation).length === 0) {
      console.log('  (learned.json exists but escalation section is empty)');
    } else {
      for (const [k, v] of Object.entries(learnedEscalation)) {
        console.log(`  ${k}: ${v}`);
      }
    }

    console.log('\n=== DIFF (learned vs default) ===');
    let hasDiff = false;
    for (const [k, v] of Object.entries(learnedEscalation)) {
      const defaultVal = defaults[k];
      if (defaultVal !== undefined && defaultVal !== v) {
        const delta = (v as number) - defaultVal;
        console.log(`  ${k}: ${defaultVal} → ${v} (${delta > 0 ? '+' : ''}${delta.toFixed(4)})`);
        hasDiff = true;
      } else if (defaultVal === undefined) {
        console.log(`  ${k}: (new) ${v}`);
        hasDiff = true;
      }
    }
    if (!hasDiff) {
      console.log('  (no differences)');
    }

    // Effective (merged) weights
    console.log('\n=== EFFECTIVE WEIGHTS (after merge) ===');
    const merged = { ...defaults, ...learnedEscalation };
    for (const [k, v] of Object.entries(merged)) {
      const isLearned = learnedEscalation[k] !== undefined && learnedEscalation[k] !== defaults[k];
      console.log(`  ${k}: ${v}${isLearned ? ' (learned)' : ''}`);
    }
  }

  // Count sleep sessions
  console.log('\n=== SLEEP SESSION HISTORY ===');
  if (!existsSync(SLEEP_DIR)) {
    console.log('  No sleep sessions have been recorded.');
  } else {
    const files = readdirSync(SLEEP_DIR).filter(f => f.endsWith('.json'));
    console.log(`  ${files.length} sleep session(s) recorded:`);
    for (const f of files.sort()) {
      console.log(`    ${f}`);
    }
  }
}

main();
