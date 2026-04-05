/**
 * Weight Diff Tool — Phase 3.5 decay-aware edition
 *
 * Shows default weights, raw learned weights, decayed effective weights,
 * and per-adjustment session age.
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { applyDecay, readLearnedStore } from '../config/config.js';

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
  const halfLife = profile.sleep?.decayHalfLifeSessions ?? 10;

  console.log(`Profile: ${profileName}`);
  console.log(`Decay half-life: ${halfLife} sessions\n`);

  console.log('=== DEFAULT WEIGHTS ===');
  for (const [k, v] of Object.entries(defaults)) {
    console.log(`  ${k}: ${v}`);
  }

  const store = readLearnedStore(learnedPath);
  if (!store || store.adjustments.length === 0) {
    console.log('\n=== LEARNED WEIGHTS ===');
    console.log('  (no learned.json or empty store)');
    console.log('\n=== DIFF ===');
    console.log('  (no changes)');
  } else {
    console.log(`\n=== LEARNED STORE (sessionCount: ${store.sessionCount}) ===`);
    for (const adj of store.adjustments) {
      const age = adj.sessionAge;
      const decayFactor = Math.pow(0.5, age / halfLife);
      console.log(
        `  ${adj.signal}: raw=${adj.suggestedWeight} age=${age} decay=${decayFactor.toFixed(3)} appliedAt=${adj.appliedAt.slice(0, 19)}`
      );
    }

    console.log('\n=== EFFECTIVE WEIGHTS (after decay) ===');
    for (const adj of store.adjustments) {
      const defaultVal = defaults[adj.signal];
      if (defaultVal === undefined) {
        console.log(`  ${adj.signal}: (no matching default, skipped)`);
        continue;
      }
      const effective = applyDecay(defaultVal, adj.suggestedWeight, adj.sessionAge, halfLife);
      const delta = effective - defaultVal;
      const sign = delta > 0 ? '+' : '';
      console.log(
        `  ${adj.signal}: default=${defaultVal} raw=${adj.suggestedWeight} → effective=${effective.toFixed(4)} (${sign}${delta.toFixed(4)} from default, age=${adj.sessionAge})`
      );
    }

    // Signals that still use their default (no learned override)
    const learnedKeys = new Set(store.adjustments.map(a => a.signal));
    const untouched = Object.keys(defaults).filter(k => !learnedKeys.has(k));
    if (untouched.length > 0) {
      console.log('\n  (untouched — still at default)');
      for (const k of untouched) {
        console.log(`    ${k}: ${defaults[k]}`);
      }
    }
  }

  // Sleep session history
  console.log('\n=== SLEEP SESSION HISTORY ===');
  if (!existsSync(SLEEP_DIR)) {
    console.log('  No sleep sessions recorded.');
  } else {
    const files = readdirSync(SLEEP_DIR).filter(f => f.endsWith('.json'));
    console.log(`  ${files.length} sleep session file(s):`);
    for (const f of files.sort()) {
      console.log(`    ${f}`);
    }
  }
}

main();
