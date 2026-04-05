/**
 * VCR record + replay verification.
 * Runs 5 queries in record mode, then invokes replay and validates
 * the diff output.
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const VCR_DIR = resolve(__dirname, '../../logs/vcr');

async function main() {
  // Enable record mode
  process.env.KINDLING_VCR = 'record';

  const { loadConfig } = await import('../config/config.js');
  loadConfig();

  const { Router } = await import('../router/router.js');
  const router = new Router();
  await router.init();

  const queries = [
    'What is 2 + 2?',
    'What does HTTP stand for?',
    'Explain the difference between TCP and UDP',
    'Name a color',
    'Hello',
  ];

  console.log('=== VCR RECORD MODE ===');
  for (const q of queries) {
    const result = await router.queryDetailed(q);
    console.log(`  [T${result.tier}] ${q.slice(0, 50)} — conf=${result.confidence.toFixed(3)}`);
  }

  // Find the most recent VCR file
  if (!existsSync(VCR_DIR)) {
    console.error('VCR directory not created');
    process.exit(1);
  }

  const files = readdirSync(VCR_DIR)
    .filter(f => f.endsWith('.jsonl'))
    .map(f => ({ f, mtime: statSync(resolve(VCR_DIR, f)).mtime.getTime() }))
    .sort((a, b) => b.mtime - a.mtime);

  if (files.length === 0) {
    console.error('No VCR files recorded');
    process.exit(1);
  }

  const latestFile = resolve(VCR_DIR, files[0].f);
  console.log(`\nRecorded to: ${latestFile}`);
  const content = readFileSync(latestFile, 'utf-8').trim();
  const lines = content.split('\n');
  console.log(`Line count: ${lines.length}`);

  // Privacy check: no raw prompts in file
  const hasRawPrompt = queries.some(q => content.includes(q));
  console.log(`Privacy check (no raw prompts): ${hasRawPrompt ? 'FAIL' : 'PASS'}`);

  // Verify structure
  const first = JSON.parse(lines[0]);
  const hasRequiredFields =
    first.queryHash &&
    first.valence &&
    first.signals &&
    typeof first.startTier === 'number' &&
    typeof first.finalTier === 'number' &&
    typeof first.confidence === 'number';
  console.log(`Structure check: ${hasRequiredFields ? 'PASS' : 'FAIL'}`);

  console.log('\n=== VCR REPLAY MODE ===');
  // Run replay inline by importing the replay logic
  process.env.KINDLING_VCR_FILE = latestFile;
  process.env.KINDLING_VCR = 'replay';

  // Re-run replay logic directly
  const entries = lines.map(l => JSON.parse(l));

  const { ConfidenceAggregator } = await import('../router/confidence.js');
  const { MetaConfidenceModel } = await import('../meta/meta-confidence.js');

  let matches = 0;
  const mismatches: Array<{ hash: string; reason: string }> = [];

  for (const entry of entries) {
    const conf = new ConfidenceAggregator();
    const meta = new MetaConfidenceModel();
    const decision = conf.decide(entry.startTier, entry.signals, entry.valence);
    const metaDecision = meta.evaluate(decision, entry.signals, entry.valence, entry.startTier);

    let currentFinalTier = entry.startTier;
    let currentEscalated = false;
    let shouldEscalate = decision.shouldEscalate;
    if (metaDecision.action === 'suppress') shouldEscalate = false;
    else if (metaDecision.action === 'force' && entry.startTier < 3) shouldEscalate = true;
    if (shouldEscalate && decision.targetTier !== entry.startTier) {
      currentFinalTier = decision.targetTier;
      currentEscalated = true;
    }

    if (currentFinalTier === entry.finalTier && currentEscalated === entry.escalated) {
      matches++;
    } else {
      mismatches.push({
        hash: entry.queryHash.slice(0, 8),
        reason: `recorded T${entry.finalTier} esc=${entry.escalated} → current T${currentFinalTier} esc=${currentEscalated}`,
      });
    }
  }

  console.log(`  Matches: ${matches}/${entries.length}`);
  if (mismatches.length > 0) {
    console.log('  Mismatches:');
    for (const m of mismatches) console.log(`    [${m.hash}] ${m.reason}`);
  }

  const recordOk = hasRequiredFields && !hasRawPrompt && lines.length === queries.length;
  const replayOk = matches === entries.length; // Same config, should be 100% match
  const allPass = recordOk && replayOk;
  console.log(`\nMILESTONE 5 VCR GATE: ${allPass ? 'PASS' : 'FAIL'}`);
}

main().catch(err => {
  console.error('VCR test failed:', err);
  process.exit(1);
});
