/**
 * Speed Bench — Phase 7+
 *
 * Multiple rounds testing different knobs to speed up CPU+RAM inference.
 * Same 3-prompt workload each round so configurations are comparable.
 *
 * Subject models:
 *   - qwen2.5:32b  (~20GB, CPU-bound on 16GB VRAM, partial offload viable)
 *   - qwen2.5:14b  (~9GB, fits fully in 16GB VRAM — corrected from prior bench)
 *
 * Each config is run via Ollama API, which returns eval_count and
 * eval_duration so we can compute true tok/s independent of network jitter.
 *
 * Rounds (32B):
 *   1. Pure-CPU baseline (num_gpu=0, default num_thread)
 *   2. num_thread tuned to physical core count (16)
 *   3. num_thread tuned to logical core count (32)
 *   4. Partial GPU offload — sweep num_gpu = 10, 20, 30
 *   5. Combine best num_thread + best num_gpu
 *
 * Rounds (14B):
 *   6. Pure-CPU vs full-GPU offload — measures the GPU lift on a model
 *      that genuinely fits in VRAM (not the case in earlier benches).
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { writeFileSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Ollama } from 'ollama';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const RESULTS_FILE = resolve(__dirname, '../../logs/speed-bench.json');

const OLLAMA_HOST = 'http://localhost:11434';

const PROMPTS = [
  'Explain quantum entanglement in one paragraph for a smart non-physicist.',
  'Compare microservices versus monolithic architecture for a 50-engineer SaaS company.',
  'Write a TypeScript function that performs binary search and explain its time complexity.',
];

interface RunResult {
  ok: boolean;
  totalMs: number;
  loadMs: number;
  evalCount: number;
  evalMs: number;
  evalTokPerSec: number;
  promptEvalCount: number;
  promptEvalMs: number;
  text: string;
  error?: string;
}

interface ConfigSpec {
  label: string;
  model: string;
  options: Record<string, any>; // num_gpu, num_thread, num_batch, etc.
}

async function runOnce(spec: ConfigSpec, prompt: string, timeoutMs = 600_000): Promise<RunResult> {
  const client = new Ollama({ host: OLLAMA_HOST });
  const start = performance.now();
  try {
    const timeout = new Promise<never>((_, rej) =>
      setTimeout(() => rej(new Error(`timeout after ${timeoutMs}ms`)), timeoutMs)
    );
    const r = await Promise.race([
      client.generate({
        model: spec.model,
        prompt,
        options: { num_predict: 80, temperature: 0.0, ...spec.options },
        keep_alive: 0, // unload after each call so the next config truly tests its own load
      }),
      timeout,
    ]);
    const totalMs = performance.now() - start;
    const ns = (n: number) => n / 1e6;
    return {
      ok: true,
      totalMs,
      loadMs: ns(r.load_duration ?? 0),
      evalCount: r.eval_count ?? 0,
      evalMs: ns(r.eval_duration ?? 0),
      evalTokPerSec: r.eval_count && r.eval_duration ? r.eval_count / (r.eval_duration / 1e9) : 0,
      promptEvalCount: r.prompt_eval_count ?? 0,
      promptEvalMs: ns(r.prompt_eval_duration ?? 0),
      text: r.response,
    };
  } catch (err) {
    return {
      ok: false,
      totalMs: performance.now() - start,
      loadMs: 0,
      evalCount: 0,
      evalMs: 0,
      evalTokPerSec: 0,
      promptEvalCount: 0,
      promptEvalMs: 0,
      text: '',
      error: (err as Error).message,
    };
  }
}

async function runConfig(spec: ConfigSpec): Promise<{ spec: ConfigSpec; runs: RunResult[]; meanTokPerSec: number; meanLoadMs: number; }> {
  console.log(`\n--- ${spec.label} (model: ${spec.model}) ---`);
  console.log(`    options: ${JSON.stringify(spec.options)}`);
  const runs: RunResult[] = [];
  for (let i = 0; i < PROMPTS.length; i++) {
    process.stdout.write(`    [${i + 1}/${PROMPTS.length}] `);
    const r = await runOnce(spec, PROMPTS[i]);
    runs.push(r);
    if (r.ok) {
      console.log(`OK load=${(r.loadMs/1000).toFixed(1)}s eval=${r.evalCount}tok/${(r.evalMs/1000).toFixed(1)}s = ${r.evalTokPerSec.toFixed(2)} tok/s`);
    } else {
      console.log(`FAIL ${r.error}`);
    }
  }
  const ok = runs.filter(r => r.ok);
  const mean = ok.length ? ok.reduce((a, r) => a + r.evalTokPerSec, 0) / ok.length : 0;
  const meanLoad = ok.length ? ok.reduce((a, r) => a + r.loadMs, 0) / ok.length : 0;
  console.log(`    mean: ${mean.toFixed(2)} tok/s, mean load: ${(meanLoad/1000).toFixed(1)}s`);
  return { spec, runs, meanTokPerSec: mean, meanLoadMs: meanLoad };
}

async function main() {
  const args = process.argv.slice(2);
  const skipTo = args.find(a => a.startsWith('--from='))?.replace('--from=', '');

  console.log('=== Speed Bench — Phase 7+ ===');
  console.log(`Workload: ${PROMPTS.length} prompts × n_predict=80\n`);

  const allConfigs: ConfigSpec[] = [
    // ─── 32B rounds ─────────────────────────────────────────────────────
    { label: 'R1 [32B] CPU baseline (default)',           model: 'qwen2.5:32b', options: { num_gpu: 0 } },
    { label: 'R2 [32B] CPU + num_thread=16 (physical)',   model: 'qwen2.5:32b', options: { num_gpu: 0, num_thread: 16 } },
    { label: 'R3 [32B] CPU + num_thread=32 (logical)',    model: 'qwen2.5:32b', options: { num_gpu: 0, num_thread: 32 } },
    { label: 'R4a [32B] partial GPU num_gpu=10',          model: 'qwen2.5:32b', options: { num_gpu: 10, num_thread: 16 } },
    { label: 'R4b [32B] partial GPU num_gpu=20',          model: 'qwen2.5:32b', options: { num_gpu: 20, num_thread: 16 } },
    { label: 'R4c [32B] partial GPU num_gpu=30',          model: 'qwen2.5:32b', options: { num_gpu: 30, num_thread: 16 } },

    // ─── 14B rounds ─────────────────────────────────────────────────────
    { label: 'R5 [14B] CPU baseline (num_gpu=0)',         model: 'qwen2.5:14b', options: { num_gpu: 0, num_thread: 16 } },
    { label: 'R6 [14B] full GPU offload (num_gpu=999)',   model: 'qwen2.5:14b', options: { num_gpu: 999, num_thread: 16 } },
  ];

  const startIdx = skipTo ? allConfigs.findIndex(c => c.label.startsWith(skipTo)) : 0;
  const configs = startIdx >= 0 ? allConfigs.slice(startIdx) : allConfigs;

  const results: any[] = [];
  for (const spec of configs) {
    const r = await runConfig(spec);
    results.push(r);
  }

  console.log('\n\n=== SUMMARY (eval tok/s, higher is better) ===');
  const sorted = [...results].sort((a, b) => b.meanTokPerSec - a.meanTokPerSec);
  for (const r of sorted) {
    console.log(`  ${r.spec.label.padEnd(45)} ${r.meanTokPerSec.toFixed(2).padStart(6)} tok/s   load=${(r.meanLoadMs/1000).toFixed(1)}s`);
  }

  // Speedup vs baseline
  const baseline32B = results.find(r => r.spec.label.includes('R1 [32B] CPU baseline'));
  const baseline14B = results.find(r => r.spec.label.includes('R5 [14B] CPU baseline'));
  if (baseline32B) {
    console.log('\n32B speedups vs CPU baseline:');
    for (const r of results.filter(r => r.spec.model === 'qwen2.5:32b' && r !== baseline32B)) {
      const x = r.meanTokPerSec / baseline32B.meanTokPerSec;
      console.log(`  ${r.spec.label.padEnd(45)} ${x.toFixed(2)}x`);
    }
  }
  if (baseline14B) {
    console.log('\n14B speedups vs CPU baseline:');
    for (const r of results.filter(r => r.spec.model === 'qwen2.5:14b' && r !== baseline14B)) {
      const x = r.meanTokPerSec / baseline14B.meanTokPerSec;
      console.log(`  ${r.spec.label.padEnd(45)} ${x.toFixed(2)}x`);
    }
  }

  mkdirSync(dirname(RESULTS_FILE), { recursive: true });
  writeFileSync(RESULTS_FILE, JSON.stringify({
    completedAt: new Date().toISOString(),
    workload: PROMPTS,
    configs: results.map(r => ({
      label: r.spec.label,
      model: r.spec.model,
      options: r.spec.options,
      meanTokPerSec: r.meanTokPerSec,
      meanLoadMs: r.meanLoadMs,
      runs: r.runs.map((run: RunResult) => ({
        ok: run.ok, evalCount: run.evalCount, evalMs: run.evalMs,
        evalTokPerSec: run.evalTokPerSec, loadMs: run.loadMs,
        promptEvalMs: run.promptEvalMs, error: run.error,
      })),
    })),
  }, null, 2));
  console.log(`\nResults: ${RESULTS_FILE}`);
}

main().catch(e => { console.error('Fatal:', e); process.exit(1); });
