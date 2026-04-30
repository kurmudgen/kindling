/**
 * Swarm Benchmark — Phase 7
 *
 * Tests the core research hypothesis: can a CPU+RAM node produce answers
 * comparable to API ground truth, within its (long) latency budget?
 *
 * For each query, runs:
 *   1. GPU specialist  — small/medium model, fast
 *   2. CPU specialist  — large model in RAM, slow
 *   3. API ground truth — Anthropic Haiku, fast & high quality
 *
 * Records: latency, token count, coherence (same simple estimator as shadow.ts).
 * Output: logs/swarm-bench.jsonl + summary table to stdout.
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { writeFileSync, appendFileSync, mkdirSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Ollama } from 'ollama';
import Anthropic from '@anthropic-ai/sdk';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const RESULTS_FILE = resolve(__dirname, '../../logs/swarm-bench.jsonl');
const SUMMARY_FILE = resolve(__dirname, '../../logs/swarm-bench-summary.json');

// ─── Bench corpus — 2 per domain × 6 domains = 12 queries ──────────────────

interface BenchQuery {
  id: string;
  domain: 'factual' | 'code' | 'reasoning' | 'math' | 'creative' | 'general';
  prompt: string;
}

const QUERIES: BenchQuery[] = [
  { id: 'fact-1', domain: 'factual',   prompt: 'What is the capital of Japan?' },
  { id: 'fact-2', domain: 'factual',   prompt: 'Who invented the telephone, and in what year?' },

  { id: 'code-1', domain: 'code',      prompt: 'Write a TypeScript function that finds the longest palindromic substring of a given string. Include time complexity analysis.' },
  { id: 'code-2', domain: 'code',      prompt: 'Implement a thread-safe LRU cache in Python with O(1) get and put operations.' },

  { id: 'rsn-1',  domain: 'reasoning', prompt: 'Analyze the trade-offs between TCP and UDP for a real-time multiplayer game backend handling 10,000 concurrent players.' },
  { id: 'rsn-2',  domain: 'reasoning', prompt: 'Compare event-sourcing versus traditional CRUD persistence for a banking ledger. Discuss consistency, auditability, and operational cost.' },

  { id: 'math-1', domain: 'math',      prompt: 'Prove that the square root of 2 is irrational using a proof by contradiction.' },
  { id: 'math-2', domain: 'math',      prompt: 'Derive the formula for the variance of a sum of two independent random variables, then explain why independence is required.' },

  { id: 'crv-1',  domain: 'creative',  prompt: 'Write a 200-word short story about a lighthouse keeper who discovers their lamp is communicating with something below the waves.' },
  { id: 'crv-2',  domain: 'creative',  prompt: 'Compose a four-stanza poem about an old terminal in an abandoned data center, written in iambic tetrameter.' },

  { id: 'gen-1',  domain: 'general',   prompt: 'What are three habits that demonstrably improve sleep quality, and what is the mechanism behind each?' },
  { id: 'gen-2',  domain: 'general',   prompt: 'Summarize, in plain language, why electric vehicles tend to be more efficient than internal combustion vehicles.' },
];

// ─── Node assignments per domain ───────────────────────────────────────────

interface NodeSpec {
  id: string;
  model: string;
  hardware: 'gpu' | 'cpu';
}

// GPU specialist per domain — only models that fit in 6GB VRAM at Q4
const GPU_NODE: Record<BenchQuery['domain'], NodeSpec> = {
  factual:   { id: 'fast-general',    model: 'qwen2.5:1.5b',     hardware: 'gpu' },
  general:   { id: 'fast-general',    model: 'qwen2.5:1.5b',     hardware: 'gpu' },
  code:      { id: 'code-fast',       model: 'qwen2.5-coder:7b', hardware: 'gpu' },
  reasoning: { id: 'reasoning-fast',  model: 'qwen2.5:7b',       hardware: 'gpu' },
  math:      { id: 'reasoning-fast',  model: 'qwen2.5:7b',       hardware: 'gpu' },
  creative:  { id: 'reasoning-fast',  model: 'qwen2.5:7b',       hardware: 'gpu' },
};

// CPU node — qwen2.5:32b for everything (70B dropped — too big for this RAM)
const CPU_NODE: Record<BenchQuery['domain'], NodeSpec> = {
  factual:   { id: 'general-large',   model: 'qwen2.5:32b',      hardware: 'cpu' },
  general:   { id: 'general-large',   model: 'qwen2.5:32b',      hardware: 'cpu' },
  code:      { id: 'general-large',   model: 'qwen2.5:32b',      hardware: 'cpu' },
  reasoning: { id: 'general-large',   model: 'qwen2.5:32b',      hardware: 'cpu' },
  math:      { id: 'general-large',   model: 'qwen2.5:32b',      hardware: 'cpu' },
  creative:  { id: 'general-large',   model: 'qwen2.5:32b',      hardware: 'cpu' },
};

const OLLAMA_HOST = 'http://localhost:11434';
const API_MODEL = 'claude-haiku-4-5-20251001';
const PHONE_HOST = 'http://localhost:8080'; // adb forward tcp:8080 tcp:8080 for phone llama-server
const PHONE_MODEL_TAG = 'qwen2.5-1.5b-instruct'; // for reporting only — llama-server uses whatever's loaded

// ─── Coherence estimator (matches shadow.ts) ────────────────────────────────

function estimateCoherence(text: string): number {
  if (!text || text.length === 0) return 0;
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length === 0) return 0;
  const hasSentences = /[.!?]/.test(text) ? 0.3 : 0;
  const uniqueRatio = new Set(words.map(w => w.toLowerCase())).size / words.length;
  const diversityScore = Math.min(0.4, uniqueRatio * 0.5);
  const lengthScore = Math.min(0.3, words.length / 100);
  return Math.min(1, hasSentences + diversityScore + lengthScore);
}

// ─── Runners ────────────────────────────────────────────────────────────────

interface RunResult {
  ok: boolean;
  latencyMs: number;
  tokens: number;
  coherence: number;
  text: string;
  error?: string;
}

async function runOllama(model: string, prompt: string, timeoutMs: number): Promise<RunResult> {
  const client = new Ollama({ host: OLLAMA_HOST });
  const start = performance.now();
  try {
    const timeout = new Promise<never>((_, rej) =>
      setTimeout(() => rej(new Error(`timeout after ${timeoutMs}ms`)), timeoutMs)
    );
    const resp = await Promise.race([
      client.generate({
        model,
        prompt,
        options: { num_predict: 1024 },
        keep_alive: 0, // unload immediately so the next call has full RAM/VRAM
      }),
      timeout,
    ]);
    const latencyMs = performance.now() - start;
    const text = resp.response;
    const tokens = text.split(/\s+/).filter(Boolean).length;
    return { ok: true, latencyMs, tokens, coherence: estimateCoherence(text), text };
  } catch (err) {
    return {
      ok: false,
      latencyMs: performance.now() - start,
      tokens: 0,
      coherence: 0,
      text: '',
      error: (err as Error).message,
    };
  }
}

async function runPhone(prompt: string, timeoutMs: number): Promise<RunResult> {
  const start = performance.now();
  try {
    const resp = await Promise.race([
      fetch(`${PHONE_HOST}/completion`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, n_predict: 1024, temperature: 0.7 }),
      }),
      new Promise<never>((_, rej) =>
        setTimeout(() => rej(new Error(`phone timeout after ${timeoutMs}ms`)), timeoutMs)
      ),
    ]);
    if (!resp.ok) throw new Error(`phone HTTP ${resp.status}`);
    const data = await resp.json() as { content: string };
    const latencyMs = performance.now() - start;
    const text = data.content;
    const tokens = text.split(/\s+/).filter(Boolean).length;
    return { ok: true, latencyMs, tokens, coherence: estimateCoherence(text), text };
  } catch (err) {
    return {
      ok: false,
      latencyMs: performance.now() - start,
      tokens: 0,
      coherence: 0,
      text: '',
      error: (err as Error).message,
    };
  }
}

async function runAPI(prompt: string): Promise<RunResult> {
  const client = new Anthropic();
  const start = performance.now();
  try {
    const resp = await client.messages.create({
      model: API_MODEL,
      max_tokens: 1024,
      messages: [{ role: 'user', content: prompt }],
    });
    const latencyMs = performance.now() - start;
    const text = resp.content
      .filter((b): b is Anthropic.TextBlock => b.type === 'text')
      .map(b => b.text)
      .join('');
    const tokens = text.split(/\s+/).filter(Boolean).length;
    return { ok: true, latencyMs, tokens, coherence: estimateCoherence(text), text };
  } catch (err) {
    return {
      ok: false,
      latencyMs: performance.now() - start,
      tokens: 0,
      coherence: 0,
      text: '',
      error: (err as Error).message,
    };
  }
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
  // --filter=domain1,domain2 lets us rerun only specific domains
  const filterArg = process.argv.find(a => a.startsWith('--filter='));
  const filterDomains = filterArg ? filterArg.replace('--filter=', '').split(',') : null;
  const queries = filterDomains
    ? QUERIES.filter(q => filterDomains.includes(q.domain))
    : QUERIES;

  console.log('=== Swarm Benchmark — Phase 7 ===');
  console.log(`Queries: ${queries.length}${filterDomains ? ` (filtered: ${filterDomains.join(', ')})` : ''} (${new Set(queries.map(q => q.domain)).size} domains)`);
  console.log(`Comparing: GPU specialist vs CPU node vs API ground truth\n`);

  if (!process.env.ANTHROPIC_API_KEY) {
    console.error('ANTHROPIC_API_KEY missing — cannot run API ground truth');
    process.exit(1);
  }

  mkdirSync(dirname(RESULTS_FILE), { recursive: true });
  if (existsSync(RESULTS_FILE)) {
    writeFileSync(RESULTS_FILE, ''); // truncate
  }

  const results: any[] = [];
  const overallStart = performance.now();

  for (let i = 0; i < queries.length; i++) {
    const q = queries[i];
    const gpu = GPU_NODE[q.domain];
    const cpu = CPU_NODE[q.domain];

    console.log(`\n[${i + 1}/${queries.length}] ${q.id} (${q.domain}) — "${q.prompt.slice(0, 70)}..."`);
    console.log(`  GPU: ${gpu.model}  |  CPU: ${cpu.model}  |  API: ${API_MODEL}`);

    // GPU first (fastest, frees VRAM before CPU run)
    process.stdout.write(`  GPU running... `);
    const gpuResult = await runOllama(gpu.model, q.prompt, 90_000);
    if (gpuResult.ok) {
      console.log(`OK ${Math.round(gpuResult.latencyMs)}ms, ${gpuResult.tokens}tok, coh=${gpuResult.coherence.toFixed(2)}`);
    } else {
      console.log(`FAIL ${gpuResult.error}`);
    }

    // CPU node — long budget (32B ~3min, 70B ~10min worst case)
    process.stdout.write(`  CPU running... `);
    const cpuResult = await runOllama(cpu.model, q.prompt, 900_000);
    if (cpuResult.ok) {
      console.log(`OK ${Math.round(cpuResult.latencyMs)}ms, ${cpuResult.tokens}tok, coh=${cpuResult.coherence.toFixed(2)}`);
    } else {
      console.log(`FAIL ${cpuResult.error}`);
    }

    // API ground truth
    process.stdout.write(`  API running... `);
    const apiResult = await runAPI(q.prompt);
    if (apiResult.ok) {
      console.log(`OK ${Math.round(apiResult.latencyMs)}ms, ${apiResult.tokens}tok, coh=${apiResult.coherence.toFixed(2)}`);
    } else {
      console.log(`FAIL ${apiResult.error}`);
    }

    const row = {
      id: q.id,
      domain: q.domain,
      prompt: q.prompt,
      gpu: { ...gpu, ...gpuResult, text: gpuResult.text.slice(0, 500) },
      cpu: { ...cpu, ...cpuResult, text: cpuResult.text.slice(0, 500) },
      api: { model: API_MODEL, ...apiResult, text: apiResult.text.slice(0, 500) },
      gpuVsApiCoherenceDelta: gpuResult.ok && apiResult.ok ? apiResult.coherence - gpuResult.coherence : null,
      cpuVsApiCoherenceDelta: cpuResult.ok && apiResult.ok ? apiResult.coherence - cpuResult.coherence : null,
    };
    results.push(row);
    appendFileSync(RESULTS_FILE, JSON.stringify(row) + '\n');
  }

  const totalMs = performance.now() - overallStart;

  // ─── Summary ──────────────────────────────────────────────────────────────
  console.log('\n\n=== SUMMARY ===');
  console.log(`Total wall time: ${(totalMs / 1000 / 60).toFixed(1)} min`);

  const okGpu = results.filter(r => r.gpu.ok);
  const okCpu = results.filter(r => r.cpu.ok);
  const okApi = results.filter(r => r.api.ok);

  console.log(`\nSuccess rates:  GPU ${okGpu.length}/${results.length}  |  CPU ${okCpu.length}/${results.length}  |  API ${okApi.length}/${results.length}`);

  const avg = (arr: number[]) => arr.length === 0 ? 0 : arr.reduce((a, b) => a + b, 0) / arr.length;

  const gpuLatency = avg(okGpu.map(r => r.gpu.latencyMs));
  const cpuLatency = avg(okCpu.map(r => r.cpu.latencyMs));
  const apiLatency = avg(okApi.map(r => r.api.latencyMs));

  const gpuCoh = avg(okGpu.map(r => r.gpu.coherence));
  const cpuCoh = avg(okCpu.map(r => r.cpu.coherence));
  const apiCoh = avg(okApi.map(r => r.api.coherence));

  console.log(`\nAvg latency:    GPU ${(gpuLatency/1000).toFixed(1)}s  |  CPU ${(cpuLatency/1000).toFixed(1)}s  |  API ${(apiLatency/1000).toFixed(1)}s`);
  console.log(`Avg coherence:  GPU ${gpuCoh.toFixed(3)}  |  CPU ${cpuCoh.toFixed(3)}  |  API ${apiCoh.toFixed(3)}`);
  console.log(`Avg tokens:     GPU ${avg(okGpu.map(r => r.gpu.tokens)).toFixed(0)}  |  CPU ${avg(okCpu.map(r => r.cpu.tokens)).toFixed(0)}  |  API ${avg(okApi.map(r => r.api.tokens)).toFixed(0)}`);

  // Per-domain breakdown
  console.log('\nPer-domain coherence (GPU / CPU / API):');
  const domains = Array.from(new Set(results.map(r => r.domain)));
  for (const d of domains) {
    const subset = results.filter(r => r.domain === d);
    const g = avg(subset.filter(r => r.gpu.ok).map(r => r.gpu.coherence));
    const c = avg(subset.filter(r => r.cpu.ok).map(r => r.cpu.coherence));
    const a = avg(subset.filter(r => r.api.ok).map(r => r.api.coherence));
    console.log(`  ${d.padEnd(10)}  GPU ${g.toFixed(3)}  CPU ${c.toFixed(3)}  API ${a.toFixed(3)}`);
  }

  // The thesis question: where does CPU close the gap with API?
  console.log('\nCPU-vs-API coherence delta (negative = CPU close to API):');
  for (const d of domains) {
    const subset = results.filter(r => r.domain === d);
    const deltas = subset.map(r => r.cpuVsApiCoherenceDelta).filter(x => x !== null) as number[];
    if (deltas.length > 0) {
      console.log(`  ${d.padEnd(10)}  delta=${avg(deltas).toFixed(3)}`);
    }
  }

  const summary = {
    completedAt: new Date().toISOString(),
    totalQueries: results.length,
    totalWallMs: totalMs,
    successRates: {
      gpu: okGpu.length / results.length,
      cpu: okCpu.length / results.length,
      api: okApi.length / results.length,
    },
    avgLatencyMs: { gpu: gpuLatency, cpu: cpuLatency, api: apiLatency },
    avgCoherence: { gpu: gpuCoh, cpu: cpuCoh, api: apiCoh },
    perDomain: domains.map(d => {
      const subset = results.filter(r => r.domain === d);
      return {
        domain: d,
        gpu: { coherence: avg(subset.filter(r => r.gpu.ok).map(r => r.gpu.coherence)), latencyMs: avg(subset.filter(r => r.gpu.ok).map(r => r.gpu.latencyMs)) },
        cpu: { coherence: avg(subset.filter(r => r.cpu.ok).map(r => r.cpu.coherence)), latencyMs: avg(subset.filter(r => r.cpu.ok).map(r => r.cpu.latencyMs)) },
        api: { coherence: avg(subset.filter(r => r.api.ok).map(r => r.api.coherence)), latencyMs: avg(subset.filter(r => r.api.ok).map(r => r.api.latencyMs)) },
        cpuVsApiDelta: avg(subset.map(r => r.cpuVsApiCoherenceDelta).filter(x => x !== null) as number[]),
      };
    }),
  };
  writeFileSync(SUMMARY_FILE, JSON.stringify(summary, null, 2));
  console.log(`\nResults: ${RESULTS_FILE}\nSummary: ${SUMMARY_FILE}`);
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
