/**
 * Phone Mini Benchmark — Phase 7
 *
 * Tight comparison: 3 short factual queries through:
 *   - Phone llama-server (qwen2.5:1.5b, no GPU accel)
 *   - PC Ollama 1.5B GPU
 *   - API ground truth
 *
 * Phone is ~8s/tok, so we cap n_predict=50 to keep total runtime bounded.
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { writeFileSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Ollama } from 'ollama';
import Anthropic from '@anthropic-ai/sdk';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const RESULTS = resolve(__dirname, '../../logs/phone-bench.json');

const PHONE = 'http://localhost:8090';
const OLLAMA = 'http://localhost:11434';
const API_MODEL = 'claude-haiku-4-5-20251001';

const QUERIES = [
  'The capital of Japan is',
  'Two plus two equals',
  'Water boils at temperature',
];

function coh(text: string): number {
  if (!text) return 0;
  const w = text.split(/\s+/).filter(Boolean);
  if (!w.length) return 0;
  const sent = /[.!?]/.test(text) ? 0.3 : 0;
  const div = Math.min(0.4, (new Set(w.map(x => x.toLowerCase())).size / w.length) * 0.5);
  const len = Math.min(0.3, w.length / 100);
  return Math.min(1, sent + div + len);
}

interface R { ok: boolean; latencyMs: number; tokens: number; coherence: number; text: string; tokPerSec?: number; }

async function runPhone(prompt: string): Promise<R> {
  const start = performance.now();
  try {
    // Use streaming mode — non-streaming /completion produces zero bytes until done,
    // which trips Node fetch's default body timeout on slow phone generation.
    const resp = await fetch(`${PHONE}/completion`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, n_predict: 50, temperature: 0.0, stream: true }),
      signal: AbortSignal.timeout(900_000),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    if (!resp.body) throw new Error('no body');

    let text = '';
    let tokens = 0;
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data: ')) continue;
        const payload = trimmed.slice(6);
        if (payload === '[DONE]') continue;
        try {
          const evt = JSON.parse(payload) as { content?: string; stop?: boolean };
          if (evt.content) { text += evt.content; tokens++; }
        } catch { /* skip malformed */ }
      }
    }
    const latencyMs = performance.now() - start;
    const tokPerSec = tokens / (latencyMs / 1000);
    return { ok: true, latencyMs, tokens, coherence: coh(text), text, tokPerSec };
  } catch (e) {
    return { ok: false, latencyMs: performance.now() - start, tokens: 0, coherence: 0, text: '' };
  }
}

async function runOllama(model: string, prompt: string): Promise<R> {
  const c = new Ollama({ host: OLLAMA });
  const start = performance.now();
  try {
    const r = await c.generate({ model, prompt, options: { num_predict: 50, temperature: 0.0 }, keep_alive: 0 });
    const latencyMs = performance.now() - start;
    const tokens = r.response.split(/\s+/).filter(Boolean).length;
    return { ok: true, latencyMs, tokens, coherence: coh(r.response), text: r.response, tokPerSec: tokens / (latencyMs / 1000) };
  } catch (e) {
    return { ok: false, latencyMs: performance.now() - start, tokens: 0, coherence: 0, text: '' };
  }
}

async function runAPI(prompt: string): Promise<R> {
  const c = new Anthropic();
  const start = performance.now();
  try {
    const r = await c.messages.create({
      model: API_MODEL,
      max_tokens: 50,
      messages: [{ role: 'user', content: prompt }],
    });
    const latencyMs = performance.now() - start;
    const text = r.content.filter((b): b is Anthropic.TextBlock => b.type === 'text').map(b => b.text).join('');
    const tokens = text.split(/\s+/).filter(Boolean).length;
    return { ok: true, latencyMs, tokens, coherence: coh(text), text, tokPerSec: tokens / (latencyMs / 1000) };
  } catch (e) {
    return { ok: false, latencyMs: performance.now() - start, tokens: 0, coherence: 0, text: '' };
  }
}

async function main() {
  console.log('=== Phone Mini Bench — Phase 7 ===');
  console.log('3 short factual queries × 3 paths (phone, PC-GPU 1.5B, API)\n');

  const results: any[] = [];
  const overallStart = performance.now();

  for (let i = 0; i < QUERIES.length; i++) {
    const q = QUERIES[i];
    console.log(`\n[${i + 1}/${QUERIES.length}] "${q}"`);

    process.stdout.write('  Phone (qwen2.5:1.5b on 4GB Moto)... ');
    const p = await runPhone(q);
    console.log(p.ok ? `OK ${(p.latencyMs / 1000).toFixed(1)}s, ${p.tokens}tok, ${p.tokPerSec?.toFixed(2)}tok/s, coh=${p.coherence.toFixed(2)}` : 'FAIL');

    process.stdout.write('  PC-GPU (qwen2.5:1.5b)................... ');
    const g = await runOllama('qwen2.5:1.5b', q);
    console.log(g.ok ? `OK ${(g.latencyMs / 1000).toFixed(1)}s, ${g.tokens}tok, ${g.tokPerSec?.toFixed(2)}tok/s, coh=${g.coherence.toFixed(2)}` : 'FAIL');

    process.stdout.write('  API (Haiku)............................. ');
    const a = await runAPI(q);
    console.log(a.ok ? `OK ${(a.latencyMs / 1000).toFixed(1)}s, ${a.tokens}tok, ${a.tokPerSec?.toFixed(2)}tok/s, coh=${a.coherence.toFixed(2)}` : 'FAIL');

    results.push({ prompt: q, phone: p, pcGpu: g, api: a });
  }

  const totalMs = performance.now() - overallStart;

  console.log('\n=== SUMMARY ===');
  console.log(`Total wall time: ${(totalMs / 1000 / 60).toFixed(1)} min`);

  const avg = (xs: number[]) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;

  const phone = results.map(r => r.phone).filter(r => r.ok);
  const pc = results.map(r => r.pcGpu).filter(r => r.ok);
  const api = results.map(r => r.api).filter(r => r.ok);

  console.log(`\nSuccess: phone=${phone.length}/${results.length}, pc=${pc.length}/${results.length}, api=${api.length}/${results.length}`);
  console.log(`\nAvg latency:  phone ${(avg(phone.map(r => r.latencyMs)) / 1000).toFixed(1)}s | pc ${(avg(pc.map(r => r.latencyMs)) / 1000).toFixed(1)}s | api ${(avg(api.map(r => r.latencyMs)) / 1000).toFixed(1)}s`);
  console.log(`Avg tok/s:    phone ${avg(phone.map(r => r.tokPerSec)).toFixed(2)}    | pc ${avg(pc.map(r => r.tokPerSec)).toFixed(2)}    | api ${avg(api.map(r => r.tokPerSec)).toFixed(2)}`);
  console.log(`Avg coherence: phone ${avg(phone.map(r => r.coherence)).toFixed(3)} | pc ${avg(pc.map(r => r.coherence)).toFixed(3)} | api ${avg(api.map(r => r.coherence)).toFixed(3)}`);

  if (phone.length && pc.length) {
    const phoneSlowdown = avg(phone.map(r => r.latencyMs)) / avg(pc.map(r => r.latencyMs));
    console.log(`\nPhone is ${phoneSlowdown.toFixed(1)}× slower than PC-GPU on the same 1.5B model.`);
  }

  mkdirSync(dirname(RESULTS), { recursive: true });
  writeFileSync(RESULTS, JSON.stringify({ completedAt: new Date().toISOString(), totalMs, results }, null, 2));
  console.log(`\nResults: ${RESULTS}`);
}

main().catch(e => { console.error('Fatal:', e); process.exit(1); });
