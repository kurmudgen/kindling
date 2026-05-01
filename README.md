# Kindling

A research project trying to figure out: **can a small/weak GPU be made comparable to a bigger one by leaning on CPU and RAM?** I know I can't actually replace VRAM, but I wanted to see how close I could get and where it falls apart.

This is not a product. It's a build log with code attached. Some of it works. Some of it doesn't. The interesting parts are the things I learned while it failed.

## What's actually here

A 3-tier LLM router that tries to do most work cheaply and only escalate when it has to:

```
prompt → valence scorer → Tier 1 (small Ollama, always hot)
                              ↓ low confidence
                         Tier 2 (medium Ollama + API fallback)
                              ↓ low confidence
                         Tier 3 (Anthropic API)
                              ↑
                       sleep analyst reviews logs during idle,
                       retrains an ML classifier that decides
                       when to escalate next time
```

The novel-ish bit is **the meta-confidence loop**: every Nth query, the same prompt also gets sent to the Anthropic API as a teacher signal. We compare local-vs-API output and label whether escalation was actually needed. That builds a training dataset, which we use to retrain a logistic-regression classifier that the router consults on the next query. It's been retraining itself in the background and is currently at ~92% accuracy on 89 examples.

There's also a **swarm coordinator** that routes queries to specialist nodes by detected domain (code/math/reasoning/creative/factual/general), with hardware-aware fallback chains. This is where things got interesting in Phase 7 — see "Findings."

## What I built (phase by phase)

| Phase | What | Status |
|-------|------|--------|
| 1-3 | Core 3-tier router, valence scoring, escalation signals | done |
| 3.5 | 55-query benchmark | done |
| 4 | Shadow eval (API as teacher) + ML meta-confidence classifier | done, 92% accuracy |
| 5A | Continuous retraining loop (collect → retrain → hot reload) | done |
| 5B | Per-token streaming routing | done |
| 6 | Swarm coordinator + node registry + domain detector + collect-loop | done |
| 7 | Bench validation (desktop swarm + mobile node) | done |

## Findings (the actually interesting part)

### CPU 32B is competitive on the easy stuff
On factual / creative / general queries, the CPU-resident 32B model produced output within 0.06 coherence of the API at ~30× the latency (122s vs 4s). That's not a free lunch but it's a real one — bulk RAM does serve a purpose for non-reasoning work.

### Reasoning + math broke local
Nothing 14B+ fit in my 6GB VRAM. The 14B got CPU-served and timed out. The 70B caused RAM thrashing bad enough I had to kill it mid-bench. So those domains have to fall through to the API on this hardware.

### "Swarm on one machine" doesn't really exist
Every "node" in `swarm.json` competes for the same Ollama instance, the same RAM, the same model-load queue. They're not independent lanes — they're sequential queue slots. The architecture only starts to make sense with separate hardware. This was the most useful negative finding from Phase 7.

### Phones can run LLMs but not for live queries
I sideloaded Termux on a 4GB Moto G Play, got `qwen2.5:1.5b` running via llama-cpp, exposed the HTTP server back to the PC. It works. **It runs at 0.12 tok/s — about 7 minutes per simple query.** That's 175× slower than the same model on the PC GPU. So phones are real as an offline/deferred lane (queued background work, dream-task runners), not as live inference nodes. The mobile-LLM future probably needs NPU acceleration to be practical.

### Coherence as a quality metric is noisy
Current "coherence" is just sentence punctuation + word diversity + length. Good enough to spot total failures, not good enough to draw fine-grained quality conclusions from. Future benches should use LLM-as-judge.

## Quick start

```bash
git clone https://github.com/kurmudgen/kindling
cd kindling
npm install
ollama pull qwen2.5:1.5b
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env for Tier 3 + shadow eval
npm run start
```

## Commands

| Command | What it does |
|---------|--------------|
| `npm run start` | Interactive REPL |
| `npm run bench` | 55-query benchmark suite |
| `npm test` | Unit tests |
| `npx tsx src/tools/swarm-bench.ts` | 12-query GPU/CPU/API comparison |
| `npx tsx src/tools/phone-bench.ts` | Phone vs PC-GPU vs API (needs phone setup) |
| `npx tsx src/tools/collect-loop.ts` | Background data collection + auto-retrain |
| `/sleep` | Trigger sleep-stage analysis in REPL |
| `/clear` | Clear conversation context |

## Hardware profiles

| Profile | Target | Tier 1 | Tier 2 |
|---------|--------|--------|--------|
| `default` | General | qwen2.5:1.5b | gemma2:27b (or Haiku) |
| `ddr4-budget` | Limited RAM | qwen2.5:0.5b | qwen2.5:7b |
| `prosumer` | DDR5, no GPU | qwen2.5:3b | gemma2:27b |

Set via `KINDLING_PROFILE`.

## BYOK

You bring your own Anthropic key. Kindling never touches billing. Without a key, Tier 1 and local Tier 2 still work — you just lose the API fallback and shadow-eval-driven classifier improvement.

## What's still open

- README in your hands now, but multi-machine swarm is the obvious next experiment — actually verifying the "independent lanes" idea with separate hardware
- Replace coherence metric with LLM-as-judge so future benches mean more
- Split `swarm.json` into explicit "live" (sub-30s) and "deferred" (minutes-OK) lanes; coordinator picks based on caller's urgency
- HTTP API server so things outside this repo can use Kindling without embedding TS
- Mobile NPU — try MLC Chat / llama.rn / ONNX Runtime Mobile to see if the 175× phone slowdown can be cut

## License

MIT — Formation Labs LLC
