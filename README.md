# Kindling

**Adaptive tiered inference runtime. Low burn, always present. Flares when needed.**

![build](https://img.shields.io/badge/build-passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![version](https://img.shields.io/badge/version-0.1.0-orange)

## What It Is

Most local LLM setups assume you have a GPU with enough VRAM to hold an entire model, or you eat the latency of running everything on CPU. Kindling rejects that tradeoff. It runs a small, fast model on CPU as the always-hot default, then escalates to heavier compute only when confidence signals say the small model can't handle the query. No VRAM required. No GPU required. Just CPU, RAM, and a willingness to let the cheap tier do most of the work.

Kindling monitors four real-time escalation signals during generation — token probability spread, semantic velocity, surprise score, and attention anomaly patterns. When signals cross configurable thresholds, the query escalates to the next tier. When signals stabilize, it drops back down. During idle periods, a "sleep stage" analyst reviews the session's escalation logs and adjusts routing weights so the system gets smarter about when to escalate over time.

## How It Works

```
Query → Valence Scorer → Tier 1 (always hot)
                              ↓ confidence low
                         Tier 2 (warm standby)
                              ↓ confidence low
                         Tier 3 (API / streamed)
                              ↑
                    Sleep Analyst (idle)
                    learns from escalation logs
                    adjusts weights for next session
```

**Tier 1 (Shallow)** — Small Ollama model (default: qwen2.5:1.5b). Always loaded, always fast. Handles the majority of queries. Writes tokens into a speculative buffer while Tier 2 verifies in parallel.

**Tier 2 (Medium)** — Larger Ollama model (default: gemma2:27b) with API fallback to Claude Haiku. Activates when Tier 1 confidence drops. Takes over from the buffer position, not from scratch.

**Tier 3 (Deep)** — Local large model via Ollama NVMe streaming (default: qwen2.5:32b), with Anthropic API fallback. Reserved for high-complexity, high-stakes queries. Gracefully degrades: GPU VRAM → RAM mmap → disk streaming.

**Confidence Router** — Aggregates weighted escalation signals per token, makes escalate/de-escalate decisions against configurable thresholds.

**Speculative Buffer** — Tier 1 generates ahead into a fixed-size buffer. On confirmation, tokens flush to output. On rejection, Tier 2 resumes from the buffer boundary — no wasted work.

**Sleep Analyst** — Runs during idle. Bundles escalation logs, sends them to the API for pattern analysis, and persists routing weight adjustments to `config/learned.json`.

## Hardware Profiles

Set via `KINDLING_PROFILE` env var:

| Profile | Target Hardware | Tier 1 | Tier 2 | Buffer | Escalation |
|---------|----------------|--------|--------|--------|------------|
| `default` | General | qwen2.5:1.5b | gemma2:27b | 4 tokens | Moderate |
| `ddr4-budget` | DDR4, limited RAM | qwen2.5:0.5b | qwen2.5:7b | 2 tokens | Aggressive (escalates earlier) |
| `prosumer` | DDR5, no GPU | qwen2.5:3b | gemma2:27b | 6 tokens | Conservative (tries harder locally) |

## Quick Start

```bash
git clone https://github.com/kurmudgen/kindling
cd kindling
npm install
ollama pull qwen2.5:1.5b
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env for Tier 2 fallback and Tier 3
npm run start
```

## Commands

| Command | Description |
|---------|-------------|
| `npm run start` | Interactive REPL |
| `npm run bench` | 55-query benchmark suite |
| `npm test` | Run unit tests |
| `/sleep` | Trigger sleep stage analysis manually |
| `/clear` | Clear conversation context |

## Bring Your Own Key

Kindling uses BYOK for all API calls — you supply your own Anthropic API key, and Kindling never touches billing. Set `ANTHROPIC_API_KEY` in `.env`. Tier 2 falls back to Claude Haiku when the local model isn't available. Tier 3 uses Claude Sonnet for deep queries. Tier 1 is always local via Ollama and never hits an API. Without an API key, the system still works — Tier 1 and local Tier 2 handle everything, and Tier 3 is simply unavailable.

## Phase Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | **Complete** | Core runtime, API stand-in for Tier 3, sleep stage learning |
| Phase 2 | **Complete** | Logprob-based escalation, local Tier 3 NVMe streaming, cold concept warming |
| Phase 3 | Planned | Sleep state soft weight updates, streaming per-token routing |
| Phase 4 | Planned | Meta-confidence model, full benchmark suite |
| Phase 5 | Planned | Open swarm prototype |

## License

MIT — Formation Labs LLC
