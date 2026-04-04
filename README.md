# Kindling

Kindling is an adaptive tiered inference runtime for LLMs that runs CPU-primary without hard VRAM dependency. It routes queries across three compute tiers — small local models for fast simple work, medium models for moderate complexity, and deep API models for the hardest problems — using real-time confidence signals and speculative token buffering.

The system learns from its own escalation patterns during idle periods (the "sleep stage"), adjusting routing weights over time so cheaper tiers handle more work without quality loss. Bring your own API key for the deep tier; everything else runs on your local hardware via Ollama.

## Architecture

| Tier | Default Model | Backend | Purpose |
|------|--------------|---------|---------|
| 1 — Shallow | qwen2.5:1.5b | Ollama local | Fast, always-warm. Handles simple queries. |
| 2 — Medium | gemma2:27b | Ollama local + API fallback | Moderate complexity. Falls back to Claude Haiku if local model unavailable. |
| 3 — Deep | claude-sonnet-4-6 | Anthropic API | Complex/high-stakes queries. Phase 2 will add local streaming. |

A **confidence router** monitors escalation signals (token probability spread, semantic velocity, surprise score, attention anomalies) and a **speculative buffer** allows Tier 1 to generate ahead while Tier 2 verifies, minimizing latency on handoffs.

## Quick Start

### Prerequisites
- Node.js 20+
- [Ollama](https://ollama.ai) installed and running
- Pull at least the Tier 1 model: `ollama pull qwen2.5:1.5b`

### Setup
```bash
cd kindling
npm install
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY for Tier 2 fallback and Tier 3
```

### Run
```bash
npm run start
```

Type prompts into the REPL. Commands:
- `/sleep` — trigger manual sleep analysis
- `/clear` — clear conversation context
- `Ctrl+C` — exit

### Benchmark
```bash
npm run bench
```

Results saved to `logs/benchmark/`.

## Hardware Profiles

Set via `KINDLING_PROFILE` env var:

| Profile | Target Hardware | Tier 1 Model | Buffer Size | Escalation Sensitivity |
|---------|----------------|-------------|-------------|----------------------|
| `default` | General | qwen2.5:1.5b | 4 | Moderate |
| `ddr4-budget` | DDR4, limited RAM | qwen2.5:0.5b | 2 | Aggressive (escalates earlier) |
| `prosumer` | DDR5, no GPU | qwen2.5:3b | 6 | Conservative (tries harder locally) |

## BYOK (Bring Your Own Key)

Tier 2 API fallback and Tier 3 use the Anthropic API. Set `ANTHROPIC_API_KEY` in your `.env` file. Without it, Tier 2 is Ollama-only and Tier 3 is unavailable.

Cost tracking is built into Tier 3 responses — check `metadata.costEstimateUSD` in the response.

## Current Status

**Phase 1** — Core runtime complete. Tier 3 uses API as a stand-in for local streaming (coming in Phase 2). Sleep stage analysis operational. Speculative buffer implemented with escalation/de-escalation handoffs.

## License

MIT
