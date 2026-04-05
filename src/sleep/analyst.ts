import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import Anthropic from '@anthropic-ai/sdk';
import pino from 'pino';
import { getConfig } from '../config/config.js';
import { getLogFilePath, getSessionId } from './logger.js';
import { ConceptPrewarmer } from './prewarmer.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const LOGS_DIR = resolve(__dirname, '../../logs');
const SLEEP_DIR = resolve(LOGS_DIR, 'sleep');
const CONFIG_DIR = resolve(__dirname, '../../config');

const log = pino({ level: process.env.KINDLING_LOG_LEVEL ?? 'info' });

export interface SleepAnalysis {
  patterns_identified: string[];
  concepts_to_prewarm: string[];
  routing_adjustments: RoutingAdjustment[];
  confidence_model_notes: string[];
  session_summary: string;
}

export interface RoutingAdjustment {
  signal: string;
  currentWeight: number;
  suggestedWeight: number;
  reason: string;
}

const SYSTEM_PROMPT = `You are analyzing the inference behavior of Kindling, an adaptive tiered LLM runtime. You will receive a session escalation log showing when and why the system escalated queries from cheaper to more expensive compute tiers. Your job is to identify patterns, suggest routing weight adjustments to improve future escalation decisions, and identify concept clusters worth pre-warming. Respond only in the JSON format specified. Be specific and actionable.`;

export class SleepAnalyst {
  private client: Anthropic;
  private model: string;
  private lastQueryTime = Date.now();
  private idleThresholdMs: number;
  private idleCheckInterval: ReturnType<typeof setInterval> | null = null;

  constructor() {
    if (!process.env.ANTHROPIC_API_KEY) {
      throw new Error('SleepAnalyst requires ANTHROPIC_API_KEY');
    }
    this.client = new Anthropic();
    const config = getConfig();
    this.model = config.sleep.apiModel;
    this.idleThresholdMs = config.sleep.idleThresholdMinutes * 60 * 1000;
  }

  /** Call this on every query to track activity */
  recordActivity(): void {
    this.lastQueryTime = Date.now();
  }

  /** Start monitoring for idle periods */
  startIdleMonitor(): void {
    if (this.idleCheckInterval) return;
    this.idleCheckInterval = setInterval(async () => {
      const idleMs = Date.now() - this.lastQueryTime;
      if (idleMs >= this.idleThresholdMs) {
        log.info('Idle threshold reached, running sleep analysis...');
        try {
          await this.analyze();
        } catch (err) {
          log.error({ err }, 'Sleep analysis failed');
        }
        // Reset timer so we don't re-run immediately
        this.lastQueryTime = Date.now();
      }
    }, 60_000); // check every minute
  }

  stopIdleMonitor(): void {
    if (this.idleCheckInterval) {
      clearInterval(this.idleCheckInterval);
      this.idleCheckInterval = null;
    }
  }

  async analyze(): Promise<SleepAnalysis | null> {
    const logPath = getLogFilePath();
    if (!existsSync(logPath)) {
      log.info('No escalation log found, skipping analysis');
      return null;
    }

    const rawLog = readFileSync(logPath, 'utf-8').trim();
    if (!rawLog) {
      log.info('Escalation log empty, skipping analysis');
      return null;
    }

    // Filter to current session
    const sessionId = getSessionId();
    const events = rawLog
      .split('\n')
      .filter(line => line.includes(sessionId));

    if (events.length === 0) {
      log.info('No events for current session, skipping analysis');
      return null;
    }

    const config = getConfig();
    const payload = {
      sessionId,
      eventCount: events.length,
      events: events.map(e => JSON.parse(e)),
      currentWeights: config.escalation,
    };

    const response = await this.client.messages.create({
      model: this.model,
      max_tokens: 2048,
      system: SYSTEM_PROMPT,
      messages: [
        {
          role: 'user',
          content: `Analyze this escalation session and respond with JSON only:\n\n${JSON.stringify(payload, null, 2)}`,
        },
      ],
    });

    const text = response.content
      .filter((b): b is Anthropic.TextBlock => b.type === 'text')
      .map(b => b.text)
      .join('');

    let analysis: SleepAnalysis;
    try {
      // Extract JSON from response (may be wrapped in markdown code blocks)
      // Use balanced brace matching to find the first complete JSON object
      const jsonStart = text.indexOf('{');
      if (jsonStart === -1) throw new Error('No JSON found in response');
      let depth = 0;
      let jsonEnd = -1;
      for (let i = jsonStart; i < text.length; i++) {
        if (text[i] === '{') depth++;
        else if (text[i] === '}') { depth--; if (depth === 0) { jsonEnd = i + 1; break; } }
      }
      if (jsonEnd === -1) throw new Error('Unclosed JSON object in response');
      const jsonMatch = [text.slice(jsonStart, jsonEnd)];
      if (!jsonMatch[0]) throw new Error('Empty JSON object in response');
      analysis = JSON.parse(jsonMatch[0]) as SleepAnalysis;
    } catch (err) {
      log.error({ err, text }, 'Failed to parse sleep analysis response');
      return null;
    }

    // Save analysis to sleep log
    this.saveSleepLog(analysis);

    // Apply routing adjustments
    this.applyAdjustments(analysis.routing_adjustments);

    // Pre-warm concepts identified by the analyst
    if (analysis.concepts_to_prewarm && analysis.concepts_to_prewarm.length > 0) {
      try {
        const prewarmer = new ConceptPrewarmer();
        await prewarmer.prewarm(analysis.concepts_to_prewarm);
      } catch (err) {
        log.warn({ err }, 'Concept pre-warming failed');
      }
    }

    log.info({ summary: analysis.session_summary }, 'Sleep analysis complete');
    return analysis;
  }

  private saveSleepLog(analysis: SleepAnalysis): void {
    if (!existsSync(SLEEP_DIR)) {
      mkdirSync(SLEEP_DIR, { recursive: true });
    }
    const now = new Date();
    const filename = `${now.toISOString().slice(0, 13).replace(/[T:]/g, '-')}.json`;
    writeFileSync(
      resolve(SLEEP_DIR, filename),
      JSON.stringify(analysis, null, 2),
      'utf-8'
    );
  }

  private applyAdjustments(adjustments: RoutingAdjustment[]): void {
    if (adjustments.length === 0) return;

    const learnedPath = resolve(CONFIG_DIR, 'learned.json');
    let learned: Record<string, unknown> = {};
    if (existsSync(learnedPath)) {
      learned = JSON.parse(readFileSync(learnedPath, 'utf-8'));
    }

    const escalation = (learned.escalation ?? {}) as Record<string, number>;

    for (const adj of adjustments) {
      const key = this.signalToConfigKey(adj.signal);
      if (key) {
        escalation[key] = adj.suggestedWeight;
        log.info(
          { signal: adj.signal, from: adj.currentWeight, to: adj.suggestedWeight },
          `Applying routing adjustment: ${adj.reason}`
        );
      }
    }

    learned.escalation = escalation;
    writeFileSync(learnedPath, JSON.stringify(learned, null, 2), 'utf-8');
  }

  private signalToConfigKey(signal: string): string | null {
    const map: Record<string, string> = {
      tokenProbabilitySpread: 'tokenProbabilitySpreadWeight',
      semanticVelocity: 'semanticVelocityWeight',
      surpriseScore: 'surpriseScoreWeight',
      attentionAnomaly: 'attentionAnomalyWeight',
      attentionAnomalyScore: 'attentionAnomalyWeight',
      escalationThreshold: 'escalationThreshold',
      deescalationThreshold: 'deescalationThreshold',
    };
    return map[signal] ?? null;
  }
}
