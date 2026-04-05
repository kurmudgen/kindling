/**
 * Three-tier log compaction for sleep analyst input.
 *
 * Tier 1 — Strip redundant fields from events (keep only analysis-relevant data)
 * Tier 2 — Collapse runs of similar events into repetition summaries
 * Tier 3 — Session summary header
 *
 * Goal: a 50-escalation session must fit in under 2000 tokens.
 */

import pino from 'pino';
import type { EscalationEvent } from './logger.js';

const log = pino({ level: process.env.KINDLING_LOG_LEVEL ?? 'info' });

export interface CompactionInput {
  sessionId: string;
  events: EscalationEvent[];
  nonEscalatedQueryCount?: number;
  sessionStartTime?: string;
  sessionEndTime?: string;
}

/** Compact representation of a single escalation event — strips redundant fields. */
export interface CompactEvent {
  t: string;       // timestamp (short)
  v: number;       // valence composite
  cx: number;      // complexity
  u: number;       // urgency (only if non-zero, else omitted)
  s: number;       // stakes (only if non-zero, else omitted)
  c: number;       // confidence at handoff
  to: number;      // escalated to tier
  tok: number;     // tokens before escalation
}

export interface RepetitionSummary {
  rep: true;
  n: number;         // count
  tier: number;      // tier outcome
  avgC: number;      // avg confidence
  vRange: [number, number]; // valence min/max
  cxRange: [number, number]; // complexity min/max
  from: string;      // first timestamp
  to: string;        // last timestamp
}

export type CompactedEntry = CompactEvent | RepetitionSummary;

export interface CompactedLog {
  summaryHeader: string;
  compactedEvents: CompactedEntry[];
  tokenBudgetEstimate: {
    rawCharCount: number;
    compactedCharCount: number;
    rawTokensEst: number;
    compactedTokensEst: number;
    reductionPct: number;
  };
}

function estimateTokens(chars: number): number {
  return Math.ceil(chars / 4);
}

export function compactLog(input: CompactionInput): CompactedLog {
  const rawJson = JSON.stringify({
    sessionId: input.sessionId,
    events: input.events,
  });
  const rawChars = rawJson.length;

  // Tier 1: Strip to essential fields
  const compactEvents = applyFieldStripping(input.events);

  // Tier 2: Collapse repetition
  const afterRepetition = applyRepetitionCollapse(compactEvents, input.events);

  // Tier 3: Summary header
  const summaryHeader = buildSessionSummary(input, input.events);

  const compactedJson = JSON.stringify({
    summary: summaryHeader,
    events: afterRepetition,
  });
  const compactedChars = compactedJson.length;

  const rawTokensEst = estimateTokens(rawChars);
  const compactedTokensEst = estimateTokens(compactedChars);
  const reductionPct =
    rawChars > 0 ? ((rawChars - compactedChars) / rawChars) * 100 : 0;

  log.info(
    {
      rawTokens: rawTokensEst,
      compactedTokens: compactedTokensEst,
      reductionPct: reductionPct.toFixed(1),
    },
    'Log compaction complete'
  );

  return {
    summaryHeader,
    compactedEvents: afterRepetition,
    tokenBudgetEstimate: {
      rawCharCount: rawChars,
      compactedCharCount: compactedChars,
      rawTokensEst,
      compactedTokensEst,
      reductionPct,
    },
  };
}

function applyFieldStripping(events: EscalationEvent[]): CompactEvent[] {
  return events.map(e => {
    const compact: CompactEvent = {
      t: e.timestamp.slice(11, 19), // just HH:MM:SS
      v: round3(e.valenceScore.composite),
      cx: round3(e.valenceScore.complexity),
      u: round3(e.valenceScore.urgency),
      s: round3(e.valenceScore.stakes),
      c: round3(e.tier1ConfidenceAtHandoff),
      to: e.escalatedToTier,
      tok: e.tokensBeforeEscalation,
    };
    return compact;
  });
}

function round3(n: number): number {
  return Math.round(n * 1000) / 1000;
}

function applyRepetitionCollapse(
  compacts: CompactEvent[],
  original: EscalationEvent[]
): CompactedEntry[] {
  if (compacts.length === 0) return [];

  const groups: CompactedEntry[] = [];
  let i = 0;

  while (i < compacts.length) {
    const current = compacts[i];
    const targetTier = current.to;
    const targetValence = current.v;

    // Find run length: 2+ consecutive events with same tier and similar valence
    let runEnd = i + 1;
    while (runEnd < compacts.length) {
      const next = compacts[runEnd];
      if (next.to === targetTier && Math.abs(next.v - targetValence) < 0.1) {
        runEnd++;
      } else {
        break;
      }
    }

    const runLength = runEnd - i;
    if (runLength >= 3) {
      const runEvents = compacts.slice(i, runEnd);
      const confs = runEvents.map(e => e.c);
      const vs = runEvents.map(e => e.v);
      const cxs = runEvents.map(e => e.cx);
      groups.push({
        rep: true,
        n: runLength,
        tier: targetTier,
        avgC: round3(confs.reduce((a, b) => a + b, 0) / confs.length),
        vRange: [round3(Math.min(...vs)), round3(Math.max(...vs))],
        cxRange: [round3(Math.min(...cxs)), round3(Math.max(...cxs))],
        from: original[i].timestamp.slice(11, 19),
        to: original[runEnd - 1].timestamp.slice(11, 19),
      });
    } else {
      for (let j = i; j < runEnd; j++) {
        groups.push(compacts[j]);
      }
    }
    i = runEnd;
  }

  return groups;
}

function buildSessionSummary(
  input: CompactionInput,
  events: EscalationEvent[]
): string {
  const totalEscalations = events.length;
  const totalQueries = (input.nonEscalatedQueryCount ?? 0) + totalEscalations;
  const escalationRate = totalQueries > 0 ? (totalEscalations / totalQueries) * 100 : 0;

  const tierCounts = { t1: 0, t2: 0, t3: 0 };
  for (const e of events) {
    if (e.escalatedToTier === 1) tierCounts.t1++;
    else if (e.escalatedToTier === 2) tierCounts.t2++;
    else if (e.escalatedToTier === 3) tierCounts.t3++;
  }

  const avgComposite = events.length > 0
    ? events.reduce((s, e) => s + e.valenceScore.composite, 0) / events.length
    : 0;
  const avgComplexity = events.length > 0
    ? events.reduce((s, e) => s + e.valenceScore.complexity, 0) / events.length
    : 0;
  const avgConfidence = events.length > 0
    ? events.reduce((s, e) => s + e.tier1ConfidenceAtHandoff, 0) / events.length
    : 0;

  // Compact one-line header
  return `Session ${input.sessionId.slice(0, 8)}: ${totalQueries}q (${input.nonEscalatedQueryCount ?? 0}T1 / ${totalEscalations}esc=${escalationRate.toFixed(0)}%) → T1=${tierCounts.t1} T2=${tierCounts.t2} T3=${tierCounts.t3} | avg v=${avgComposite.toFixed(2)} cx=${avgComplexity.toFixed(2)} conf=${avgConfidence.toFixed(2)}`;
}
