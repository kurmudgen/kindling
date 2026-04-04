import { getConfig } from '../config/config.js';

export type BufferState = 'filling' | 'verifying' | 'flushed' | 'rejected' | 'idle';

export interface BufferSnapshot {
  tokens: string[];
  state: BufferState;
  fillPosition: number;
  capacity: number;
}

export class SpeculativeBuffer {
  private tokens: string[] = [];
  private state: BufferState = 'idle';
  private capacity: number;
  private flushedTokens: string[] = [];

  constructor(capacity?: number) {
    this.capacity = capacity ?? getConfig().buffer.size;
  }

  /** Tier 1 writes a token into the buffer */
  push(token: string): boolean {
    if (this.tokens.length >= this.capacity) {
      return false; // buffer full, must verify or flush
    }
    this.tokens.push(token);
    this.state = 'filling';
    return true;
  }

  /** Check if the buffer is full and ready for verification */
  isFull(): boolean {
    return this.tokens.length >= this.capacity;
  }

  /** Tier 2 confirms the buffer contents — flush to output */
  confirm(): string[] {
    const confirmed = [...this.tokens];
    this.flushedTokens.push(...confirmed);
    this.tokens = [];
    this.state = 'flushed';
    return confirmed;
  }

  /** Tier 2 rejects the buffer — returns the position to resume from */
  reject(): { rejectedTokens: string[]; resumeFromToken: number } {
    const rejected = [...this.tokens];
    const resumePoint = this.flushedTokens.length;
    this.tokens = [];
    this.state = 'rejected';
    return { rejectedTokens: rejected, resumeFromToken: resumePoint };
  }

  /** Get all tokens that have been flushed (committed) so far */
  getFlushedTokens(): string[] {
    return [...this.flushedTokens];
  }

  /** Replace flushed tokens when Tier 2 takes over after rejection */
  overrideFlushed(tokens: string[]): void {
    this.flushedTokens = [...tokens];
  }

  /** Append tokens directly to flushed (when a higher tier produces output) */
  appendFlushed(tokens: string[]): void {
    this.flushedTokens.push(...tokens);
  }

  /** Get current buffer state for logging */
  snapshot(): BufferSnapshot {
    return {
      tokens: [...this.tokens],
      state: this.state,
      fillPosition: this.tokens.length,
      capacity: this.capacity,
    };
  }

  /** Get the full assembled output */
  getOutput(): string {
    return this.flushedTokens.join(' ');
  }

  /** Reset the buffer for a new query */
  reset(): void {
    this.tokens = [];
    this.flushedTokens = [];
    this.state = 'idle';
  }
}
