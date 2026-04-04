import { describe, it, expect, beforeEach } from 'vitest';
import { SpeculativeBuffer } from './speculative.js';

// Mock config so tests don't need config files
import * as config from '../config/config.js';
import { vi } from 'vitest';

vi.mock('../config/config.js', () => ({
  getConfig: () => ({
    buffer: { size: 4 },
    escalation: {},
  }),
  loadConfig: () => ({}),
  getOllamaHost: () => 'http://localhost:11434',
}));

describe('SpeculativeBuffer', () => {
  let buffer: SpeculativeBuffer;

  beforeEach(() => {
    buffer = new SpeculativeBuffer(4);
  });

  describe('Normal flush — Tier 1 confident, buffer commits', () => {
    it('should accept tokens up to capacity', () => {
      expect(buffer.push('hello')).toBe(true);
      expect(buffer.push('world')).toBe(true);
      expect(buffer.push('how')).toBe(true);
      expect(buffer.push('are')).toBe(true);
      expect(buffer.isFull()).toBe(true);
    });

    it('should reject push when full', () => {
      buffer.push('a');
      buffer.push('b');
      buffer.push('c');
      buffer.push('d');
      expect(buffer.push('e')).toBe(false);
    });

    it('should flush confirmed tokens to output', () => {
      buffer.push('hello');
      buffer.push('world');
      buffer.push('how');
      buffer.push('are');

      const confirmed = buffer.confirm();
      expect(confirmed).toEqual(['hello', 'world', 'how', 'are']);
      expect(buffer.getFlushedTokens()).toEqual(['hello', 'world', 'how', 'are']);
      expect(buffer.getOutput()).toBe('hello world how are');
    });

    it('should allow multiple fill-confirm cycles', () => {
      buffer.push('a');
      buffer.push('b');
      buffer.push('c');
      buffer.push('d');
      buffer.confirm();

      buffer.push('e');
      buffer.push('f');
      buffer.push('g');
      buffer.push('h');
      buffer.confirm();

      expect(buffer.getOutput()).toBe('a b c d e f g h');
    });
  });

  describe('Escalation handoff — Tier 2 takes over mid-buffer', () => {
    it('should reject buffer and provide resume point', () => {
      // Simulate: first batch confirmed, second batch rejected
      buffer.push('good');
      buffer.push('tokens');
      buffer.push('here');
      buffer.push('ok');
      buffer.confirm();

      buffer.push('bad');
      buffer.push('tokens');
      const result = buffer.reject();

      expect(result.rejectedTokens).toEqual(['bad', 'tokens']);
      expect(result.resumeFromToken).toBe(4); // resume after 4 confirmed tokens
    });

    it('should allow Tier 2 to override output after rejection', () => {
      buffer.push('a');
      buffer.push('b');
      buffer.push('c');
      buffer.push('d');
      buffer.confirm();

      buffer.push('wrong1');
      buffer.push('wrong2');
      buffer.reject();

      // Tier 2 takes over and appends its own tokens
      buffer.appendFlushed(['better', 'output', 'here']);
      expect(buffer.getOutput()).toBe('a b c d better output here');
    });

    it('should allow complete override of flushed tokens', () => {
      buffer.push('x');
      buffer.push('y');
      buffer.confirm();

      buffer.overrideFlushed(['completely', 'new', 'output']);
      expect(buffer.getOutput()).toBe('completely new output');
    });
  });

  describe('De-escalation — Tier 2 hands back to Tier 1', () => {
    it('should reset cleanly for new tier 1 generation', () => {
      buffer.push('tier2');
      buffer.push('output');
      buffer.confirm();

      // Tier 1 resumes
      buffer.push('tier1');
      buffer.push('back');
      buffer.push('in');
      buffer.push('control');
      buffer.confirm();

      expect(buffer.getOutput()).toBe('tier2 output tier1 back in control');
    });

    it('should track state transitions correctly', () => {
      expect(buffer.snapshot().state).toBe('idle');

      buffer.push('a');
      expect(buffer.snapshot().state).toBe('filling');

      buffer.confirm();
      expect(buffer.snapshot().state).toBe('flushed');

      buffer.push('b');
      buffer.reject();
      expect(buffer.snapshot().state).toBe('rejected');

      buffer.reset();
      expect(buffer.snapshot().state).toBe('idle');
      expect(buffer.getOutput()).toBe('');
    });
  });

  describe('Edge cases', () => {
    it('should handle empty buffer confirm', () => {
      const confirmed = buffer.confirm();
      expect(confirmed).toEqual([]);
    });

    it('should handle empty buffer reject', () => {
      const result = buffer.reject();
      expect(result.rejectedTokens).toEqual([]);
      expect(result.resumeFromToken).toBe(0);
    });

    it('should expose snapshot for logging', () => {
      buffer.push('token1');
      buffer.push('token2');
      const snap = buffer.snapshot();
      expect(snap.tokens).toEqual(['token1', 'token2']);
      expect(snap.fillPosition).toBe(2);
      expect(snap.capacity).toBe(4);
    });
  });
});
