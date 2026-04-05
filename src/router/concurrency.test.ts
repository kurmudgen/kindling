import { describe, it, expect, beforeEach } from 'vitest';
import { ConcurrencyGuard } from './concurrency.js';

describe('ConcurrencyGuard', () => {
  let guard: ConcurrencyGuard;

  beforeEach(() => {
    guard = new ConcurrencyGuard();
  });

  it('assigns monotonically increasing generation numbers', () => {
    const a = guard.acquire();
    const b = guard.acquire();
    const c = guard.acquire();
    expect(a).toBe(1);
    expect(b).toBe(2);
    expect(c).toBe(3);
  });

  it('tracks active generations', () => {
    const a = guard.acquire();
    const b = guard.acquire();
    expect(guard.isActive(a)).toBe(true);
    expect(guard.isActive(b)).toBe(true);
    expect(guard.inFlightCount()).toBe(2);
  });

  it('releases generations correctly', () => {
    const a = guard.acquire();
    const b = guard.acquire();
    guard.release(a);
    expect(guard.isActive(a)).toBe(false);
    expect(guard.isActive(b)).toBe(true);
    expect(guard.inFlightCount()).toBe(1);
  });

  it('release is idempotent', () => {
    const a = guard.acquire();
    guard.release(a);
    guard.release(a);
    expect(guard.isActive(a)).toBe(false);
    expect(guard.inFlightCount()).toBe(0);
  });

  it('handles many concurrent generations', () => {
    const gens: number[] = [];
    for (let i = 0; i < 100; i++) {
      gens.push(guard.acquire());
    }
    expect(guard.inFlightCount()).toBe(100);
    for (let i = 0; i < 100; i++) {
      expect(guard.isActive(gens[i])).toBe(true);
    }
    // Release half
    for (let i = 0; i < 50; i++) {
      guard.release(gens[i]);
    }
    expect(guard.inFlightCount()).toBe(50);
    // Latest should still be 100
    expect(guard.latest()).toBe(100);
  });

  it('detects stale generations correctly', () => {
    const a = guard.acquire();
    guard.release(a);
    const b = guard.acquire();
    expect(guard.isActive(a)).toBe(false); // stale
    expect(guard.isActive(b)).toBe(true);
    expect(b).toBe(2); // monotonic
  });

  it('reset clears all state', () => {
    guard.acquire();
    guard.acquire();
    guard.reset();
    expect(guard.inFlightCount()).toBe(0);
    expect(guard.latest()).toBe(0);
    expect(guard.acquire()).toBe(1);
  });
});
