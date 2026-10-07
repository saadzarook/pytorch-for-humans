import { describe, expect, it } from 'vitest';
import { close } from '../kit/testing';
import golden from './golden.json';
import { computeRound } from './math';

describe('backprop vs torch autograd (golden.json)', () => {
  for (const c of golden.cases) {
    it(`w=${c.w}, b=${c.b}: loss, w.grad, b.grad and the update`, () => {
      const r = computeRound(c.w, c.b);
      expect(close(r.L, c.loss)).toBe(true);
      expect(close(r.gw, c.w_grad)).toBe(true);
      expect(close(r.g, c.b_grad)).toBe(true);
      expect(close(r.w2, c.w2)).toBe(true);
      expect(close(r.b2, c.b2)).toBe(true);
    });
  }

  it('the "Skip 10 rounds" sequence matches round by round', () => {
    let { w, b } = golden.rounds[0];
    for (const ref of golden.rounds) {
      const r = computeRound(w, b);
      expect(close(r.L, ref.loss)).toBe(true);
      ({ w2: w, b2: b } = r);
    }
  });
});
