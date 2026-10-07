/**
 * Parity: the live sims on this lesson's page must behave exactly as facts.json
 * (computed in Python by facts.py) says, because the lesson's prose quotes those facts.
 */
import { describe, expect, it } from 'vitest';
import { makeCurve, RULES, simulate as simulate1d, type CurveSpec } from '../../../site/src/sims/ball/math';
import { lineFit, simulate as simulate2d } from '../../../site/src/sims/contour/math';
import data from './data.json';
import facts from './facts.json';

describe('ball sim reproduces facts.json', () => {
  it('uses the same stopping rules as facts.py (data.json ballRules)', () => {
    const { $comment, ...rules } = data.ballRules;
    void $comment;
    expect(rules).toEqual(RULES);
  });

  for (const [cid, spec] of Object.entries(data.curves)) {
    const curve = makeCurve(spec as CurveSpec);
    for (const p of spec.presets) {
      const expected = (facts.ball as Record<string, Record<string, { kind: string; steps: number }>>)[cid][p.id];
      it(`${cid} / ${p.label}: ${expected.kind} after ${expected.steps} steps`, () => {
        const out = simulate1d(curve, p.lr, p.start);
        expect({ kind: out.kind, steps: out.steps }).toEqual(expected);
      });
    }
  }

  it(`bowl explodes above lr = ${facts.bowl.chaosLr}, bounces forever at it`, () => {
    const bowl = makeCurve(data.curves.bowl as CurveSpec);
    const chaos = Number(facts.bowl.chaosLr);
    expect(simulate1d(bowl, chaos * 0.95, bowl.defaultStart).kind).toBe('bottom');
    expect(simulate1d(bowl, chaos, bowl.defaultStart).kind).toBe('gave up');
    expect(simulate1d(bowl, chaos * 1.05, bowl.defaultStart).kind).toBe('exploded');
  });
});

describe('contour sim reproduces facts.json', () => {
  const fit = lineFit(data.delivery);
  for (const p of data.contour.presets) {
    const outcome = (facts.presetOutcomes as Record<string, string>)[p.id];
    it(`${p.label} (lr ${p.lr}): ${outcome}`, () => {
      const out = simulate2d(fit, p.lr, data.contour.start, data.contour.maxSteps);
      expect(`${out.kind} after ${out.steps} steps`).toBe(outcome);
    });
  }
  it(`best fit ≈ ${facts.bestFit.b} min + ${facts.bestFit.w} min/km`, () => {
    expect(fit.best.w.toFixed(2)).toBe(facts.bestFit.w);
    expect(fit.best.b.toFixed(2)).toBe(facts.bestFit.b);
  });
});
