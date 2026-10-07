import { describe, expect, it } from 'vitest';
import { close, firstMismatch } from '../kit/testing';
import cfg from './config.json';
import facts from './facts.json';
import golden from './golden.json';
import { createState, OPT_NAMES, path, race, sliderLr, stepState, type OptName, type OptState } from './math';

interface TorchState {
  p: number[];
  v?: number[];
  m?: number[];
  s?: number[];
  t?: number;
}

/** Rebuild the JS optimizer state from torch.optim's state dict. */
function fromTorch(name: OptName, st: TorchState): OptState {
  const o = createState(name, st.p[0], st.p[1]);
  if (st.v) [o.vx, o.vy] = st.v;
  if (st.m && st.s) {
    [o.mx, o.my] = st.m;
    [o.s2x, o.s2y] = st.s;
    o.t = st.t ?? 0;
  }
  return o;
}

const runs = golden.runs as Record<string, Record<OptName, TorchState[]>>;
const outcomes = golden.outcomes as Record<string, Record<OptName, { arrived: number | null; diverged: number | null }>>;

describe('optimizer update rules vs real torch.optim, one step at a time (golden.json)', () => {
  for (const [lr, byName] of Object.entries(runs)) {
    for (const name of OPT_NAMES) {
      it(`${name}, lr ${lr}: each of ${cfg.golden.steps} updates from PyTorch's exact state`, () => {
        const states = byName[name];
        for (let i = 0; i + 1 < states.length; i++) {
          const o = fromTorch(name, states[i]);
          stepState(o, Number(lr));
          const m = firstMismatch([o.x, o.y], states[i + 1].p, `update ${i + 1}`);
          expect(m).toBeNull();
        }
      });
    }
  }
});

describe('whole paths vs real torch.optim (golden.json)', () => {
  for (const [lr, byName] of Object.entries(runs)) {
    for (const name of OPT_NAMES) {
      const diverges = outcomes[lr][name].diverged !== null;
      it(`${name}, lr ${lr}: ${diverges ? 'diverges at the same step' : 'same path, all steps'}`, () => {
        const ref = byName[name].map((s) => s.p);
        if (!diverges) {
          expect(firstMismatch(path(name, Number(lr), cfg.golden.steps), ref)).toBeNull();
        } else {
          // A diverging run amplifies last-bit sin/cos differences ~10× per step, so
          // compare what the learner sees: when it blows up, and the path before chaos.
          expect(race(Number(lr))[name]).toEqual(outcomes[lr][name]);
          const n = outcomes[lr][name].diverged! - 5;
          expect(firstMismatch(path(name, Number(lr), n), ref.slice(0, n + 1))).toBeNull();
        }
      });
    }
  }
});

describe('the live race reproduces the narration facts (facts.json)', () => {
  it(`arrival steps at lr ${facts.defaultLr}: ${facts.orderText}`, () => {
    const r = race(cfg.defaultLr);
    for (const name of OPT_NAMES) expect(r[name].arrived).toBe(facts.arrival[name]);
    const order = [...OPT_NAMES].sort((a, b) => r[a].arrived! - r[b].arrived!);
    expect(order).toEqual(facts.order);
  });

  it(`at lr ${facts.topLr}, SGD diverges and Momentum doesn't`, () => {
    const r = race(cfg.lrRange[1]);
    for (const name of OPT_NAMES) expect(r[name]).toEqual(facts.top[name]);
  });

  it(`SGD first diverges at slider position ${facts.sgdDivergesFromPos} (lr ≈ ${facts.sgdDivergesFromText})`, () => {
    let first = -1;
    for (let pos = 0; pos <= cfg.sliderSteps && first < 0; pos++) if (race(sliderLr(pos)).SGD.diverged) first = pos;
    expect(first).toBe(facts.sgdDivergesFromPos);
    expect(close(sliderLr(first), facts.sgdDivergesFromLr, 1e-5)).toBe(true);
  });
});
