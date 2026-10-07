/**
 * 1D gradient descent on a polynomial loss curve, as pure functions.
 *
 * Curves come from the lesson's data.json (coefficients, lowest power first),
 * so the lesson's facts.py can replay exactly the same runs in Python, and the
 * lesson's parity test checks this file reproduces facts.json.
 */

export interface Preset {
  id: string;
  label: string;
  lr: number;
  start: number;
  /** One-line hint shown after picking the preset. */
  hint: string;
}

export interface CurveSpec {
  id: string;
  name: string;
  /** Human-readable formula, e.g. "(w − 2)²". */
  formula: string;
  /** Polynomial coefficients, lowest power first: c0 + c1·w + c2·w² + … */
  coeffs: number[];
  domain: [number, number];
  range: [number, number];
  defaultStart: number;
  defaultLr: number;
  presets: Preset[];
}

export interface Curve extends CurveSpec {
  f: (w: number) => number;
  df: (w: number) => number;
  /** Lowest point on the domain (where "you found the bottom" applies). */
  globalMin: number;
}

/** Stopping rules shared with facts.py. */
export const RULES = { maxSteps: 300, moveTol: 5e-4, slopeTol: 5e-3, explodeAbs: 1e6, globalTol: 0.05 };

export function makeCurve(spec: CurveSpec): Curve {
  const c = spec.coeffs;
  const f = (w: number) => c.reduceRight((acc, k) => acc * w + k, 0);
  const d = c.slice(1).map((k, i) => k * (i + 1));
  const df = (w: number) => d.reduceRight((acc, k) => acc * w + k, 0);
  // Global minimum: dense scan, then Newton polish on f'.
  const [lo, hi] = spec.domain;
  let best = lo;
  for (let i = 0; i <= 4000; i++) {
    const w = lo + ((hi - lo) * i) / 4000;
    if (f(w) < f(best)) best = w;
  }
  const d2 = d.slice(1).map((k, i) => k * (i + 1));
  const ddf = (w: number) => d2.reduceRight((acc, k) => acc * w + k, 0);
  for (let i = 0; i < 20 && ddf(best) > 0; i++) best -= df(best) / ddf(best);
  return { ...spec, f, df, globalMin: best };
}

export type OutcomeKind = 'bottom' | 'local' | 'exploded' | 'gave up';
export interface Outcome {
  kind: OutcomeKind;
  steps: number;
  final: number;
}

/** One gradient-descent step. */
export const nextW = (curve: Curve, w: number, lr: number) => w - lr * curve.df(w);

/**
 * Classify the move from `prev` to `w` (after `steps` steps) with the sim's
 * stopping rules, or null if the run should keep going.
 */
export function classify(curve: Curve, prev: number, w: number, steps: number): Outcome | null {
  if (!Number.isFinite(curve.f(w)) || Math.abs(w) > RULES.explodeAbs) return { kind: 'exploded', steps, final: w };
  if (Math.abs(w - prev) < RULES.moveTol && Math.abs(curve.df(w)) < RULES.slopeTol)
    return { kind: Math.abs(w - curve.globalMin) < RULES.globalTol ? 'bottom' : 'local', steps, final: w };
  if (steps >= RULES.maxSteps) return { kind: 'gave up', steps, final: w };
  return null;
}

/** Run to the end exactly like the live sim does. */
export function simulate(curve: Curve, lr: number, start: number): Outcome {
  let w = start;
  for (let steps = 1; ; steps++) {
    const next = nextW(curve, w, lr);
    const out = classify(curve, w, next, steps);
    if (out) return out;
    w = next;
  }
}
