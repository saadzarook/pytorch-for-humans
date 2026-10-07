/**
 * Fitting a line y = w·x + b by gradient descent on mean squared error, as
 * pure functions. Data and settings come from the lesson's data.json; the
 * lesson's facts.py replays the same runs, and the lesson's parity test checks
 * this file reproduces facts.json.
 */

export interface Dataset {
  x: number[];
  y: number[];
}

/** Mean squared error of the line y = w·x + b, its gradient, and the exact best fit. */
export function lineFit({ x, y }: Dataset) {
  const n = x.length;
  const mse = (w: number, b: number) => {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const e = w * x[i] + b - y[i];
      s += e * e;
    }
    return s / n;
  };
  const grad = (w: number, b: number): [number, number] => {
    let gw = 0;
    let gb = 0;
    for (let i = 0; i < n; i++) {
      const e = w * x[i] + b - y[i];
      gw += 2 * e * x[i];
      gb += 2 * e;
    }
    return [gw / n, gb / n];
  };
  // Closed-form least squares: w = cov(x, y) / var(x), b = mean(y) − w·mean(x).
  const mx = x.reduce((a, v) => a + v, 0) / n;
  const my = y.reduce((a, v) => a + v, 0) / n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) ** 2;
  }
  const bestW = sxy / sxx;
  return { mse, grad, best: { w: bestW, b: my - bestW * mx } };
}

export type Fit = ReturnType<typeof lineFit>;

/** Stopping rules shared with the lesson's facts.py. */
export const RULES = { explodeLoss: 1e8, moveTol: 1e-4 };

export type OutcomeKind = 'converged' | 'exploded' | 'gave up';

/** One step; returns the new point and, if the run is over, how it ended. */
export function stepFit(fit: Fit, w: number, b: number, lr: number, steps: number, maxSteps: number) {
  const [gw, gb] = fit.grad(w, b);
  const nw = w - lr * gw;
  const nb = b - lr * gb;
  const loss = fit.mse(nw, nb);
  let end: OutcomeKind | null = null;
  if (!Number.isFinite(loss) || loss > RULES.explodeLoss) end = 'exploded';
  else if (Math.hypot(nw - w, nb - b) < RULES.moveTol) end = 'converged';
  else if (steps >= maxSteps) end = 'gave up';
  return { w: nw, b: nb, end };
}

/** Run to the end exactly like the live sim does. */
export function simulate(fit: Fit, lr: number, start: { w: number; b: number }, maxSteps: number) {
  let { w, b } = start;
  for (let steps = 1; ; steps++) {
    const r = stepFit(fit, w, b, lr, steps, maxSteps);
    if (r.end) return { kind: r.end, steps, w: r.w, b: r.b };
    ({ w, b } = r);
  }
}
