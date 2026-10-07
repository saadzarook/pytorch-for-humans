/**
 * The 1D loss curves used by the ball sim, plus a generic line-fit helper.
 *
 * Datasets are not defined here: each lesson keeps its data in its own
 * data.json and passes it to the sims as props.
 */

export interface Preset {
  label: string;
  lr: number;
  start: number;
  /** One-line hint shown after picking the preset. */
  hint: string;
}

export interface Curve1D {
  id: 'bowl' | 'bumpy';
  name: string;
  f: (w: number) => number;
  df: (w: number) => number;
  /** Visible w range. */
  domain: [number, number];
  /** Visible loss range. */
  range: [number, number];
  /** The lowest point (for "you found it / you got stuck" messages). */
  globalMin: number;
  defaultStart: number;
  defaultLr: number;
  presets: Preset[];
}

/** A simple bowl: loss = (w − 2)². Minimum at w = 2. Converges for 0 < lr < 1. */
export const bowl: Curve1D = {
  id: 'bowl',
  name: 'Smooth bowl',
  f: (w) => (w - 2) ** 2,
  df: (w) => 2 * (w - 2),
  domain: [-3, 7],
  range: [0, 26],
  globalMin: 2,
  defaultStart: -2,
  defaultLr: 0.3,
  presets: [
    {
      label: '🐢 Too slow',
      lr: 0.02,
      start: -2,
      hint: 'Baby steps. It will get there… eventually. Maybe go make chai.',
    },
    {
      label: '👌 Just right',
      lr: 0.3,
      start: -2,
      hint: 'Big steps while the slope is steep, smaller ones as it flattens out.',
    },
    {
      label: '🏓 Bouncy',
      lr: 0.9,
      start: -2,
      hint: 'Overshoots the bottom every time, bouncing side to side, but the bounces shrink.',
    },
    {
      label: '🔥 Chaos mode',
      lr: 1.05,
      start: -2,
      hint: 'Every hop overshoots by more than the last. Watch the loss chart.',
    },
  ],
};

/**
 * A bumpy curve with a shallow "trap" valley on the right (w ≈ 2.33) and a
 * deeper valley on the left (w ≈ −2.64).
 * loss = 0.06w⁴ − 0.75w² + 0.45w + 4
 */
export const bumpy: Curve1D = {
  id: 'bumpy',
  name: 'Bumpy hills',
  f: (w) => 0.06 * w ** 4 - 0.75 * w ** 2 + 0.45 * w + 4,
  df: (w) => 0.24 * w ** 3 - 1.5 * w + 0.45,
  domain: [-4.4, 4.4],
  range: [0, 13],
  globalMin: -2.6383,
  defaultStart: 3.5,
  defaultLr: 0.05,
  presets: [
    {
      label: '🕳️ Stuck in a dip',
      lr: 0.05,
      start: 3.5,
      hint: 'It settles happily in the nearest valley… which is not the deepest one.',
    },
    {
      label: '🦘 Lucky leap',
      lr: 0.4,
      start: 4,
      hint: 'A steep start plus a big step hops right over the hill.',
    },
    {
      label: '🎯 Good start',
      lr: 0.05,
      start: -0.5,
      hint: 'Same small steps, different starting point: now it finds the deep valley.',
    },
  ],
};

export const curves = { bowl, bumpy };

export interface Dataset {
  x: number[];
  y: number[];
}

/** Mean squared error of the line y = w·x + b on a dataset, its gradient, and the exact best fit. */
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
