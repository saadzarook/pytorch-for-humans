/**
 * The optimizer race, as pure functions (no DOM), so Vitest can check it
 * against real torch.optim (golden.json) and the narration's claims (facts.json).
 *
 * Loss landscape: a curved valley,
 *   f(x, y) = a·x² + b·(y − c·sin x)²
 * whose floor is the curve y = c·sin x and whose lowest point is (0, 0).
 */
import cfg from './config.json';

const { a, b, c } = cfg.valley;

export type OptName = 'SGD' | 'Momentum' | 'Adam';
export const OPT_NAMES: OptName[] = ['SGD', 'Momentum', 'Adam'];

export const floorY = (x: number) => c * Math.sin(x);
export const f = (x: number, y: number) => a * x * x + b * (y - floorY(x)) ** 2;
/** ∇f = [∂f/∂x, ∂f/∂y]. */
export function grad(x: number, y: number): [number, number] {
  const r = y - floorY(x);
  return [2 * a * x - 2 * b * c * r * Math.cos(x), 2 * b * r];
}

export interface OptState {
  name: OptName;
  x: number;
  y: number;
  /** Momentum buffer (torch: momentum_buffer). */
  vx: number;
  vy: number;
  /** Adam first / second moment estimates (torch: exp_avg / exp_avg_sq). */
  mx: number;
  my: number;
  s2x: number;
  s2y: number;
  /** Number of updates taken (torch: state['step']). */
  t: number;
}

export function createState(name: OptName, x: number, y: number): OptState {
  return { name, x, y, vx: 0, vy: 0, mx: 0, my: 0, s2x: 0, s2y: 0, t: 0 };
}

/**
 * One update, mirroring torch.optim:
 * - SGD:      p ← p − lr·g
 * - Momentum: v ← μ·v + g  (first step v = g);  p ← p − lr·v      [SGD(momentum=μ), dampening 0]
 * - Adam:     m ← β1·m + (1−β1)·g;  s ← β2·s + (1−β2)·g²;
 *             p ← p − lr·(m / (1−β1ᵗ)) / (√(s / (1−β2ᵗ)) + ε)     [default betas, eps]
 */
export function stepState(o: OptState, lr: number): void {
  const [gx, gy] = grad(o.x, o.y);
  o.t++;
  if (o.name === 'SGD') {
    o.x -= lr * gx;
    o.y -= lr * gy;
  } else if (o.name === 'Momentum') {
    const mu = cfg.momentum;
    o.vx = mu * o.vx + gx;
    o.vy = mu * o.vy + gy;
    o.x -= lr * o.vx;
    o.y -= lr * o.vy;
  } else {
    const [b1, b2] = cfg.betas;
    o.mx = b1 * o.mx + (1 - b1) * gx;
    o.my = b1 * o.my + (1 - b1) * gy;
    o.s2x = b2 * o.s2x + (1 - b2) * gx * gx;
    o.s2y = b2 * o.s2y + (1 - b2) * gy * gy;
    const c1 = 1 - b1 ** o.t;
    const c2 = 1 - b2 ** o.t;
    o.x -= (lr * (o.mx / c1)) / (Math.sqrt(o.s2x / c2) + cfg.eps);
    o.y -= (lr * (o.my / c1)) / (Math.sqrt(o.s2y / c2) + cfg.eps);
  }
}

export const isDiverged = (o: { x: number; y: number }) =>
  !Number.isFinite(o.x) || !Number.isFinite(o.y) || Math.abs(o.x) > cfg.divergeAbs || Math.abs(o.y) > cfg.divergeAbs;

/** The raw path of one optimizer for `steps` updates (no stopping), for golden tests. */
export function path(name: OptName, lr: number, steps: number, start = cfg.start): [number, number][] {
  const o = createState(name, start[0], start[1]);
  const out: [number, number][] = [[o.x, o.y]];
  for (let i = 0; i < steps; i++) {
    stepState(o, lr);
    out.push([o.x, o.y]);
  }
  return out;
}

export interface RaceOutcome {
  /** Step at which loss first dropped below cfg.arriveLoss, or null. */
  arrived: number | null;
  /** Step at which it diverged, or null. */
  diverged: number | null;
}

/** Run the whole race exactly as the live sim does (same stopping rules). */
export function race(lr: number, start = cfg.start): Record<OptName, RaceOutcome> {
  const out = {} as Record<OptName, RaceOutcome>;
  for (const name of OPT_NAMES) {
    const o = createState(name, start[0], start[1]);
    let arrived: number | null = null;
    let diverged: number | null = null;
    for (let step = 1; step <= cfg.maxSteps; step++) {
      stepState(o, lr);
      if (isDiverged(o)) {
        diverged = step;
        break;
      }
      if (arrived === null && f(o.x, o.y) < cfg.arriveLoss) arrived = step;
    }
    out[name] = { arrived, diverged };
  }
  return out;
}

/** Slider position (0..sliderSteps) → learning rate, log scale over lrRange. */
export function sliderLr(pos: number): number {
  const [lo, hi] = cfg.lrRange.map(Math.log10);
  return 10 ** (lo + ((hi - lo) * pos) / cfg.sliderSteps);
}
