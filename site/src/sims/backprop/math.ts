/**
 * Backprop "chain of blame": a one-neuron model p = w·x + b, loss L = (p − y)².
 * Pure functions, checked against torch autograd in backprop.test.ts (golden.json).
 */
import cfg from './config.json';

export const X = cfg.x;
export const Y = cfg.y;
export const LR = cfg.lr;
export const START = cfg.start;

export interface Round {
  /** forward: z = w·x, p = z + b, e = p − y, L = e² */
  z: number;
  p: number;
  e: number;
  L: number;
  /** backward: ∂L/∂e = ∂L/∂p = ∂L/∂z = ∂L/∂b = 2e,  ∂L/∂w = 2e·x */
  g: number;
  gw: number;
  /** update: w ← w − lr·∂L/∂w,  b ← b − lr·∂L/∂b */
  w2: number;
  b2: number;
}

export function computeRound(w: number, b: number, x = X, y = Y, lr = LR): Round {
  const z = w * x;
  const p = z + b;
  const e = p - y;
  const L = e * e;
  const g = 2 * e;
  const gw = g * x;
  return { z, p, e, L, g, gw, w2: w - lr * gw, b2: b - lr * g };
}
