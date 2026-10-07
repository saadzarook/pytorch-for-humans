/**
 * A real 2 → H → H → 1 network: tanh/ReLU hidden layers, sigmoid output,
 * binary cross-entropy, full-batch Adam. Equivalent to nn.Sequential +
 * BCEWithLogitsLoss + torch.optim.Adam; checked in neural-network.test.ts.
 *
 * Weight layout matches torch.nn.Linear: W[j·m + k] = weight[j][k] (out × in).
 */
import { gauss, rng } from '../kit/random';
import cfg from './config.json';

export type Activation = 'tanh' | 'relu';
export type DatasetName = 'circle' | 'xor' | 'spiral';
export type Point = [x: number, y: number, label: number];

export interface Layer {
  n: number;
  m: number;
  W: Float64Array;
  b: Float64Array;
  gW: Float64Array;
  gb: Float64Array;
  mW: Float64Array;
  vW: Float64Array;
  mb: Float64Array;
  vb: Float64Array;
}

export interface Net {
  layers: Layer[];
  act: Activation;
  /** Adam step count. */
  t: number;
}

export const N_POINTS: number = cfg.nPoints;

export function genData(kind: DatasetName, seed: number = cfg.dataSeed): Point[] {
  const r = rng(seed);
  const pts: Point[] = [];
  const N = N_POINTS;
  if (kind === 'circle') {
    for (let i = 0; i < N; i++) {
      const inner = i < N / 2;
      const rad = inner ? r() * 0.45 : 0.65 + r() * 0.3;
      const a = r() * 2 * Math.PI;
      pts.push([rad * Math.cos(a), rad * Math.sin(a), inner ? 1 : 0]);
    }
  } else if (kind === 'xor') {
    for (let i = 0; i < N; i++) {
      let x = r() * 2 - 1;
      let y = r() * 2 - 1;
      x += x > 0 ? 0.08 : -0.08;
      y += y > 0 ? 0.08 : -0.08;
      pts.push([x * 0.9, y * 0.9, x * y > 0 ? 1 : 0]);
    }
  } else {
    for (let i = 0; i < N / 2; i++) {
      const s = i / (N / 2);
      const rad = 0.1 + 0.85 * s;
      const ang = s * 2 * Math.PI * 1.1;
      for (const c of [0, 1]) {
        const a = ang + c * Math.PI;
        pts.push([rad * Math.sin(a) + (r() - 0.5) * 0.08, rad * Math.cos(a) + (r() - 0.5) * 0.08, c]);
      }
    }
  }
  return pts;
}

function emptyLayer(n: number, m: number): Layer {
  const z = () => new Float64Array(n * m);
  const zb = () => new Float64Array(n);
  return { n, m, W: z(), b: zb(), gW: z(), gb: zb(), mW: z(), vW: z(), mb: zb(), vb: zb() };
}

/** Random init (He-style for ReLU hidden layers, 1/√fan-in otherwise), zero biases, seeded. */
export function initNet(hidden: number, act: Activation, seed: number = cfg.initSeed): Net {
  const r = rng(seed);
  const sizes = [2, hidden, hidden, 1];
  const layers: Layer[] = [];
  for (let i = 1; i < sizes.length; i++) {
    const L = emptyLayer(sizes[i], sizes[i - 1]);
    const scale = Math.sqrt((act === 'relu' && i < sizes.length - 1 ? 2 : 1) / L.m);
    for (let k = 0; k < L.n * L.m; k++) L.W[k] = gauss(r) * scale;
    layers.push(L);
  }
  return { layers, act, t: 0 };
}

/** Build a net from explicit weights (torch layout), e.g. from golden.json. */
export function netFromWeights(weights: { W: number[][]; b: number[] }[], act: Activation): Net {
  const layers = weights.map(({ W, b }) => {
    const L = emptyLayer(W.length, W[0].length);
    W.forEach((row, j) => row.forEach((v, k) => (L.W[j * L.m + k] = v)));
    b.forEach((v, j) => (L.b[j] = v));
    return L;
  });
  return { layers, act, t: 0 };
}

const activate = (act: Activation, z: number) => (act === 'tanh' ? Math.tanh(z) : z > 0 ? z : 0);
/** Derivative, written in terms of the activation's OUTPUT a. */
const dActivate = (act: Activation, a: number) => (act === 'tanh' ? 1 - a * a : a > 0 ? 1 : 0);

/** Activations of every layer; the last is [sigmoid(logit)]. */
export function forward(net: Net, x: number, y: number): number[][] {
  let a = [x, y];
  const as = [a];
  net.layers.forEach((L, li) => {
    const last = li === net.layers.length - 1;
    const o = new Array<number>(L.n);
    for (let j = 0; j < L.n; j++) {
      let s = L.b[j];
      for (let k = 0; k < L.m; k++) s += L.W[j * L.m + k] * a[k];
      o[j] = last ? 1 / (1 + Math.exp(-s)) : activate(net.act, s);
    }
    a = o;
    as.push(a);
  });
  return as;
}

export const predict = (net: Net, x: number, y: number) => forward(net, x, y)[net.layers.length][0];

/** Mean BCE loss, accuracy and the misclassified points. */
export function evaluate(net: Net, data: Point[]): { loss: number; accuracy: number; wrong: Point[] } {
  let loss = 0;
  let correct = 0;
  const wrong: Point[] = [];
  for (const pt of data) {
    const p = predict(net, pt[0], pt[1]);
    const pc = Math.min(Math.max(p, 1e-7), 1 - 1e-7);
    loss += -(pt[2] * Math.log(pc) + (1 - pt[2]) * Math.log(1 - pc));
    if (p > 0.5 === (pt[2] === 1)) correct++;
    else wrong.push(pt);
  }
  return { loss: loss / data.length, accuracy: correct / data.length, wrong };
}

/** Fill gW / gb with the MEAN gradient of the BCE loss over `data` (backprop). */
export function computeGradients(net: Net, data: Point[]): void {
  for (const L of net.layers) {
    L.gW.fill(0);
    L.gb.fill(0);
  }
  for (const [x, y, c] of data) {
    const as = forward(net, x, y);
    const p = as[as.length - 1][0];
    let delta = [p - c]; // d(BCE)/d(logit) for a sigmoid output
    for (let li = net.layers.length - 1; li >= 0; li--) {
      const L = net.layers[li];
      const aPrev = as[li];
      const nd = li > 0 ? new Array<number>(L.m).fill(0) : null;
      for (let j = 0; j < L.n; j++) {
        const d = delta[j];
        L.gb[j] += d;
        for (let k = 0; k < L.m; k++) {
          L.gW[j * L.m + k] += d * aPrev[k];
          if (nd) nd[k] += L.W[j * L.m + k] * d;
        }
      }
      if (nd) {
        for (let k = 0; k < L.m; k++) nd[k] *= dActivate(net.act, aPrev[k]);
        delta = nd;
      }
    }
  }
  const N = data.length;
  for (const L of net.layers) {
    for (let i = 0; i < L.gW.length; i++) L.gW[i] /= N;
    for (let i = 0; i < L.gb.length; i++) L.gb[i] /= N;
  }
}

/** One Adam update (default betas, eps) using the stored mean gradients. */
export function adamStep(net: Net, lr: number): void {
  const b1 = 0.9;
  const b2 = 0.999;
  net.t++;
  const c1 = 1 - b1 ** net.t;
  const c2 = 1 - b2 ** net.t;
  const update = (p: Float64Array, g: Float64Array, m: Float64Array, v: Float64Array) => {
    for (let i = 0; i < p.length; i++) {
      m[i] = b1 * m[i] + (1 - b1) * g[i];
      v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
      p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
    }
  };
  for (const L of net.layers) {
    update(L.W, L.gW, L.mW, L.vW);
    update(L.b, L.gb, L.mb, L.vb);
  }
}

/** One epoch: full-batch gradients, then one Adam step. */
export function trainEpoch(net: Net, data: Point[], lr: number): void {
  computeGradients(net, data);
  adamStep(net, lr);
}
