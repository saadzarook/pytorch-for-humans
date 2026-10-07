import { describe, expect, it } from 'vitest';
import { close, firstMismatch } from '../kit/testing';
import golden from './golden.json';
import { adamStep, computeGradients, netFromWeights, predict, type Activation, type Point } from './math';

/** Mean BCE without the display clamp, to compare with BCEWithLogitsLoss. */
function bce(probs: number[], points: Point[]): number {
  return probs.reduce((s, p, i) => s - (points[i][2] * Math.log(p) + (1 - points[i][2]) * Math.log(1 - p)), 0) / probs.length;
}

const layerArrays = (net: ReturnType<typeof netFromWeights>, key: 'W' | 'gW' | 'b' | 'gb') =>
  net.layers.map((L) =>
    key === 'W' || key === 'gW'
      ? Array.from({ length: L.n }, (_, j) => Array.from(L[key].subarray(j * L.m, (j + 1) * L.m)))
      : Array.from(L[key]),
  );

describe('neural network vs nn.Sequential + BCEWithLogitsLoss + Adam (golden.json)', () => {
  for (const c of golden.cases) {
    const label = `${c.act}, hidden ${c.hidden}`;
    const points = c.points as Point[];

    it(`${label}: forward probabilities and loss`, () => {
      const net = netFromWeights(c.weights, c.act as Activation);
      const probs = points.map(([x, y]) => predict(net, x, y));
      expect(firstMismatch(probs, c.probs)).toBeNull();
      expect(close(bce(probs, points), c.loss)).toBe(true);
    });

    it(`${label}: gradients of every weight and bias`, () => {
      const net = netFromWeights(c.weights, c.act as Activation);
      computeGradients(net, points);
      expect(firstMismatch(layerArrays(net, 'gW'), c.grads.map((g) => g.W))).toBeNull();
      expect(firstMismatch(layerArrays(net, 'gb'), c.grads.map((g) => g.b))).toBeNull();
    });

    it(`${label}: one Adam step`, () => {
      const net = netFromWeights(c.weights, c.act as Activation);
      computeGradients(net, points);
      adamStep(net, c.lr);
      expect(firstMismatch(layerArrays(net, 'W'), c.afterAdam.map((g) => g.W))).toBeNull();
      expect(firstMismatch(layerArrays(net, 'b'), c.afterAdam.map((g) => g.b))).toBeNull();
    });
  }
});
