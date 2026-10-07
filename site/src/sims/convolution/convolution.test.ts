import { describe, expect, it } from 'vitest';
import { firstMismatch } from '../kit/testing';
import facts from './facts.json';
import golden from './golden.json';
import { conv2d, IMG, O, PRESETS } from './math';

describe('convolution vs F.conv2d (golden.json)', () => {
  for (const [name, K] of Object.entries(PRESETS)) {
    it(`${name} kernel on the 12×12 "A"`, () => {
      const ref = (golden as Record<string, number[]>)[name];
      expect(firstMismatch(conv2d(IMG, K), ref)).toBeNull();
    });
  }
  it(`output is ${facts.outSize}×${facts.outSize} (${facts.windows} windows)`, () => {
    expect(O).toBe(facts.outSize);
    expect(conv2d(IMG, PRESETS.identity)).toHaveLength(facts.windows);
  });
});
