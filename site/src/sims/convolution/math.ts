/**
 * Convolution (really cross-correlation, like F.conv2d): a 3×3 kernel slides
 * over a 12×12 image, stride 1, no padding → a 10×10 output.
 * Checked against F.conv2d in convolution.test.ts (golden.json).
 */
import cfg from './config.json';

export const IMG_ROWS: string[] = cfg.imageRows;
export const S = 12;
export const O = S - 3 + 1;
export const IMG: number[][] = IMG_ROWS.map((r) => [...r].map((ch) => (ch === '#' ? 1 : 0)));

export const PRESETS: Record<string, number[]> = cfg.presets;

/** Row-major O×O output: out[i·O + j] = Σ img[i+u][j+v] · K[u·3+v]. No kernel flip. */
export function conv2d(img: number[][], K: readonly number[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < O; i++)
    for (let j = 0; j < O; j++) {
      let s = 0;
      for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) s += img[i + u][j + v] * K[u * 3 + v];
      out.push(s);
    }
  return out;
}
