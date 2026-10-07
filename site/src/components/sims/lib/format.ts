/** Format a number for display: fixed decimals normally, scientific when huge/tiny, and honest about NaN/∞. */
export function fmt(v: number, digits = 3): string {
  if (Number.isNaN(v)) return 'NaN';
  if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(2).replace('-', '−');
  return v.toFixed(digits).replace('-', '−');
}

/**
 * Map a 0..1000 slider position onto a log scale between `min` and `max`.
 * Learning rates span orders of magnitude (0.001 → 1), so a log slider gives
 * every regime a fair share of the track.
 */
export function logSlider(min: number, max: number) {
  const lo = Math.log10(min);
  const hi = Math.log10(max);
  return {
    toValue: (pos: number) => 10 ** (lo + ((hi - lo) * pos) / 1000),
    toPos: (value: number) => Math.round(((Math.log10(value) - lo) / (hi - lo)) * 1000),
  };
}

/** Round a learning rate to something readable (2 significant figures). */
export function niceLr(v: number): number {
  return Number(v.toPrecision(2));
}

/** Short axis label: 3 significant digits at most, so it fits a narrow margin. */
export function fmtAxis(v: number): string {
  if (!Number.isFinite(v)) return fmt(v);
  const a = Math.abs(v);
  if (a >= 1e4) return v.toExponential(1).replace('-', '−');
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}
