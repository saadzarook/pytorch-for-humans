/** Format a number for display: fixed decimals normally, scientific when huge/tiny, and honest about NaN/∞. */
export function fmt(v: number, digits = 3): string {
  if (Number.isNaN(v)) return 'NaN';
  if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(2).replace('-', '−');
  return v.toFixed(digits).replace('-', '−');
}

/** Two decimals, the prototype's `f2` (keeps a plain hyphen-minus to match its narration). */
export function f2(v: number): string {
  return (Math.round(v * 100) / 100).toFixed(2);
}

/**
 * Map a slider position (0..steps) onto a log scale between `min` and `max`.
 * Learning rates span orders of magnitude, so a log slider gives every
 * regime a fair share of the track.
 */
export function logSlider(min: number, max: number, steps = 1000) {
  const lo = Math.log10(min);
  const hi = Math.log10(max);
  return {
    steps,
    toValue: (pos: number) => 10 ** (lo + ((hi - lo) * pos) / steps),
    toPos: (value: number) => Math.round(((Math.log10(value) - lo) / (hi - lo)) * steps),
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

/** "1 point" / "3 points". */
export function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}
