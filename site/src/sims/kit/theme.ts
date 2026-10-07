/**
 * Theme tokens for canvas/SVG sims.
 *
 * Canvas can't use CSS variables directly, so sims read the `--sim-*` tokens
 * (defined in styles/course.css, validated with the data-viz palette checker)
 * at draw time, and redraw when the theme changes.
 */

export interface SimColors {
  surface: string;
  /** Panel background behind charts/grids (prototype: "sunk"). */
  sunk: string;
  text: string;
  textMuted: string;
  grid: string;
  axis: string;
  curve: string;
  point: string;
  mark: string;
  markRing: string;
  tangent: string;
  good: string;
  bad: string;
  /** Loss landscapes: lighter = lower loss, in both themes. */
  heatLow: string;
  heatHigh: string;
  /** Diverging pair (positive / negative, class 1 / class 0). Narration names them "amber" and "teal". */
  amber: string;
  teal: string;
  /** Gradients / backward pass. */
  plum: string;
  /** Categorical series, fixed order (validated all-pairs for 3 slots). */
  series: [string, string, string];
  fontMono: string;
  fontSans: string;
}

export function readColors(el: Element = document.documentElement): SimColors {
  const cs = getComputedStyle(el);
  const v = (name: string) => cs.getPropertyValue(name).trim();
  return {
    surface: v('--sim-surface'),
    sunk: v('--sim-sunk'),
    text: v('--sim-text'),
    textMuted: v('--sim-text-muted'),
    grid: v('--sim-grid'),
    axis: v('--sim-axis'),
    curve: v('--sim-curve'),
    point: v('--sim-point'),
    mark: v('--sim-mark'),
    markRing: v('--sim-mark-ring'),
    tangent: v('--sim-tangent'),
    good: v('--sim-good'),
    bad: v('--sim-bad'),
    heatLow: v('--sim-heat-low'),
    heatHigh: v('--sim-heat-high'),
    amber: v('--sim-amber'),
    teal: v('--sim-teal'),
    plum: v('--sim-plum'),
    series: [v('--sim-series-1'), v('--sim-series-2'), v('--sim-series-3')],
    fontMono: v('--pfh-font-mono') || 'ui-monospace, monospace',
    fontSans: v('--pfh-font') || 'system-ui, sans-serif',
  };
}

/**
 * Call `cb` when the theme changes: Starlight's picker (`<html data-theme>`)
 * or the OS colour scheme. Returns an unsubscribe function for destroy().
 */
export function onThemeChange(cb: () => void): () => void {
  const mo = new MutationObserver(cb);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  mq.addEventListener('change', cb);
  return () => {
    mo.disconnect();
    mq.removeEventListener('change', cb);
  };
}

/** True when the reader asked their OS for less motion. Checked live, not cached. */
export function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Parse `#rgb` / `#rrggbb` into [r, g, b]. */
export function hexToRgb(hex: string): [number, number, number] {
  let h = hex.trim().replace('#', '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  if (!/^[0-9a-f]{6}$/i.test(h)) return [128, 128, 128];
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Linear mix of two hex colours, t in 0..1, as an `rgb()` string. */
export function mix(a: string, b: string, t: number): string {
  const A = hexToRgb(a);
  const B = hexToRgb(b);
  const c = A.map((x, i) => Math.round(x + (B[i] - x) * t));
  return `rgb(${c.join(',')})`;
}
