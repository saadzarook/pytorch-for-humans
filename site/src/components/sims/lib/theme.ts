/**
 * Theme + motion preferences for canvas sims.
 *
 * Canvas can't use CSS variables directly, so we read the `--sim-*` tokens
 * (defined in styles/course.css) at draw time and redraw when the theme flips.
 */

export interface SimColors {
  surface: string;
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
  heatLow: string;
  heatHigh: string;
  fontMono: string;
  fontSans: string;
}

export function readColors(el: Element = document.documentElement): SimColors {
  const cs = getComputedStyle(el);
  const v = (name: string) => cs.getPropertyValue(name).trim();
  return {
    surface: v('--sim-surface'),
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
    fontMono: v('--pfh-font-mono') || 'ui-monospace, monospace',
    fontSans: v('--pfh-font') || 'system-ui, sans-serif',
  };
}

/** Run `cb` whenever Starlight's theme picker changes `<html data-theme>`. */
export function onThemeChange(cb: () => void): void {
  new MutationObserver(cb).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  });
}

const motionQuery =
  typeof window !== 'undefined' ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;

/** True when the reader asked their OS for less motion. Checked live, not cached. */
export function prefersReducedMotion(): boolean {
  return motionQuery?.matches ?? false;
}

/** Parse `#rrggbb` into [r, g, b]. */
export function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return [128, 128, 128];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
