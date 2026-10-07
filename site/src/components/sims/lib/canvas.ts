/**
 * Small helpers for crisp, responsive <canvas> drawing.
 *
 * Every sim draws in CSS pixels; `fitCanvas` sizes the backing store for the
 * device pixel ratio so lines stay sharp on phones and retina screens.
 */

export interface Surface {
  ctx: CanvasRenderingContext2D;
  /** Width in CSS pixels. */
  width: number;
  /** Height in CSS pixels. */
  height: number;
}

/** Resize the canvas backing store to its current CSS size × devicePixelRatio. */
export function fitCanvas(canvas: HTMLCanvasElement, aspect: number, maxHeight = 420): Surface {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas not supported');
  const width = Math.max(200, canvas.parentElement?.clientWidth ?? canvas.clientWidth ?? 300);
  const height = Math.min(maxHeight, Math.round(width / aspect));
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

/** Call `cb` whenever `el` changes width (debounced to one call per frame). */
export function onResize(el: Element, cb: () => void): void {
  let frame = 0;
  let lastWidth = -1;
  new ResizeObserver((entries) => {
    const w = Math.round(entries[0]?.contentRect.width ?? 0);
    if (w === lastWidth) return;
    lastWidth = w;
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(cb);
  }).observe(el);
}

/** A linear map from a data interval to a pixel interval. */
export function scale(d0: number, d1: number, r0: number, r1: number) {
  const k = (r1 - r0) / (d1 - d0);
  const fn = (v: number) => r0 + (v - d0) * k;
  fn.invert = (p: number) => d0 + (p - r0) / k;
  return fn;
}

/** Draw a filled circle with a surface-coloured ring so it stands out on any background. */
export function dot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  fill: string,
  ring: string,
): void {
  ctx.beginPath();
  ctx.arc(x, y, r + 2, 0, Math.PI * 2);
  ctx.fillStyle = ring;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
}

/** Small filled triangle pointing in `dir` (radians), used for off-screen markers. */
export function arrowHead(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  dir: number,
  size: number,
  fill: string,
): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(dir);
  ctx.beginPath();
  ctx.moveTo(size, 0);
  ctx.lineTo(-size * 0.6, size * 0.7);
  ctx.lineTo(-size * 0.6, -size * 0.7);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.restore();
}
