/**
 * Small helpers for crisp, responsive <canvas> drawing.
 *
 * Sims draw in CSS pixels; `fitCanvas` sizes the backing store for the device
 * pixel ratio so lines stay sharp on phones and retina screens.
 */

export interface Surface {
  ctx: CanvasRenderingContext2D;
  /** Width in CSS pixels. */
  width: number;
  /** Height in CSS pixels. */
  height: number;
}

/** Resize the canvas backing store to its container's width × devicePixelRatio. Height = width / aspect. */
export function fitCanvas(canvas: HTMLCanvasElement, aspect: number, maxHeight = 420): Surface {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas not supported');
  const width = Math.max(200, canvas.parentElement?.clientWidth || canvas.clientWidth || 300);
  const height = Math.min(maxHeight, Math.round(width / aspect));
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

/** Call `cb` whenever `el` changes width (at most once per frame). Returns a disconnect function. */
export function onResize(el: Element, cb: () => void): () => void {
  let frame = 0;
  let lastWidth = -1;
  const ro = new ResizeObserver((entries) => {
    const w = Math.round(entries[0]?.contentRect.width ?? 0);
    if (w === lastWidth) return;
    lastWidth = w;
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(cb);
  });
  ro.observe(el);
  return () => {
    cancelAnimationFrame(frame);
    ro.disconnect();
  };
}

/** A linear map from a data interval to a pixel interval (with `.invert`). */
export function scale(d0: number, d1: number, r0: number, r1: number) {
  const k = (r1 - r0) / (d1 - d0);
  const fn = (v: number) => r0 + (v - d0) * k;
  fn.invert = (p: number) => d0 + (p - r0) / k;
  return fn;
}

/** Filled circle with a surface-coloured ring, so it stands out on any background. */
export function dot(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, fill: string, ring: string): void {
  ctx.beginPath();
  ctx.arc(x, y, r + 2, 0, Math.PI * 2);
  ctx.fillStyle = ring;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
}

/** Small filled triangle pointing in `dir` (radians). */
export function arrowHead(ctx: CanvasRenderingContext2D, x: number, y: number, dir: number, size: number, fill: string): void {
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

/** A line from (x1, y1) to (x2, y2) with an arrowhead at the end. */
export function arrow(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  color: string,
  width = 2,
): void {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  const a = Math.atan2(y2 - y1, x2 - x1);
  const h = 7;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - h * Math.cos(a - 0.45), y2 - h * Math.sin(a - 0.45));
  ctx.lineTo(x2 - h * Math.cos(a + 0.45), y2 - h * Math.sin(a + 0.45));
  ctx.closePath();
  ctx.fill();
}

export interface ChartFrameOptions {
  width: number;
  height: number;
  pad: { l: number; r: number; t: number; b: number };
  yTicks: number[];
  yFmt: (v: number) => string;
  yMap: (v: number) => number;
  xMax: number;
  xLabel?: string;
  colors: { background: string; grid: string; text: string; font: string };
}

/** Background, horizontal gridlines with y labels, and 0 / mid / max x labels. */
export function chartFrame(c: CanvasRenderingContext2D, o: ChartFrameOptions): void {
  const { width: W, height: H, pad, colors } = o;
  c.clearRect(0, 0, W, H);
  c.fillStyle = colors.background;
  c.fillRect(0, 0, W, H);
  c.font = `11px ${colors.font}`;
  c.textBaseline = 'middle';
  for (const v of o.yTicks) {
    const y = Math.round(o.yMap(v)) + 0.5;
    c.strokeStyle = colors.grid;
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(pad.l, y);
    c.lineTo(W - pad.r, y);
    c.stroke();
    c.fillStyle = colors.text;
    c.textAlign = 'right';
    c.fillText(o.yFmt(v), pad.l - 6, y);
  }
  c.textAlign = 'center';
  c.textBaseline = 'top';
  for (const v of [0, Math.round(o.xMax / 2), o.xMax]) {
    c.fillText(String(v), pad.l + (v / o.xMax) * (W - pad.l - pad.r), H - pad.b + 5);
  }
  if (o.xLabel) {
    c.textAlign = 'left';
    c.fillText(o.xLabel, pad.l, H - pad.b + 5);
  }
  c.textBaseline = 'alphabetic';
}

/** Text with a surface-coloured halo, readable on top of heatmaps and lines. */
export function label(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  color: string,
  halo: string,
  font: string,
  align: CanvasTextAlign = 'left',
): void {
  ctx.font = font;
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round';
  ctx.lineWidth = 4;
  ctx.strokeStyle = halo;
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
  ctx.textBaseline = 'alphabetic';
}
