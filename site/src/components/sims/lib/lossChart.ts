import { fitCanvas, onResize, scale, type Surface } from './canvas';
import { readColors, type SimColors } from './theme';
import { fmt, fmtAxis } from './format';

/**
 * A tiny "loss vs step" line chart (single series, so no legend: the
 * caption names it). Values that blow up are clamped to the top edge and
 * flagged, rather than squashing the interesting part of the curve.
 *
 * Hovering (or touching) shows a crosshair with the exact step and loss.
 */
export class LossChart {
  private values: number[] = [];
  private surface!: Surface;
  private hoverIndex: number | null = null;
  private readonly canvas: HTMLCanvasElement;
  private readonly aspect: number;

  constructor(canvas: HTMLCanvasElement, aspect = 3.2) {
    this.canvas = canvas;
    this.aspect = aspect;
    this.resize();
    onResize(canvas.parentElement ?? canvas, () => {
      this.resize();
      this.draw();
    });
    canvas.addEventListener('pointermove', (e) => this.onPointer(e));
    canvas.addEventListener('pointerdown', (e) => this.onPointer(e));
    canvas.addEventListener('pointerleave', () => {
      this.hoverIndex = null;
      this.draw();
    });
  }

  setValues(values: number[]): void {
    this.values = values;
    if (this.hoverIndex !== null && this.hoverIndex >= values.length) this.hoverIndex = null;
    this.draw();
  }

  resize(): void {
    this.surface = fitCanvas(this.canvas, this.aspect, 200);
  }

  private layout() {
    const { width, height } = this.surface;
    const pad = { l: 44, r: 12, t: 10, b: 24 };
    const n = Math.max(this.values.length - 1, 10);
    const finite = this.values.filter(Number.isFinite);
    const first = finite[0] ?? 1;
    // Clamp the y-range: anything beyond 1.5× the starting loss counts as "exploding".
    const cap = Math.max(first * 1.5, 1e-9);
    const yMax = Math.min(Math.max(...finite, 1e-9), cap);
    const x = scale(0, n, pad.l, width - pad.r);
    const y = scale(0, yMax * 1.05, height - pad.b, pad.t);
    return { pad, n, yMax, cap, x, y, width, height };
  }

  private onPointer(e: PointerEvent): void {
    if (this.values.length === 0) return;
    const rect = this.canvas.getBoundingClientRect();
    const { x } = this.layout();
    const i = Math.round(x.invert(e.clientX - rect.left));
    this.hoverIndex = Math.max(0, Math.min(this.values.length - 1, i));
    this.draw();
  }

  draw(colors: SimColors = readColors()): void {
    const { ctx } = this.surface;
    const { pad, n, yMax, cap, x, y, width, height } = this.layout();
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = colors.surface;
    ctx.fillRect(0, 0, width, height);

    // Recessive grid: baseline + top line.
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1;
    for (const v of [0, yMax / 2, yMax]) {
      ctx.beginPath();
      ctx.moveTo(pad.l, Math.round(y(v)) + 0.5);
      ctx.lineTo(width - pad.r, Math.round(y(v)) + 0.5);
      ctx.stroke();
    }

    ctx.fillStyle = colors.textMuted;
    ctx.font = `11px ${colors.fontMono}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(fmtAxis(yMax), pad.l - 6, y(yMax));
    ctx.fillText('0', pad.l - 6, y(0));
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('step →', pad.l, height - 6);
    ctx.textAlign = 'right';
    ctx.fillText(String(n), width - pad.r, height - 6);

    if (this.values.length === 0) return;

    // The loss line (2px), clamped at the cap.
    ctx.strokeStyle = colors.mark;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let exploded = false;
    this.values.forEach((v, i) => {
      const clamped = Number.isFinite(v) ? Math.min(v, cap) : cap;
      if (!Number.isFinite(v) || v > cap) exploded = true;
      const px = x(i);
      const py = y(Math.min(clamped, yMax * 1.05));
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // End marker.
    const last = this.values.length - 1;
    const lv = this.values[last];
    ctx.beginPath();
    ctx.arc(x(last), y(Number.isFinite(lv) ? Math.min(lv, yMax * 1.05) : yMax * 1.05), 4, 0, Math.PI * 2);
    ctx.fillStyle = colors.mark;
    ctx.fill();

    if (exploded) {
      ctx.fillStyle = colors.bad;
      ctx.font = `bold 12px ${colors.fontSans}`;
      ctx.textAlign = 'right';
      ctx.fillText('↑ loss went off the chart', width - pad.r, pad.t + 12);
    }

    // Hover crosshair + label.
    if (this.hoverIndex !== null) {
      const i = this.hoverIndex;
      const v = this.values[i];
      const px = x(i);
      ctx.strokeStyle = colors.axis;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(px, pad.t);
      ctx.lineTo(px, height - pad.b);
      ctx.stroke();
      ctx.setLineDash([]);
      const label = `step ${i} · loss ${fmt(v, 4)}`;
      ctx.font = `12px ${colors.fontMono}`;
      const w = ctx.measureText(label).width + 12;
      const lx = Math.min(Math.max(px - w / 2, pad.l), width - pad.r - w);
      ctx.fillStyle = colors.surface;
      ctx.strokeStyle = colors.grid;
      ctx.beginPath();
      ctx.roundRect(lx, pad.t, w, 20, 4);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = colors.text;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, lx + 6, pad.t + 10);
      ctx.textBaseline = 'alphabetic';
    }
  }
}
