/**
 * Fitting a line y = w·x + b with gradient descent: the data and current line
 * on the left, the loss landscape over (w, b) from above on the right.
 *
 * Course sim standard (docs/SIM_STANDARD.md): narrated phases, Step + speed,
 * labelled canvases (best fit, start, downhill arrow), the numbers on show,
 * and a log of key moments (loss going up, zig-zag, convergence, explosion).
 */
import { lineFit, stepFit, type Dataset, type OutcomeKind } from './math';
import { arrow, dot, fitCanvas, label, onResize, scale, type Surface } from '../kit/canvas';
import { wireRunControls } from '../kit/dom';
import { EventLog } from '../kit/eventLog';
import { fmt, logSlider, niceLr } from '../kit/format';
import { LossChart } from '../kit/lossChart';
import { Narrator } from '../kit/narrator';
import { hexToRgb, onThemeChange, readColors, type SimColors } from '../kit/theme';
import { part, type MountFn } from '../kit/types';

export interface ContourConfig {
  start: { w: number; b: number };
  maxSteps: number;
  view: { w: number[]; b: number[]; x: number[]; y: number[] };
  xLabel: string;
  yLabel: string;
  describe: string;
  presets: { id: string; label: string; lr: number }[];
}

const LR = logSlider(0.001, 0.25);

/** About `count` round-numbered ticks between lo and hi. */
function ticks(lo: number, hi: number, count = 5): number[] {
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6);
  return out;
}
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

export const mount: MountFn<{ data: Dataset; config: ContourConfig }> = (root, { data, config: cfg }) => {
  const fit = lineFit(data);
  const narr = new Narrator(part(root, 'narrator'));
  const log = new EventLog(part<HTMLUListElement>(root, 'log'));
  const dataCanvas = part<HTMLCanvasElement>(root, 'data');
  const mapCanvas = part<HTMLCanvasElement>(root, 'map');
  const chart = new LossChart(part<HTMLCanvasElement>(root, 'chart'));
  const readout = part(root, 'readout');
  const plain = part(root, 'plain');
  const presetsEl = part(root, 'presets');
  const lrInput = part<HTMLInputElement>(root, 'lr');
  const w0In = part<HTMLInputElement>(root, 'w0');
  const b0In = part<HTMLInputElement>(root, 'b0');

  // The valley's steep direction (largest-eigenvalue eigenvector of the constant
  // Hessian H = 2·[[mean x², mean x], [mean x, 1]]): zig-zag = flipping along it.
  const n = data.x.length;
  const sxx = (2 * data.x.reduce((a, v) => a + v * v, 0)) / n;
  const sx = (2 * data.x.reduce((a, v) => a + v, 0)) / n;
  const lam = (sxx + 2 + Math.sqrt((sxx - 2) ** 2 + 4 * sx * sx)) / 2;
  const steepDir = (() => {
    const v = [sx, lam - sxx];
    const m = Math.hypot(v[0], v[1]);
    return [v[0] / m, v[1] / m];
  })();
  const minLoss = fit.mse(fit.best.w, fit.best.b);

  let lr = cfg.presets.find((p) => p.id === 'good')?.lr ?? cfg.presets[0].lr;
  let w0 = cfg.start.w;
  let b0 = cfg.start.b;
  let path: [number, number][] = [];
  let end: OutcomeKind | null = null;
  let rises = 0;
  let flips = 0;
  let lastProj = 0;
  let zigLogged = false;
  let presetHint = '';
  let C: SimColors = readColors();
  let dataSurf!: Surface;
  let mapSurf!: Surface;
  let heat: HTMLCanvasElement | null = null;

  const cur = () => path[path.length - 1];
  const steps = () => path.length - 1;

  function reset() {
    path = [[w0, b0]];
    end = null;
    rises = 0;
    flips = 0;
    lastProj = 0;
    zigLogged = false;
    log.clear();
    chart.setValues([fit.mse(w0, b0)]);
    texts();
  }

  function step(): boolean {
    if (end) return false;
    const [w, b] = cur();
    const r = stepFit(fit, w, b, lr, steps() + 1, cfg.maxSteps);
    path.push([r.w, r.b]);
    const k = steps();
    chart.setValues(path.map(([pw, pb]) => fit.mse(pw, pb)));
    const before = fit.mse(w, b);
    const after = fit.mse(r.w, r.b);
    if (after > before && Number.isFinite(after)) {
      rises++;
      if (rises === 1)
        log.add(`step ${k}`, `<b>Loss went up</b> (${fmt(before, 2)} → ${fmt(after, 2)}): learning rate too big, you overshot across the valley.`);
    }
    const proj = (r.w - w) * steepDir[0] + (r.b - b) * steepDir[1];
    if (Math.abs(proj) > 1e-3) {
      if (lastProj && Math.sign(proj) !== Math.sign(lastProj)) flips++;
      else flips = 0;
      lastProj = proj;
    }
    if (flips >= 3 && !zigLogged) {
      zigLogged = true;
      log.add(`step ${k}`, '<b>Zig-zagging</b> across the valley: the walls are steep, so each step overshoots sideways.');
    }
    end = r.end;
    if (end === 'exploded') log.add(`step ${k}`, '💥 <b>Exploded.</b> Each step overshoots the valley by more than the last.');
    else if (end === 'converged') log.add(`step ${k}`, `🎯 <b>Converged</b>: w ≈ ${fmt(r.w, 2)}, b ≈ ${fmt(r.b, 2)}.`);
    else if (end === 'gave up') log.add(`step ${k}`, `⏱️ Stopped after ${k} steps, still not there.`);
    texts();
    return !end;
  }

  function phase() {
    const k = steps();
    const [w, b] = cur();
    const L = fit.mse(w, b);
    if (end === 'exploded')
      return narr.set(
        `Step ${k} · exploded`,
        '💥 Loss exploded. 🐶☕🔥 "This is fine."',
        'Each step overshoots the valley a bit more than the last, along the steep direction of the bowl. Lower the learning rate.',
      );
    if (end === 'converged')
      return narr.set(`Step ${k} · converged`, `🎯 Converged after ${k} steps`, "The dot reached the ×, so the line on the left is the best line through the data. Every step from here would be tiny.");
    if (end === 'gave up')
      return narr.set(`Step ${k} · out of time`, '⏱️ Still getting there', 'A bigger learning rate would be faster. Push it up until it starts to zig-zag.');
    if (k === 0)
      return narr.set(
        'Ready',
        'Two knobs, one bowl, seen from above',
        `Left: the data and the current line. Right: the loss for every (w, b): <b>lighter means lower</b>, and the × is the best line. The arrow on the dot points straight downhill. ${presetHint || 'Pick a preset, then press <b>Step</b>.'}`,
      );
    if (flips >= 2)
      return narr.set(
        `Step ${k} · zig-zag`,
        'Bouncing between the valley walls',
        'This valley is long and narrow: steep across, gentle along. A step that suits the gentle direction overshoots the steep one, so the dot bounces from wall to wall while creeping along.',
      );
    if (L > minLoss * 1.5 + 0.5)
      return narr.set(
        `Step ${k} · falling in`,
        'Falling into the valley',
        'Far from the bottom the slope is steep, so the steps are big and the line on the left swings quickly towards the points.',
      );
    narr.set(
      `Step ${k} · crawling`,
      'Creeping along the valley floor',
      'Down here the slope is gentle, so the steps are small. The line is almost right; these last steps fine-tune it.',
    );
  }

  function texts() {
    const [w, b] = cur();
    readout.textContent = `step ${steps()} · w = ${fmt(w)} · b = ${fmt(b)} · loss = ${fmt(fit.mse(w, b))}`;
    plain.textContent =
      Number.isFinite(w) && Math.abs(w) < 1e4 && Math.abs(b) < 1e4
        ? `In plain words, the line says: ${cfg.describe.replace('{w}', fmt(w, 1)).replace('{b}', fmt(b, 1))}.`
        : 'In plain words, the line says: nonsense. 🫠';
    phase();
  }

  // ---------- drawing ----------
  const pad = { l: 40, r: 12, t: 12, b: 34 };
  function drawData(w: number, b: number) {
    const { ctx, width, height } = dataSurf;
    const { view } = cfg;
    const x = scale(view.x[0], view.x[1], pad.l, width - pad.r);
    const y = scale(view.y[0], view.y[1], height - pad.b, pad.t);
    const mono = `11px ${C.fontMono}`;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = C.surface;
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    ctx.fillStyle = C.textMuted;
    ctx.font = mono;
    for (const v of ticks(view.y[0], view.y[1], 3)) {
      ctx.beginPath();
      ctx.moveTo(pad.l, Math.round(y(v)) + 0.5);
      ctx.lineTo(width - pad.r, Math.round(y(v)) + 0.5);
      ctx.stroke();
      ctx.textAlign = 'right';
      ctx.fillText(String(v), pad.l - 6, y(v) + 4);
    }
    ctx.textAlign = 'center';
    for (const v of ticks(view.x[0], view.x[1], 4)) ctx.fillText(String(v), x(v), height - pad.b + 14);
    ctx.font = `12px ${C.fontSans}`;
    ctx.fillText(cfg.xLabel, (pad.l + width - pad.r) / 2, height - 4);
    ctx.save();
    ctx.translate(11, (pad.t + height - pad.b) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(cfg.yLabel, 0, 0);
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.l, pad.t, width - pad.l - pad.r, height - pad.t - pad.b);
    ctx.clip();
    if (Number.isFinite(w) && Number.isFinite(b)) {
      ctx.strokeStyle = C.mark;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(x(view.x[0]), y(w * view.x[0] + b));
      ctx.lineTo(x(view.x[1]), y(w * view.x[1] + b));
      ctx.stroke();
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 1;
      data.x.forEach((xi, i) => {
        ctx.beginPath();
        ctx.moveTo(x(xi), y(data.y[i]));
        ctx.lineTo(x(xi), y(w * xi + b));
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
      // Label the line where it is on screen.
      const lx = view.x[0] + (view.x[1] - view.x[0]) * 0.08;
      const ly = clamp(w * lx + b, view.y[0] + 1, view.y[1] - 1);
      label(ctx, 'your line', x(lx), y(ly) - 12, C.mark, C.surface, mono);
    }
    ctx.restore();
    data.x.forEach((xi, i) => dot(ctx, x(xi), y(data.y[i]), 4.5, C.point, C.surface));
    // Residual label on the last point.
    const li = data.x.length - 1;
    if (Number.isFinite(w))
      label(ctx, 'miss', x(data.x[li]) - 6, (y(data.y[li]) + y(clamp(w * data.x[li] + b, view.y[0], view.y[1]))) / 2, C.textMuted, C.surface, mono, 'right');
  }

  function mapScales() {
    const { width, height } = mapSurf;
    return {
      xw: scale(cfg.view.w[0], cfg.view.w[1], pad.l, width - pad.r),
      yb: scale(cfg.view.b[0], cfg.view.b[1], height - pad.b, pad.t),
    };
  }

  function buildHeat() {
    const { width, height } = mapSurf;
    const { xw, yb } = mapScales();
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const pw = Math.round((width - pad.l - pad.r) * dpr);
    const ph = Math.round((height - pad.t - pad.b) * dpr);
    const off = document.createElement('canvas');
    off.width = pw;
    off.height = ph;
    const octx = off.getContext('2d')!;
    const img = octx.createImageData(pw, ph);
    const lo = hexToRgb(C.heatLow);
    const hi = hexToRgb(C.heatHigh);
    const BANDS = 14;
    const band = new Int16Array(pw * ph);
    const { w: vw, b: vb } = cfg.view;
    const corners = [vw[0], vw[1]].flatMap((cw) => [vb[0], vb[1]].map((cb) => fit.mse(cw, cb)));
    const maxLog = Math.log(Math.max(...corners) - minLoss + 1);
    for (let j = 0; j < ph; j++) {
      const bv = yb.invert(pad.t + j / dpr);
      for (let i = 0; i < pw; i++) {
        const wv = xw.invert(pad.l + i / dpr);
        const v = Math.log(fit.mse(wv, bv) - minLoss + 1) / maxLog;
        band[j * pw + i] = Math.min(BANDS - 1, Math.floor(Math.sqrt(Math.max(v, 0)) * BANDS));
      }
    }
    for (let k = 0; k < pw * ph; k++) {
      const i = k % pw;
      const f = band[k] / (BANDS - 1);
      const edge = (i + 1 < pw && band[k + 1] !== band[k]) || (k + pw < pw * ph && band[k + pw] !== band[k]);
      const mix = edge ? Math.min(1, f + 0.18) : f;
      for (let c = 0; c < 3; c++) img.data[k * 4 + c] = lo[c] + (hi[c] - lo[c]) * mix;
      img.data[k * 4 + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    heat = off;
  }

  function drawMap(w: number, b: number) {
    const { ctx, width, height } = mapSurf;
    const { xw, yb } = mapScales();
    const mono = `11px ${C.fontMono}`;
    if (!heat) buildHeat();
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = C.surface;
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(heat!, pad.l, pad.t, width - pad.l - pad.r, height - pad.t - pad.b);
    ctx.fillStyle = C.textMuted;
    ctx.font = mono;
    ctx.textAlign = 'center';
    const { w: vw, b: vb } = cfg.view;
    for (const v of ticks(vw[0], vw[1], 5)) ctx.fillText(String(v).replace('-', '−'), xw(v), height - pad.b + 14);
    ctx.textAlign = 'right';
    for (const v of ticks(vb[0], vb[1], 4)) ctx.fillText(String(v).replace('-', '−'), pad.l - 6, yb(v) + 4);
    ctx.font = `12px ${C.fontSans}`;
    ctx.textAlign = 'center';
    ctx.fillText('slope w →', (pad.l + width - pad.r) / 2, height - 4);
    ctx.save();
    ctx.translate(11, (pad.t + height - pad.b) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('intercept b', 0, 0);
    ctx.restore();

    // Best fit ×, labelled.
    const bx = xw(fit.best.w);
    const by = yb(fit.best.b);
    ctx.strokeStyle = C.text;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(bx - 5, by - 5);
    ctx.lineTo(bx + 5, by + 5);
    ctx.moveTo(bx + 5, by - 5);
    ctx.lineTo(bx - 5, by + 5);
    ctx.stroke();
    label(ctx, 'best line', bx + 9, by - 9, C.text, C.surface, mono);
    // Start ring, labelled.
    ctx.beginPath();
    ctx.arc(xw(w0), yb(b0), 6, 0, 7);
    ctx.strokeStyle = C.text;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    label(ctx, 'start', xw(w0) + 9, yb(b0) - 9, C.text, C.surface, mono);

    // Path so far (clipped), then the current point and its downhill arrow.
    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.l, pad.t, width - pad.l - pad.r, height - pad.t - pad.b);
    ctx.clip();
    ctx.strokeStyle = C.mark;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    path.slice(0, -1)
      .concat([[w, b]])
      .forEach(([pw, pb], i) => {
        const X = xw(clamp(pw, -1e4, 1e4));
        const Y = yb(clamp(pb, -1e4, 1e4));
        if (i === 0) ctx.moveTo(X, Y);
        else ctx.lineTo(X, Y);
      });
    ctx.stroke();
    ctx.restore();

    const inside = w >= vw[0] && w <= vw[1] && b >= vb[0] && b <= vb[1];
    if (inside) {
      if (!end) {
        const [gw, gb] = fit.grad(w, b);
        const sxp = xw(w - gw) - xw(w);
        const syp = yb(b - gb) - yb(b);
        const m = Math.hypot(sxp, syp);
        if (m > 1e-6) {
          arrow(ctx, xw(w), yb(b), xw(w) + (sxp / m) * 28, yb(b) + (syp / m) * 28, C.mark, 2);
          if (steps() === 0) label(ctx, 'downhill', xw(w) + (sxp / m) * 34, yb(b) + (syp / m) * 34 + 10, C.mark, C.surface, mono);
        }
      }
      dot(ctx, xw(w), yb(b), 6, C.mark, C.markRing);
    } else {
      label(ctx, '💥 point left the map', (pad.l + width - pad.r) / 2, pad.t + 18, C.bad, C.surface, `bold 12px ${C.fontSans}`, 'center');
    }
  }

  function render(t: number) {
    const n = path.length;
    const [w1, b1] = cur();
    const [wp, bp] = n > 1 ? path[n - 2] : cur();
    const w = wp + (w1 - wp) * t;
    const b = bp + (b1 - bp) * t;
    drawData(w, b);
    drawMap(w, b);
  }

  function resize() {
    dataSurf = fitCanvas(dataCanvas, 1.25, 340);
    mapSurf = fitCanvas(mapCanvas, 1.25, 340);
    heat = null;
  }

  // ---------- controls ----------
  const { loop, cleanup } = wireRunControls(root, {
    rates: { slow: 4, normal: 15, fast: 60 },
    tweenMs: 160,
    step,
    render,
    onReset: () => {
      reset();
      loop.redraw();
    },
    beforePlay: () => {
      if (end) reset();
    },
    beforeStep: () => {
      if (end) reset();
    },
  });

  const setLr = (v: number, moveSlider = true) => {
    lr = v;
    if (moveSlider) lrInput.value = String(LR.toPos(v));
    part(root, 'lr-out').textContent = String(v);
  };
  const setStart = (w: number, b: number) => {
    w0 = w;
    b0 = b;
    w0In.value = String(w);
    b0In.value = String(b);
    part(root, 'w0-out').textContent = fmt(w, 1);
    part(root, 'b0-out').textContent = fmt(b, 1);
    loop.pause();
    reset();
    loop.redraw();
  };
  const clearPressed = () => presetsEl.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', 'false'));

  presetsEl.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest<HTMLButtonElement>('button[data-preset]');
    if (!btn) return;
    const p = cfg.presets[Number(btn.dataset.preset)];
    setLr(p.lr);
    presetHint = `<b>${p.label}</b> (learning rate ${p.lr}). Press <b>Step</b> or <b>Play</b>.`;
    presetsEl.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
    loop.pause();
    reset();
    loop.redraw();
  });
  lrInput.addEventListener('input', () => {
    setLr(niceLr(LR.toValue(Number(lrInput.value))), false);
    presetHint = '';
    clearPressed();
    loop.pause();
    reset();
    loop.redraw();
  });
  const onStart = () => setStart(Number(w0In.value), Number(b0In.value));
  w0In.addEventListener('input', onStart);
  b0In.addEventListener('input', onStart);
  mapCanvas.addEventListener('click', (e) => {
    const rect = mapCanvas.getBoundingClientRect();
    const { xw, yb } = mapScales();
    const w = Math.round(xw.invert(e.clientX - rect.left) * 10) / 10;
    const b = Math.round(yb.invert(e.clientY - rect.top) * 10) / 10;
    setStart(clamp(w, cfg.view.w[0], cfg.view.w[1]), clamp(b, cfg.view.b[0], cfg.view.b[1]));
  });

  resize();
  setLr(lr);
  setStart(cfg.start.w, cfg.start.b);
  const stopResize = onResize(part(root, 'panels'), () => {
    resize();
    loop.redraw();
  });
  const stopTheme = onThemeChange(() => {
    C = readColors();
    heat = null;
    loop.redraw();
    chart.draw();
  });

  return {
    destroy() {
      cleanup();
      stopResize();
      stopTheme();
      chart.destroy();
    },
  };
};
