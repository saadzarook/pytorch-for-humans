/**
 * Gradient descent in 1D: a ball hopping down a loss curve.
 *
 * Course sim standard (docs/SIM_STANDARD.md): narrated phases, Step + speed
 * control, the arithmetic on show, labels drawn on the canvas, and a log of
 * key moments (overshoots, loss going up, settling, exploding).
 */
import { classify, makeCurve, nextW, type Curve, type CurveSpec } from './math';
import { arrowHead, dot, fitCanvas, label, onResize, scale, type Surface } from '../kit/canvas';
import { wireRunControls } from '../kit/dom';
import { EventLog } from '../kit/eventLog';
import { bindLogSlider, bindPresets } from '../kit/controls';
import { fmt, niceLr } from '../kit/format';
import { LossChart } from '../kit/lossChart';
import { Narrator } from '../kit/narrator';
import { onThemeChange, readColors, type SimColors } from '../kit/theme';
import { part, type MountFn } from '../kit/types';

export interface BallOptions {
  curves: Record<string, CurveSpec>;
  curve: string;
  /** From the lesson's facts.json: the bowl's chaos threshold, for the narration. */
  chaosLr?: string;
}

export const mount: MountFn<BallOptions> = (root, options) => {
  const narr = new Narrator(part(root, 'narrator'));
  const log = new EventLog(part<HTMLUListElement>(root, 'log'));
  const canvas = part<HTMLCanvasElement>(root, 'curve');
  const chart = new LossChart(part<HTMLCanvasElement>(root, 'chart'));
  const readoutEl = part(root, 'readout');
  const equationEl = part(root, 'equation');
  const presetsEl = part(root, 'presets');
  const lrInput = part<HTMLInputElement>(root, 'lr');
  const startInput = part<HTMLInputElement>(root, 'start');
  const curveSelect = root.querySelector<HTMLSelectElement>('[data-part="curve-select"]');
  const curves = Object.fromEntries(Object.entries(options.curves).map(([k, s]) => [k, makeCurve(s)]));

  let curve: Curve = curves[options.curve];
  let lr = curve.defaultLr;
  let start = curve.defaultStart;
  let ws: number[] = [];
  let outcome: ReturnType<typeof classify> = null;
  let flips = 0;
  let rises = 0;
  let presetHint = '';
  let C: SimColors = readColors();
  let surf: Surface = fitCanvas(canvas, 1.7, 360);

  const w = () => ws[ws.length - 1];
  const steps = () => ws.length - 1;

  function reset() {
    ws = [start];
    outcome = null;
    flips = 0;
    rises = 0;
    log.clear();
    chart.setValues([curve.f(start)]);
    updateTexts();
  }

  function step(): boolean {
    if (outcome) return false;
    const prev = w();
    const next = nextW(curve, prev, lr);
    ws.push(next);
    const n = steps();
    chart.setValues(ws.map(curve.f));
    const lossPrev = curve.f(prev);
    const lossNow = curve.f(next);
    // Key moments.
    if (Number.isFinite(lossNow) && Math.sign(curve.df(next)) !== Math.sign(curve.df(prev)) && Math.abs(curve.df(next)) > 1e-3) {
      flips++;
      if (flips === 1)
        log.add(`step ${n}`, `Hopped <b>past the bottom</b> to the other side: the step (${fmt(lr * curve.df(prev), 2)}) was bigger than the distance to the low point.`);
    }
    if (lossNow > lossPrev && Number.isFinite(lossNow)) {
      rises++;
      if (rises === 1)
        log.add(`step ${n}`, `<b>Loss went up</b> (${fmt(lossPrev, 2)} → ${fmt(lossNow, 2)}): learning rate too big, you overshot.`);
    }
    outcome = classify(curve, prev, next, n);
    if (outcome?.kind === 'exploded') log.add(`step ${n}`, '💥 <b>Exploded.</b> Every hop overshoots by more than the last.');
    else if (outcome?.kind === 'bottom') log.add(`step ${n}`, `🎯 <b>Settled at the bottom</b> (w = ${fmt(next, 2)}).`);
    else if (outcome?.kind === 'local') log.add(`step ${n}`, `🦋 <b>Settled in a local minimum</b> at w = ${fmt(next, 2)}, not the deepest valley.`);
    else if (outcome?.kind === 'gave up') log.add(`step ${n}`, `⏱️ Gave up after ${n} steps without settling.`);
    updateTexts();
    return !outcome;
  }

  function phase() {
    const n = steps();
    if (outcome?.kind === 'exploded')
      return narr.set(
        `Step ${n} · exploded`,
        '💥 Loss exploded. 🐶☕🔥 "This is fine." (It is not fine.)',
        `Every hop overshoots the valley by more than the last, so the ball flies off.${
          curve.id === 'bowl' && options.chaosLr ? ` For this bowl that happens above lr = ${options.chaosLr}.` : ''
        } Try a smaller learning rate.`,
      );
    if (outcome?.kind === 'bottom')
      return narr.set(`Step ${n} · settled`, `🎯 Settled at the bottom after ${n} steps`, 'The slope here is flat, so the step (learning rate × slope) is about zero: gradient descent stops by itself.');
    if (outcome?.kind === 'local')
      return narr.set(
        `Step ${n} · stuck`,
        '🦋 "Is this the global minimum?" Nope.',
        `It settled at w = ${fmt(w(), 2)}, a <b>local minimum</b>: the slope is flat here too, so it can't tell. There's a deeper valley at w ≈ ${fmt(curve.globalMin, 2)}. Try another start, or a bigger step.`,
      );
    if (outcome?.kind === 'gave up')
      return narr.set(`Step ${n} · gave up`, '⏱️ Still not settled', 'Too slow? Too bouncy? Try another learning rate.');
    if (n === 0)
      return narr.set(
        'Ready',
        'The ball is your current guess',
        `The curve is the loss for every value of w. The dashed line is the slope your feet feel, and the arrow shows where the next hop lands: <b>w − lr × slope</b>. ${presetHint || 'Pick a preset, then press <b>Step</b>.'}`,
      );
    if (rises > 0 && curve.f(w()) > curve.f(ws[ws.length - 2]))
      return narr.set(`Step ${n} · overshooting`, 'The loss went UP', 'Each hop jumps past the bottom and lands higher than before. The learning rate is too big for this slope.');
    if (flips > 0)
      return narr.set(`Step ${n} · bouncing`, 'Overshooting, but shrinking', 'It jumps past the bottom each time, yet every bounce is smaller than the last, so it still settles. Watch the loss chart zig-zag down.');
    const shrink = n >= 2 ? Math.abs(ws[n] - ws[n - 1]) / Math.max(Math.abs(ws[n - 1] - ws[n - 2]), 1e-12) : 1;
    narr.set(
      `Step ${n} · rolling downhill`,
      shrink < 0.98 ? 'Steps shrink as the slope flattens' : 'Feeling the slope, stepping downhill',
      'Every step is learning rate × slope. The learning rate never changes, but near the bottom the slope gets flatter, so the steps get smaller on their own.',
    );
  }

  function updateTexts() {
    const cur = w();
    const slope = curve.df(cur);
    const next = cur - lr * slope;
    readoutEl.textContent = `step ${steps()} · w = ${fmt(cur)} · slope = ${fmt(slope)} · loss = ${fmt(curve.f(cur))}`;
    const s = slope < 0 ? `(${fmt(slope)})` : fmt(slope);
    equationEl.textContent = Number.isFinite(next) && !outcome ? `next w = w − lr × slope = ${fmt(cur)} − ${lr} × ${s} = ${fmt(next)}` : '';
    phase();
  }

  const visible = (x: number) => Number.isFinite(x) && x >= curve.domain[0] && x <= curve.domain[1];

  function render(t: number) {
    const { ctx, width, height } = surf;
    const pad = { l: 36, r: 14, t: 14, b: 30 };
    const x = scale(curve.domain[0], curve.domain[1], pad.l, width - pad.r);
    const y = scale(curve.range[0], curve.range[1], height - pad.b, pad.t);
    const mono = `11px ${C.fontMono}`;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = C.surface;
    ctx.fillRect(0, 0, width, height);

    // Axes (recessive).
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, height - pad.b + 0.5);
    ctx.lineTo(width - pad.r, height - pad.b + 0.5);
    ctx.moveTo(pad.l + 0.5, pad.t);
    ctx.lineTo(pad.l + 0.5, height - pad.b);
    ctx.stroke();
    ctx.fillStyle = C.textMuted;
    ctx.font = `12px ${C.fontSans}`;
    ctx.textAlign = 'right';
    ctx.fillText('w →', width - pad.r, height - 8);
    ctx.textAlign = 'left';
    ctx.fillText('loss', 4, pad.t + 4);
    ctx.font = mono;
    ctx.textAlign = 'center';
    const [d0, d1] = curve.domain;
    for (let v = Math.ceil(d0); v <= Math.floor(d1); v++) {
      if (v % 2 !== 0 && width < 480) continue;
      ctx.fillText(String(v).replace('-', '−'), x(v), height - pad.b + 14);
    }

    // The loss curve.
    ctx.strokeStyle = C.curve;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= 240; i++) {
      const wv = d0 + ((d1 - d0) * i) / 240;
      const py = y(Math.min(curve.f(wv), curve.range[1] * 1.2));
      if (i === 0) ctx.moveTo(x(wv), py);
      else ctx.lineTo(x(wv), py);
    }
    ctx.stroke();

    // Lowest point, labelled on the canvas.
    const gm = curve.globalMin;
    ctx.strokeStyle = C.textMuted;
    ctx.setLineDash([2, 4]);
    // Guide from the top of the plot down to the minimum; the space above the
    // lowest point is always empty, so the label never collides with the ball.
    ctx.beginPath();
    ctx.moveTo(x(gm), pad.t + 18);
    ctx.lineTo(x(gm), y(curve.f(gm)));
    ctx.stroke();
    ctx.setLineDash([]);
    label(ctx, 'lowest point', x(gm), pad.t + 8, C.textMuted, C.surface, mono, 'center');

    // Trail.
    const trail = ws.slice(Math.max(0, ws.length - 30), -1);
    trail.forEach((wv, i) => {
      if (!visible(wv)) return;
      ctx.globalAlpha = 0.15 + (0.45 * (i + 1)) / trail.length;
      ctx.beginPath();
      ctx.arc(x(wv), y(curve.f(wv)), 3, 0, Math.PI * 2);
      ctx.fillStyle = C.mark;
      ctx.fill();
    });
    ctx.globalAlpha = 1;

    // The ball, hopping from its previous position.
    const cur = w();
    const prev = ws.length > 1 ? ws[ws.length - 2] : cur;
    const wx = prev + (cur - prev) * t;
    const ly = curve.f(prev) + (curve.f(cur) - curve.f(prev)) * t;
    const hop = Math.min(36, Math.abs(x(cur) - x(prev)) * 0.25) * Math.sin(Math.PI * t);
    const bx = x(wx);
    const by = y(ly) - hop;
    const r = 9;
    if (visible(cur) && curve.f(cur) <= curve.range[1] * 1.2) {
      if (t >= 1) {
        // Tangent (the slope your feet feel), labelled.
        const slope = curve.df(cur);
        const kx = x(1) - x(0);
        const ky = y(1) - y(0);
        const dy = (slope * ky) / kx;
        const len = Math.hypot(1, dy);
        const L = Math.min(90, width * 0.18);
        const cy = y(curve.f(cur));
        ctx.strokeStyle = C.tangent;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(bx - L / len, cy - (dy / len) * L);
        ctx.lineTo(bx + L / len, cy + (dy / len) * L);
        ctx.stroke();
        ctx.setLineDash([]);

        // Next hop: arrow along the floor + ghost ball.
        const next = cur - lr * slope;
        if (!outcome && Number.isFinite(next)) {
          const nx = x(Math.max(d0, Math.min(d1, next)));
          const floor = height - pad.b - 8;
          if (Math.abs(nx - bx) > 4) {
            ctx.strokeStyle = C.mark;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(bx, floor);
            ctx.lineTo(nx, floor);
            ctx.stroke();
            arrowHead(ctx, nx, floor, nx > bx ? 0 : Math.PI, 7, C.mark);
            label(ctx, '−lr × slope', (bx + nx) / 2, floor - 12, C.mark, C.surface, mono, 'center');
          }
          if (visible(next) && curve.f(next) <= curve.range[1]) {
            ctx.strokeStyle = C.mark;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.arc(x(next), y(curve.f(next)) - r, r, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      }
      dot(ctx, bx, by - r, r, C.mark, C.markRing);
      // The slope your feet feel, right next to the ball (flipped near the right edge).
      if (t >= 1) {
        const right = bx < width * 0.7;
        label(ctx, `slope ${fmt(curve.df(cur), 2)}`, bx + (right ? r + 8 : -r - 8), by - r, C.tangent, C.surface, mono, right ? 'left' : 'right');
      }
    } else {
      // Off the visible world: point at where it went.
      const right = cur > (d0 + d1) / 2;
      const ax = right ? width - pad.r - 10 : pad.l + 10;
      arrowHead(ctx, ax, pad.t + 20, right ? -Math.PI / 4 : (-3 * Math.PI) / 4, 10, C.bad);
      label(ctx, `ball flew off to w = ${fmt(cur, 1)}`, right ? ax - 14 : ax + 14, pad.t + 40, C.bad, C.surface, `bold 12px ${C.fontSans}`, right ? 'right' : 'left');
    }
  }

  // ---------- controls ----------
  function apply(c: Curve, newLr: number, newStart: number) {
    curve = c;
    lr = newLr;
    start = newStart;
    startInput.min = String(c.domain[0] + 0.1);
    startInput.max = String(c.domain[1] - 0.1);
    startInput.value = String(newStart);
    lrSlider.set(newLr);
    part(root, 'start-out').textContent = fmt(start, 1);
    reset();
  }

  const { loop, cleanup } = wireRunControls(root, {
    rates: { slow: 2, normal: 6, fast: 16 },
    tweenMs: 220,
    step,
    render,
    onReset: () => {
      reset();
      loop.redraw();
    },
    beforePlay: () => {
      if (outcome) reset();
    },
    beforeStep: () => {
      if (outcome) reset();
    },
  });

  const presets = bindPresets(presetsEl, curve.presets, (p) => {
    loop.pause();
    presetHint = `<b>${p.label}</b>: ${p.hint} Press <b>Step</b> or <b>Play</b>.`;
    apply(curve, p.lr, p.start);
    loop.redraw();
  });
  const lrSlider = bindLogSlider(lrInput, part(root, 'lr-out'), {
    min: 0.001,
    max: 1.5,
    steps: 1000,
    value: lr,
    snap: niceLr,
    onInput: (v) => {
      loop.pause();
      lr = v;
      presetHint = '';
      presets.clearPressed();
      reset();
      loop.redraw();
    },
  });
  startInput.addEventListener('input', () => {
    loop.pause();
    start = Number(startInput.value);
    presetHint = '';
    presets.clearPressed();
    part(root, 'start-out').textContent = fmt(start, 1);
    reset();
    loop.redraw();
  });
  curveSelect?.addEventListener('change', () => {
    loop.pause();
    const c = curves[curveSelect.value];
    presetHint = '';
    apply(c, c.defaultLr, c.defaultStart);
    presets.setPresets(c.presets);
    loop.redraw();
  });

  apply(curve, curve.defaultLr, curve.defaultStart);
  const stopResize = onResize(part(root, 'stage'), () => {
    surf = fitCanvas(canvas, 1.7, 360);
    loop.redraw();
  });
  loop.redraw();
  const stopTheme = onThemeChange(() => {
    C = readColors();
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
