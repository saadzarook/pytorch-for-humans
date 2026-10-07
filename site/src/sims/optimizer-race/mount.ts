/**
 * The optimizer race: SGD, Momentum and Adam start at the same spot on a
 * curved-valley loss landscape. Ported from reference/ml-sim-lab.html.
 *
 * Narration numbers (arrival order, the divergence threshold) come from
 * facts.json, which real torch.optim produces (facts.py) and Vitest checks
 * the JS race reproduces.
 */
import cfg from './config.json';
import facts from './facts.json';
import { createState, f, floorY, grad, isDiverged, OPT_NAMES, sliderLr, stepState, type OptName, type OptState } from './math';
import { arrow, chartFrame, fitCanvas, label, onResize, type Surface } from '../kit/canvas';
import { wireRunControls } from '../kit/dom';
import { EventLog } from '../kit/eventLog';
import { Narrator } from '../kit/narrator';
import { onThemeChange, readColors, type SimColors } from '../kit/theme';
import { part, type MountFn } from '../kit/types';

interface Runner extends OptState {
  blurb: string;
  slot: 0 | 1 | 2;
  dead: boolean;
  path: [number, number][];
  losses: number[];
  floor: boolean;
  over: boolean;
  zig: number;
  zigLogged: boolean;
  lastDy: number;
  arrived: boolean;
}

const BLURB: Record<OptName, string> = {
  SGD: 'step = lr × gradient',
  Momentum: 'remembers past steps, builds speed',
  Adam: 'rescales steps by recent gradient size',
};

export const mount: MountFn<{ start?: [number, number] }> = (root, options) => {
  const [X0, X1] = cfg.view.x;
  const [Y0, Y1] = cfg.view.y;
  const narr = new Narrator(part(root, 'narrator'));
  const log = new EventLog(part<HTMLUListElement>(root, 'log'));
  const cv = part<HTMLCanvasElement>(root, 'map');
  const lc = part<HTMLCanvasElement>(root, 'loss');
  const readout = part(root, 'readout');
  const lrInput = part<HTMLInputElement>(root, 'lr');
  const lrOut = part(root, 'lr-out');

  let [sx, sy] = options.start ?? (cfg.start as [number, number]);
  let lr = cfg.defaultLr;
  let step = 0;
  let moved = false;
  let C: SimColors = readColors();
  let map!: Surface;
  let chart!: Surface;
  let heat: HTMLCanvasElement | null = null;
  const runners: Runner[] = OPT_NAMES.map((name, i) => ({
    ...createState(name, sx, sy),
    blurb: BLURB[name],
    slot: i as 0 | 1 | 2,
    dead: false,
    path: [],
    losses: [],
    floor: false,
    over: false,
    zig: 0,
    zigLogged: false,
    lastDy: 0,
    arrived: false,
  }));

  const px = (x: number) => ((x - X0) / (X1 - X0)) * map.width;
  const py = (y: number) => (1 - (y - Y0) / (Y1 - Y0)) * map.height;
  const color = (o: Runner) => C.series[o.slot];
  const logAt = (html: string) => log.add(`step ${step}`, html);

  function reset() {
    for (const o of runners) {
      Object.assign(o, createState(o.name, sx, sy), {
        dead: false,
        path: [[sx, sy]],
        losses: [f(sx, sy)],
        floor: false,
        over: false,
        zig: 0,
        zigLogged: false,
        lastDy: 0,
        arrived: false,
      });
    }
    step = 0;
    log.clear();
  }

  /** One step for everyone, logging key moments exactly as the prototype does. */
  function stepAll(): boolean {
    step++;
    for (const o of runners) {
      if (o.dead) continue;
      const oy = o.y;
      stepState(o, lr);
      if (isDiverged(o)) {
        o.dead = true;
        logAt(`<b>${o.name}</b> blew up. Each step overshot further than the last. This learning rate is too big for it.`);
        continue;
      }
      o.path.push([o.x, o.y]);
      const L = f(o.x, o.y);
      const prevL = o.losses[o.losses.length - 1];
      o.losses.push(L);
      const dy = o.y - oy;
      if (!o.floor && Math.abs(o.y - floorY(o.x)) < 0.08) {
        o.floor = true;
        logAt(`<b>${o.name}</b> reached the valley floor. The slope here is gentle, so progress slows down.`);
      }
      if (!o.over && step > 2 && L > prevL * 1.0001 && prevL > 0.01) {
        o.over = true;
        logAt(
          o.name === 'Momentum'
            ? '<b>Momentum</b> overshot, and its loss went <i>up</i>. Like a heavy ball, it rolled past the low point and will swing back.'
            : `<b>${o.name}</b>'s loss went up. It stepped past the low point.`,
        );
      }
      if (Math.abs(dy) > 0.005) {
        if (o.lastDy && Math.sign(dy) !== Math.sign(o.lastDy)) o.zig++;
        else o.zig = 0;
        o.lastDy = dy;
      }
      if (o.zig >= 3 && !o.zigLogged) {
        o.zigLogged = true;
        logAt(`<b>${o.name}</b> is zigzagging between the valley walls. The walls are steep, so each step overshoots sideways.`);
      }
      if (!o.arrived && L < cfg.arriveLoss) {
        o.arrived = true;
        logAt(`<b>${o.name}</b> reached the minimum (loss below ${cfg.arriveLoss}) in <b>${step} steps</b>.`);
      }
    }
    return !(step >= cfg.maxSteps || runners.every((o) => o.dead || o.arrived));
  }

  function phase() {
    const alive = runners.filter((o) => !o.dead);
    if (step === 0)
      return narr.set(
        moved ? 'New start point' : 'Ready',
        'Same start, same learning rate, three strategies',
        'Each dot is a model with two weights, and its position on the map <i>is</i> those two weights. The colour is the loss: <b>lighter means lower</b>, and the ✕ marks the lowest point. The small arrow on each dot points straight downhill, which is the negative gradient. Press <b>Step</b> a few times, then <b>Play</b>.',
      );
    if (!alive.length)
      return narr.set(
        'Game over',
        'Everyone diverged',
        `The learning rate is so big that each step lands somewhere steeper than the last, so the steps keep growing. Drag the slider back to about ${facts.defaultLr} and press Reset.`,
      );
    if (alive.some((o) => !o.floor))
      return narr.set(
        'Phase 1 of 3',
        'Falling into the valley',
        'The valley walls are steep, so the gradients are big and so are the steps. The arrows point mostly <i>sideways</i>, toward the valley floor (dashed line), not toward the ✕. Gradient descent only ever sees the local slope, never the destination.',
      );
    if (alive.some((o) => !o.arrived))
      return narr.set(
        'Phase 2 of 3',
        'Crawling along the valley floor',
        '<b>SGD</b>: the gradient is small down here, so its steps are small and it slows to a crawl. <b>Momentum</b> keeps the speed it built up, so it moves faster but can overshoot. <b>Adam</b> divides each step by the recent gradient size, so its steps stay about the same size even on gentle slopes. Watch the loss chart: the curve that drops fastest is winning.',
      );
    narr.set(
      'Phase 3 of 3',
      'Everyone made it',
      `Check the log for who arrived first (at lr ${facts.defaultLr}: ${facts.orderText}). Then push the learning rate toward ${facts.topLr}: SGD is the first to blow up, because its step is directly proportional to the steep wall's gradient.`,
    );
  }

  function buildHeat() {
    const d = Math.min(window.devicePixelRatio || 1, 3);
    const w = Math.round(map.width * d);
    const h = Math.round(map.height * d);
    const off = document.createElement('canvas');
    off.width = w;
    off.height = h;
    const o = off.getContext('2d')!;
    const img = o.createImageData(w, h);
    const lo = hex(C.heatLow);
    const hi = hex(C.heatHigh);
    const fmax = Math.log1p(18);
    for (let j = 0; j < h; j++)
      for (let i = 0; i < w; i++) {
        const x = X0 + (i / w) * (X1 - X0);
        const y = Y1 - (j / h) * (Y1 - Y0);
        const lv = Math.log1p(f(x, y));
        const band = (lv * 5) % 1;
        const t = Math.min(1, lv / fmax + (band < 0.06 ? 0.15 : 0));
        const k = (j * w + i) * 4;
        for (let c = 0; c < 3; c++) img.data[k + c] = lo[c] + (hi[c] - lo[c]) * t;
        img.data[k + 3] = 255;
      }
    o.putImageData(img, 0, 0);
    heat = off;
  }

  function draw() {
    const ctx = map.ctx;
    if (!heat) buildHeat();
    ctx.drawImage(heat!, 0, 0, map.width, map.height);
    const mono = `12px ${C.fontMono}`;
    // Valley floor (dashed) + label.
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = C.textMuted;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i <= 120; i++) {
      const x = X0 + (i / 120) * (X1 - X0);
      if (i) ctx.lineTo(px(x), py(floorY(x)));
      else ctx.moveTo(px(x), py(floorY(x)));
    }
    ctx.stroke();
    ctx.setLineDash([]);
    // Right-aligned near the right edge, so it never runs off a narrow canvas.
    label(ctx, 'valley floor', px(X1 - 0.15), py(floorY(X1 - 0.6)) + 18, C.textMuted, C.surface, mono, 'right');
    // Minimum ✕.
    const mx = px(0);
    const my = py(0);
    ctx.strokeStyle = C.text;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(mx - 7, my - 7);
    ctx.lineTo(mx + 7, my + 7);
    ctx.moveTo(mx + 7, my - 7);
    ctx.lineTo(mx - 7, my + 7);
    ctx.stroke();
    // Below-left of the ✕: the dots arrive from above, so this stays readable.
    label(ctx, 'lowest loss', mx - 10, my + 18, C.text, C.surface, mono, 'right');
    // Start ring.
    ctx.beginPath();
    ctx.arc(px(sx), py(sy), 7, 0, 7);
    ctx.strokeStyle = C.text;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    label(ctx, 'start', px(sx) + 10, py(sy) - 12, C.text, C.surface, mono);
    // Trails.
    for (const o of runners) {
      const col = color(o);
      ctx.strokeStyle = col;
      ctx.lineWidth = 2.5;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      o.path.forEach(([x, y], i) => (i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y))));
      ctx.stroke();
      o.path.forEach(([x, y], i) => {
        if (i && i < o.path.length - 1) {
          ctx.beginPath();
          ctx.arc(px(x), py(y), 1.8, 0, 7);
          ctx.fillStyle = col;
          ctx.fill();
        }
      });
    }
    // Current dots, downhill arrows and direct name labels.
    for (const o of runners) {
      if (o.dead) continue;
      const col = color(o);
      const X = px(o.x);
      const Yp = py(o.y);
      const [gx, gy] = grad(o.x, o.y);
      const m = Math.hypot(gx, gy);
      if (m > 1e-4) arrow(ctx, X, Yp, X - (gx / m) * 30, Yp + (gy / m) * 30, col, 2);
      ctx.beginPath();
      ctx.arc(X, Yp, 7, 0, 7);
      ctx.fillStyle = col;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = C.surface;
      ctx.stroke();
      label(ctx, o.name, X + 10, Yp + 14 + o.slot * 13, col, C.surface, `bold 11px ${C.fontSans}`);
    }
    drawLoss();
    phase();
    readout.innerHTML =
      `<span>step ${step}</span>` +
      runners
        .map(
          (o) =>
            `<span><i class="sim-swatch" style="background:${color(o)}"></i><b>${o.name}</b> (${o.blurb}): ${
              o.dead ? '<span class="sim-warn">diverged</span>' : `loss ${f(o.x, o.y).toFixed(4)}`
            }</span>`,
        )
        .join('');
  }

  function drawLoss() {
    const c = chart.ctx;
    const pad = { l: 44, r: 10, t: 10, b: 22 };
    const xMax = Math.max(60, Math.ceil(step / 20) * 20);
    const lo = -3;
    const hi = 1.5;
    const yMap = (v: number) => pad.t + (1 - (Math.log10(v) - lo) / (hi - lo)) * (chart.height - pad.t - pad.b);
    chartFrame(c, {
      width: chart.width,
      height: chart.height,
      pad,
      yTicks: [10, 1, 0.1, 0.01, 0.001],
      yFmt: String,
      yMap,
      xMax,
      colors: { background: C.sunk, grid: C.grid, text: C.textMuted, font: C.fontMono },
    });
    for (const o of runners) {
      c.strokeStyle = color(o);
      c.lineWidth = 2;
      c.beginPath();
      o.losses.forEach((L, i) => {
        const X = pad.l + (i / xMax) * (chart.width - pad.l - pad.r);
        const Yv = yMap(Math.min(Math.max(L, 1e-3), 30));
        if (i) c.lineTo(X, Yv);
        else c.moveTo(X, Yv);
      });
      c.stroke();
      // Direct label at the line's end (secondary encoding: colour is never the only cue).
      const i = o.losses.length - 1;
      label(
        c,
        o.name,
        Math.min(pad.l + (i / xMax) * (chart.width - pad.l - pad.r) + 4, chart.width - 60),
        // Lines that finish at the same height (e.g. both clamped at 0.001) would collide: stack by slot.
        yMap(Math.min(Math.max(o.losses[i], 1e-3), 30)) - 8 - o.slot * 11,
        color(o),
        C.sunk,
        `bold 10px ${C.fontSans}`,
      );
    }
  }

  function resize() {
    map = fitCanvas(cv, 2, 460);
    chart = fitCanvas(lc, 1 / 0.55, 260);
    heat = null;
    draw();
  }

  function setLrFromSlider() {
    lr = sliderLr(Number(lrInput.value));
    lrOut.innerHTML =
      lr.toFixed(3) + (Number(lrInput.value) >= facts.sgdDivergesFromPos ? ' <span class="sim-warn">chaos zone</span>' : '');
  }

  const { loop, cleanup } = wireRunControls(root, {
    rates: cfg.rates,
    step: stepAll,
    render: () => draw(),
    onReset: () => {
      reset();
      draw();
    },
    beforePlay: () => {
      if (runners.every((o) => o.dead || o.arrived)) reset();
    },
    beforeStep: () => {
      if (runners.every((o) => o.dead || o.arrived)) reset();
    },
  });

  lrInput.max = String(cfg.sliderSteps);
  lrInput.value = String(Math.round(((Math.log10(cfg.defaultLr) - Math.log10(cfg.lrRange[0])) / (Math.log10(cfg.lrRange[1]) - Math.log10(cfg.lrRange[0]))) * cfg.sliderSteps));
  setLrFromSlider();
  lrInput.addEventListener('input', setLrFromSlider);
  cv.addEventListener('click', (e) => {
    const r = cv.getBoundingClientRect();
    sx = X0 + ((e.clientX - r.left) / r.width) * (X1 - X0);
    sy = Y1 - ((e.clientY - r.top) / r.height) * (Y1 - Y0);
    moved = true;
    loop.pause();
    reset();
    draw();
  });

  reset();
  const stopResize = onResize(cv.parentElement!, resize);
  resize();
  const stopTheme = onThemeChange(() => {
    C = readColors();
    heat = null;
    draw();
  });

  return {
    destroy() {
      cleanup();
      stopResize();
      stopTheme();
    },
  };
};

function hex(c: string): [number, number, number] {
  let h = c.trim().replace('#', '');
  if (h.length === 3) h = h.split('').map((x) => x + x).join('');
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number];
}
