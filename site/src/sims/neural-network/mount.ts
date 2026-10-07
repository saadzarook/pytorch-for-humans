/**
 * Watch a network learn: a 2 → H → H → 1 network trains live (backprop + Adam
 * in plain JS). Shading = the network's guess, solid line = decision boundary,
 * red rings = points it currently gets wrong.
 * Ported from reference/ml-sim-lab.html; maths checked against PyTorch.
 */
import cfg from './config.json';
import { evaluate, genData, initNet, predict, trainEpoch, type Activation, type DatasetName, type Net, type Point } from './math';
import { chartFrame, fitCanvas, onResize, type Surface } from '../kit/canvas';
import { wireRunControls } from '../kit/dom';
import { EventLog } from '../kit/eventLog';
import { bindLogSlider } from '../kit/controls';
import { Narrator } from '../kit/narrator';
import { hexToRgb, onThemeChange, readColors, type SimColors } from '../kit/theme';
import { part, type MountFn } from '../kit/types';

export const mount: MountFn = (root) => {
  const narr = new Narrator(part(root, 'narrator'));
  const log = new EventLog(part<HTMLUListElement>(root, 'log'));
  const cv = part<HTMLCanvasElement>(root, 'map');
  const lc = part<HTMLCanvasElement>(root, 'loss');
  const readout = part(root, 'readout');
  const dsSel = part<HTMLSelectElement>(root, 'dataset');
  const hSel = part<HTMLSelectElement>(root, 'hidden');
  const actSel = part<HTMLSelectElement>(root, 'activation');
  const lrInput = part<HTMLInputElement>(root, 'lr');
  const lrOut = part(root, 'lr-out');

  const DOM = cfg.domain;
  const G = cfg.grid;
  const pv = new Float64Array(G * G);
  const off = document.createElement('canvas');
  off.width = G;
  off.height = G;
  const octx = off.getContext('2d')!;
  const img = octx.createImageData(G, G);

  let data: Point[] = [];
  let net!: Net;
  let epoch = 0;
  let hist: number[] = [];
  let milestones = new Set<number>();
  let lr = 0; // set by the slider binding below
  let Hn = cfg.defaultHidden;
  let act: Activation = 'tanh';
  let ds: DatasetName = 'circle';
  let last = { loss: 0, accuracy: 0, wrong: [] as Point[] };
  let C: SimColors = readColors();
  let map!: Surface;
  let chart!: Surface;

  function build() {
    data = genData(ds);
    net = initNet(Hn, act);
    epoch = 0;
    hist = [];
    milestones = new Set();
    log.clear();
    last = evaluate(net, data);
  }

  function epochStep(): boolean {
    trainEpoch(net, data, lr);
    epoch++;
    last = evaluate(net, data);
    hist.push(last.loss);
    for (const m of [0.75, 0.9, 1])
      if (last.accuracy >= m - 1e-9 && !milestones.has(m)) {
        milestones.add(m);
        log.add(
          `epoch ${epoch}`,
          m === 1 ? 'Every point is on the correct side: <b>100% accuracy</b>.' : `Accuracy passed <b>${m * 100}%</b>. ${last.wrong.length} points still wrong.`,
        );
      }
    if (epoch === 1) log.add('epoch 1', 'First weight update. The boundary moved for the first time.');
    return true;
  }

  function phase() {
    const wrongN = last.wrong.length;
    if (epoch === 0)
      return narr.set(
        'Epoch 0 · untrained',
        'Random weights, random guesses',
        `The network starts with random weights, so the boundary line is just wherever chance put it. ${wrongN} of ${data.length} points are on the wrong side (ringed in red). Press <b>1 epoch</b> to see a single learning step, or <b>Play</b>.`,
      );
    if (last.accuracy >= 0.995)
      return narr.set(
        `Epoch ${epoch} · solved`,
        'Every point is on the right side',
        'The boundary is already correct. Loss still drifts down because the network keeps getting more <i>confident</i>: the shading gets stronger, but the line barely moves. Now try harder data or fewer neurons.',
      );
    const n = hist.length;
    if (n > 200 && Math.abs(hist[n - 1] - hist[n - 101]) / hist[n - 101] < 0.02)
      return narr.set(
        `Epoch ${epoch} · plateau`,
        'Stuck: the loss has stopped improving',
        Hn <= 2
          ? `With only ${Hn} neurons per layer, the network can only bend its boundary a little. It is <b>too simple</b> for this data (underfitting). Switch to 8 neurons.`
          : 'It might be stuck in a flat region. Try a bigger learning rate or more neurons, then Reset.',
      );
    if (epoch <= 10)
      return narr.set(
        `Epoch ${epoch} · first steps`,
        'The biggest changes happen first',
        `Nearly everything is wrong at the start, so the gradients are large and every weight gets a big nudge. Each epoch: <b>predict</b> all ${data.length} points, <b>measure</b> the loss, <b>backprop</b>, then <b>nudge</b> every weight.`,
      );
    const back = Math.max(0, n - 25);
    narr.set(
      `Epoch ${epoch} · learning`,
      hist[back] - last.loss > 0.02 ? 'Learning: the loss is falling' : 'Fine-tuning: slow, steady progress',
      `Loss went from ${hist[back].toFixed(3)} to ${last.loss.toFixed(3)} over the last ${n - back} epochs. ${wrongN} point${wrongN === 1 ? '' : 's'} still on the wrong side. Watch the boundary bend around the red-ringed ones.`,
    );
  }

  function draw() {
    const ctx = map.ctx;
    const W = map.width;
    const H = map.height;
    const s0 = hexToRgb(C.surface);
    const c0 = hexToRgb(C.teal);
    const c1 = hexToRgb(C.amber);
    for (let j = 0; j < G; j++)
      for (let i = 0; i < G; i++) {
        const x = -DOM + ((i + 0.5) / G) * 2 * DOM;
        const y = DOM - ((j + 0.5) / G) * 2 * DOM;
        const p = predict(net, x, y);
        pv[j * G + i] = p - 0.5;
        const tgt = p > 0.5 ? c1 : c0;
        const a = Math.min(1, Math.abs(p - 0.5) * 2) * 0.55;
        const k = (j * G + i) * 4;
        for (let c = 0; c < 3; c++) img.data[k + c] = s0[c] + (tgt[c] - s0[c]) * a;
        img.data[k + 3] = 255;
      }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, W, H);
    // Decision boundary via marching squares on p = 0.5.
    const gx = (i: number) => ((i + 0.5) / G) * W;
    const gy = (j: number) => ((j + 0.5) / G) * H;
    const v = (i: number, j: number) => pv[j * G + i];
    ctx.beginPath();
    for (let j = 0; j < G - 1; j++)
      for (let i = 0; i < G - 1; i++) {
        const A = v(i, j);
        const B = v(i + 1, j);
        const Cc = v(i + 1, j + 1);
        const D = v(i, j + 1);
        const pts: [number, number][] = [];
        if (A > 0 !== B > 0) pts.push([gx(i + A / (A - B)), gy(j)]);
        if (B > 0 !== Cc > 0) pts.push([gx(i + 1), gy(j + B / (B - Cc))]);
        if (D > 0 !== Cc > 0) pts.push([gx(i + D / (D - Cc)), gy(j + 1)]);
        if (A > 0 !== D > 0) pts.push([gx(i), gy(j + A / (A - D))]);
        if (pts.length >= 2) {
          ctx.moveTo(...pts[0]);
          ctx.lineTo(...pts[1]);
        }
        if (pts.length === 4) {
          ctx.moveTo(...pts[2]);
          ctx.lineTo(...pts[3]);
        }
      }
    ctx.strokeStyle = C.text;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    const px = (x: number) => ((x + DOM) / (2 * DOM)) * W;
    const py = (y: number) => (1 - (y + DOM) / (2 * DOM)) * H;
    for (const [x, y, c] of data) {
      ctx.beginPath();
      ctx.arc(px(x), py(y), 4.2, 0, 7);
      ctx.fillStyle = c ? C.amber : C.teal;
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = C.text;
      ctx.stroke();
    }
    for (const [x, y] of last.wrong) {
      ctx.beginPath();
      ctx.arc(px(x), py(y), 8, 0, 7);
      ctx.lineWidth = 2;
      ctx.strokeStyle = C.bad;
      ctx.stroke();
    }
    // Loss chart.
    const c = chart.ctx;
    const pad = { l: 40, r: 10, t: 10, b: 22 };
    const xMax = Math.max(100, Math.ceil(epoch / 100) * 100);
    const yMax = Math.max(0.8, Math.ceil((hist[0] || 0.8) * 10) / 10);
    const yMap = (q: number) => pad.t + (1 - q / yMax) * (chart.height - pad.t - pad.b);
    chartFrame(c, {
      width: chart.width,
      height: chart.height,
      pad,
      yTicks: [0, yMax / 2, yMax],
      yFmt: (q) => q.toFixed(2),
      yMap,
      xMax,
      colors: { background: C.sunk, grid: C.grid, text: C.textMuted, font: C.fontMono },
    });
    if (hist.length > 1) {
      c.beginPath();
      const stride = Math.max(1, Math.floor(hist.length / 400));
      for (let i = 0; i < hist.length; i += stride) {
        const X = pad.l + ((i + 1) / xMax) * (chart.width - pad.l - pad.r);
        const Yv = yMap(Math.min(hist[i], yMax));
        if (i) c.lineTo(X, Yv);
        else c.moveTo(X, Yv);
      }
      c.strokeStyle = C.plum;
      c.lineWidth = 2;
      c.stroke();
    }
    readout.innerHTML = `<span>epoch ${epoch}</span><span>loss ${last.loss.toFixed(3)}</span><span>accuracy ${(last.accuracy * 100).toFixed(0)}%</span><span>${last.wrong.length} wrong</span>`;
    phase();
  }

  function resize() {
    map = fitCanvas(cv, 1, 520);
    chart = fitCanvas(lc, 1 / 0.45, 220);
    draw();
  }

  const { loop, cleanup } = wireRunControls(root, {
    rates: cfg.rates,
    maxPerFrame: 30,
    step: epochStep,
    render: () => draw(),
    onReset: () => {
      build();
      draw();
    },
  });
  const rebuild = () => {
    loop.pause();
    build();
    draw();
  };
  dsSel.addEventListener('change', () => {
    ds = dsSel.value as DatasetName;
    rebuild();
  });
  hSel.addEventListener('change', () => {
    Hn = Number(hSel.value);
    rebuild();
  });
  actSel.addEventListener('change', () => {
    act = actSel.value as Activation;
    rebuild();
  });
  lr = bindLogSlider(lrInput, lrOut, {
    min: cfg.lrRange[0],
    max: cfg.lrRange[1],
    steps: cfg.sliderSteps,
    position: cfg.defaultSliderPos,
    display: (v) => v.toFixed(4),
    onInput: (v) => (lr = v),
  }).value;

  build();
  const stopResize = onResize(cv.parentElement!, resize);
  resize();
  const stopTheme = onThemeChange(() => {
    C = readColors();
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
