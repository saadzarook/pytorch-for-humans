/**
 * Convolution, one window at a time: a 3×3 kernel slides over a 12×12 "A".
 * In Slow mode each window plays in three beats (place, multiply, add up).
 * Ported from reference/ml-sim-lab.html; outputs checked against F.conv2d.
 *
 * Added for the course standard: a Back button (and ← key) that steps back
 * one beat, since this is a staged sim.
 */
import cfg from './config.json';
import facts from './facts.json';
import { conv2d, IMG, O, PRESETS, S } from './math';
import { fitCanvas, onResize, type Surface } from '../kit/canvas';
import { wireRunControls } from '../kit/dom';
import { bindKeys } from '../kit/keyboard';
import { Narrator } from '../kit/narrator';
import { hexToRgb, onThemeChange, readColors, type SimColors } from '../kit/theme';
import { part, type MountFn } from '../kit/types';

const DESC: Record<string, string> = {
  vedge:
    'Looks for <b>vertical edges</b>: the left column is negative and the right is positive, so it fires when ink sits on the right side of the window and not the left.',
  hedge:
    'Looks for <b>horizontal edges</b>: the top row is negative and the bottom is positive, so it fires when ink sits below and not above.',
  outline:
    'Fires when the <b>centre differs from its 8 neighbours</b>. In a flat area (all ink or all blank) the weights cancel out to 0.',
  blur: '<b>Averages</b> the 9 pixels: each one counts 1/9. Everything gets softer.',
  sharpen: 'Boosts the centre ×5 and subtracts its 4 neighbours, which <b>exaggerates differences</b>.',
  identity: 'Copies just the centre pixel, so the output is the input with its border trimmed off.',
  custom: 'Your own kernel. A positive weight rewards ink in that spot and a negative weight punishes it.',
};

export const mount: MountFn = (root) => {
  const narr = new Narrator(part(root, 'narrator'));
  const cin = part<HTMLCanvasElement>(root, 'input');
  const cout = part<HTMLCanvasElement>(root, 'output');
  const preset = part<HTMLSelectElement>(root, 'preset');
  const kEl = part(root, 'kernel');
  const kdesc = part(root, 'kdesc');
  const sumEl = part(root, 'sum');
  const prodGrid = part(root, 'm-prod');
  const mk = (name: string) => {
    const g = part(root, name);
    g.replaceChildren(...Array.from({ length: 9 }, () => document.createElement('span')));
    return [...g.children] as HTMLElement[];
  };
  const mPatch = mk('m-patch');
  const mK = mk('m-k');
  const mProd = mk('m-prod');

  let K = PRESETS.vedge.slice();
  let kname = 'vedge';
  let out: number[] = [];
  let maxAbs = 1;
  let pos = -1;
  let sub = 2;
  let C: SimColors = readColors();
  let a!: Surface;
  let b!: Surface;

  const inputs = Array.from({ length: 9 }, (_, i) => {
    const inp = document.createElement('input');
    inp.type = 'number';
    inp.step = '0.5';
    inp.setAttribute('aria-label', `kernel row ${Math.floor(i / 3) + 1} column ${(i % 3) + 1}`);
    inp.addEventListener('input', () => {
      const v = parseFloat(inp.value);
      if (!Number.isNaN(v)) {
        K[i] = v;
        kname = 'custom';
        preset.value = 'custom';
        compute();
        draw();
      }
    });
    return inp;
  });
  kEl.replaceChildren(...inputs);

  const fmt = (v: number) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(2));
  function syncInputs() {
    inputs.forEach((inp, i) => {
      if (document.activeElement !== inp) inp.value = fmt(K[i]);
    });
    kdesc.innerHTML = DESC[kname];
  }
  function compute() {
    out = conv2d(IMG, K);
    maxAbs = Math.max(1e-9, ...out.map(Math.abs));
    syncInputs();
  }
  const done = () => pos >= O * O;

  function draw() {
    const active = pos >= 0 && !done();
    const r = Math.floor(pos / O);
    const q = pos % O;
    // Input grid + window.
    const c = a.width / S;
    a.ctx.clearRect(0, 0, a.width, a.height);
    for (let i = 0; i < S; i++)
      for (let j = 0; j < S; j++) {
        a.ctx.fillStyle = IMG[i][j] ? C.text : C.sunk;
        a.ctx.fillRect(j * c + 0.5, i * c + 0.5, c - 1, c - 1);
      }
    if (active) {
      a.ctx.fillStyle = C.amber;
      a.ctx.globalAlpha = 0.28;
      a.ctx.fillRect(q * c, r * c, 3 * c, 3 * c);
      a.ctx.globalAlpha = 1;
      a.ctx.strokeStyle = C.amber;
      a.ctx.lineWidth = 3;
      a.ctx.strokeRect(q * c + 1.5, r * c + 1.5, 3 * c - 3, 3 * c - 3);
    }
    // Output grid, filled in as the window moves.
    const d = b.width / O;
    b.ctx.clearRect(0, 0, b.width, b.height);
    const pc = hexToRgb(C.amber);
    const nc = hexToRgb(C.teal);
    const base = hexToRgb(C.sunk);
    for (let k = 0; k < O * O; k++) {
      const rr = Math.floor(k / O);
      const qq = k % O;
      let col = base;
      const shown = pos < 0 ? false : done() || k < pos || (k === pos && sub === 2);
      if (shown) {
        const v = out[k] / maxAbs;
        const tgt = v >= 0 ? pc : nc;
        const t = Math.min(1, Math.abs(v));
        col = base.map((x, ci) => Math.round(x + (tgt[ci] - x) * t)) as [number, number, number];
      }
      b.ctx.fillStyle = `rgb(${col.join(',')})`;
      b.ctx.fillRect(qq * d + 0.5, rr * d + 0.5, d - 1, d - 1);
    }
    if (active) {
      b.ctx.strokeStyle = C.text;
      b.ctx.lineWidth = 2.5;
      b.ctx.strokeRect(q * d + 1.5, r * d + 1.5, d - 3, d - 3);
    }
    // Arithmetic strip.
    const pr = active ? r : 0;
    const pq = active ? q : 0;
    const terms: string[] = [];
    let lit = 0;
    for (let i = 0; i < 9; i++) {
      const u = Math.floor(i / 3);
      const v = i % 3;
      const pv = active ? IMG[pr + u][pq + v] : 0;
      const kv = K[i];
      const prod = pv * kv;
      mPatch[i].textContent = active ? String(pv) : '·';
      mPatch[i].className = pv ? 'lit' : '';
      mK[i].textContent = fmt(kv);
      mK[i].className = kv > 0 ? 'pos' : kv < 0 ? 'neg' : 'zero';
      mProd[i].textContent = fmt(prod);
      mProd[i].className = prod > 0 ? 'pos' : prod < 0 ? 'neg' : 'zero';
      if (pv) {
        lit++;
        if (kv) terms.push(fmt(kv));
      }
    }
    prodGrid.classList.toggle('hidden', !active || sub < 1);
    sumEl.textContent = active && sub === 2 ? fmt(out[pos]) : '?';
    narrate(active, r, q, terms, lit);
  }

  function narrate(active: boolean, r: number, q: number, terms: string[], lit: number) {
    if (pos < 0)
      return narr.set(
        'Ready',
        'A kernel is a tiny 3×3 pattern detector',
        `It slides across the image one pixel at a time. At every stop it multiplies the 9 pixels under it by its 9 weights and adds them up, and that sum becomes one output pixel. ${DESC[kname]} Press <b>Step</b> to go one beat at a time.`,
      );
    if (done())
      return narr.set(
        `Done · ${facts.windows} windows`,
        'The feature map is complete',
        `A 3×3 window fits in ${facts.imageSize} − 3 + 1 = ${facts.outSize} positions across and ${facts.outSize} down, giving a ${facts.outSize}×${facts.outSize} output. ${
          kname === 'vedge'
            ? 'The slanted sides lit up, amber on one side of each stroke and teal on the other, while the middle of the crossbar stayed 0: this kernel ignores horizontal lines.'
            : kname === 'hedge'
              ? 'The crossbar lit up strongly, amber along its top edge and teal along its bottom edge, while the slanted sides respond only weakly.'
              : 'Amber means a strong positive response and teal a strong negative one.'
        } A CNN <i>learns</i> dozens of kernels like this instead of having them hand-picked. Click any output pixel to inspect it.`,
      );
    if (!active) return;
    const s = out[pos];
    const where = `rows ${r + 1}–${r + 3}, columns ${q + 1}–${q + 3}`;
    const ph = `Window ${pos + 1} of ${facts.windows}`;
    const expr = terms.length ? terms.join(' + ').replace(/\+ -/g, '− ') : '0';
    let meaning: string;
    if (!lit) meaning = 'The patch is empty (all 0s), so every product is 0. Nothing to see here.';
    else if (Math.abs(s) < 1e-9)
      meaning = "There is ink here, but the positive and negative weights <b>cancel out</b> to 0. The kernel doesn't care about this pattern.";
    else if (s > 0)
      meaning = `<b>Positive (${fmt(s)})</b>: this patch looks like what the kernel is searching for, so the output pixel turns amber.`;
    else
      meaning = `<b>Negative (${fmt(s)})</b>: the opposite pattern${
        kname === 'vedge' ? ' (ink on the left, not the right)' : kname === 'hedge' ? ' (ink on top, not below)' : ''
      }, so the output pixel turns teal.`;
    const px = lit === 1 ? '' : 's';
    if (sub === 0)
      narr.set(`${ph} · beat 1: place`, 'Place the window', `The kernel sits over input ${where}. It covers ${lit} inked pixel${px} (shown as 1s in the patch grid below).`);
    else if (sub === 1)
      narr.set(
        `${ph} · beat 2: multiply`,
        'Multiply pixel by weight',
        `Each of the 9 pixels is multiplied by the weight sitting on top of it. Blank pixels are 0, so only the ${lit} inked one${px} can contribute.`,
      );
    else
      narr.set(`${ph} · beat 3: add up`, 'Add the products', `${expr} = <b>${fmt(s)}</b>. That becomes output pixel (row ${r + 1}, col ${q + 1}). ${meaning}`);
  }

  function advance() {
    if (done() || pos < 0) {
      pos = 0;
      sub = 0;
      return;
    }
    if (loop.speed !== 'slow' && loop.running) {
      pos++;
      sub = 2;
      return;
    }
    if (sub < 2) sub++;
    else {
      pos++;
      sub = 0;
    }
  }

  function back() {
    loop.pause();
    if (done()) {
      pos = O * O - 1;
      sub = 2;
    } else if (pos < 0) return;
    else if (sub > 0) sub--;
    else if (pos > 0) {
      pos--;
      sub = 2;
    } else pos = -1;
    draw();
  }

  const { loop, cleanup } = wireRunControls(root, {
    rates: cfg.beatsPerSecond,
    speed: 'slow',
    step: () => {
      advance();
      return !done();
    },
    render: () => draw(),
    onReset: () => {
      pos = -1;
      sub = 2;
      draw();
    },
    beforePlay: () => {
      if (done() || pos < 0) {
        pos = 0;
        sub = loop.speed === 'slow' ? 0 : 2;
      }
    },
  });
  const unbindBack = bindKeys(root, { back });
  part(root, 'back').addEventListener('click', back);
  part(root, 'jump').addEventListener('click', () => {
    loop.pause();
    let p = pos < 0 || done() ? 0 : pos + 1;
    while (p < O * O && Math.abs(out[p]) < 1e-9) p++;
    pos = p >= O * O ? O * O : p;
    sub = 2;
    draw();
  });
  cout.addEventListener('click', (e) => {
    const r = cout.getBoundingClientRect();
    const q = Math.floor(((e.clientX - r.left) / r.width) * O);
    const rr = Math.floor(((e.clientY - r.top) / r.height) * O);
    if (q >= 0 && q < O && rr >= 0 && rr < O) {
      loop.pause();
      pos = rr * O + q;
      sub = 2;
      draw();
    }
  });
  preset.addEventListener('change', () => {
    kname = preset.value;
    if (PRESETS[kname]) K = PRESETS[kname].slice();
    compute();
    draw();
  });

  function resize() {
    a = fitCanvas(cin, 1, 280);
    b = fitCanvas(cout, 1, 280);
    draw();
  }
  compute();
  const stopResize = onResize(cin.parentElement!, resize);
  resize();
  const stopTheme = onThemeChange(() => {
    C = readColors();
    draw();
  });

  return {
    destroy() {
      cleanup();
      unbindBack();
      stopResize();
      stopTheme();
    },
  };
};
