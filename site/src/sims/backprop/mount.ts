/**
 * Backprop as a chain of blame: a one-neuron model learns to output 7 when it
 * sees 2. Next walks through one training round: forward, backward, update.
 * Ported from reference/ml-sim-lab.html; numbers checked against torch autograd.
 */
import cfg from './config.json';
import { computeRound, LR, START, X, Y, type Round } from './math';
import { bindKeys } from '../kit/keyboard';
import { Narrator } from '../kit/narrator';
import { Stepper } from '../kit/stepper';
import { f2 } from '../kit/format';
import { part, type MountFn } from '../kit/types';

const NS = 'http://www.w3.org/2000/svg';

interface GNode {
  x: number;
  y: number;
  lbl: string;
  leaf?: boolean;
  /** Name of the value this node computes (z, p, e, L). */
  v?: string;
  g?: SVGGElement;
  val?: SVGTextElement;
  grd?: SVGTextElement;
}

interface Step {
  ph: string;
  t: string;
  b: string;
  vals?: string[];
  grads?: string[];
  fe?: string[];
  be?: string[];
  focus?: string[];
  nograd?: boolean;
  code?: number[];
}

export const mount: MountFn = (root) => {
  const svg = part<SVGSVGElement>(root, 'graph');
  const narr = new Narrator(part(root, 'narrator'));
  const codeLines = [...part(root, 'code').children] as HTMLElement[];
  const readout = part(root, 'readout');

  const N: Record<string, GNode> = {
    x: { x: 70, y: 62, leaf: true, lbl: 'x' },
    w: { x: 70, y: 178, leaf: true, lbl: 'w' },
    mul: { x: 220, y: 120, lbl: '×', v: 'z' },
    b: { x: 220, y: 250, leaf: true, lbl: 'b' },
    add: { x: 370, y: 182, lbl: '+', v: 'p' },
    y: { x: 370, y: 300, leaf: true, lbl: 'y' },
    sub: { x: 520, y: 240, lbl: '−', v: 'e' },
    sq: { x: 665, y: 240, lbl: '²', v: 'L' },
  };
  const E: [string, string][] = [['x', 'mul'], ['w', 'mul'], ['mul', 'add'], ['b', 'add'], ['add', 'sub'], ['y', 'sub'], ['sub', 'sq']];
  const el = <K extends keyof SVGElementTagNameMap>(tag: K, at: Record<string, string | number>, parent: Element = svg) => {
    const e = document.createElementNS(NS, tag);
    for (const k in at) e.setAttribute(k, String(at[k]));
    parent.appendChild(e);
    return e;
  };
  svg.replaceChildren();
  const edgeEl: Record<string, SVGLineElement> = {};
  for (const [s, t] of E) edgeEl[s + t] = el('line', { class: 'edge', x1: N[s].x, y1: N[s].y, x2: N[t].x, y2: N[t].y });
  for (const k in N) {
    const n = N[k];
    const g = el('g', {});
    n.g = g;
    if (n.leaf) el('rect', { class: 'shape', x: n.x - 34, y: n.y - 20, width: 68, height: 40, rx: 6 }, g);
    else el('circle', { class: 'shape', cx: n.x, cy: n.y, r: 28 }, g);
    el('text', { class: 'lbl', x: n.x, y: n.y }, g).textContent = n.lbl;
    n.val = el('text', { class: 'val', x: n.x, y: n.y + (n.leaf ? 40 : 48) }, g);
    n.grd = el('text', { class: 'grd', x: n.x, y: n.y + (n.leaf ? 60 : 68) }, g);
  }
  const dirF = el('text', { class: 'dir', x: 20, y: 372 });

  let w = START.w;
  let b = START.b;
  let round = 1;
  let idx = 0;
  let steps: Step[] = [];
  let R: Round;
  let V: Record<string, number> = {};
  let G: Record<string, number> = {};
  let losses: number[] = [];
  let note = '';

  function build() {
    R = computeRound(w, b);
    const { z, p, e, L, g, gw, w2, b2 } = R;
    V = { x: X, w, b, y: Y, mul: z, add: p, sub: e, sq: L };
    G = { sq: 1, sub: g, add: g, mul: g, b: g, w: gw };
    const setupBody =
      round === 1
        ? `We want <code>w·x + b</code> to output <b>${Y}</b> when x = <b>${X}</b>. Right now w = ${f2(w)} and b = ${f2(b)}. <b>x</b> and <b>y</b> are data and never change. <b>w</b> and <b>b</b> are the knobs training is allowed to turn (they have <code>requires_grad=True</code>).`
        : `${note} Last round's loss was <b>${f2(losses[losses.length - 1])}</b>. The knobs are now w = ${f2(w)}, b = ${f2(b)}. Let's see whether the loss went down.`;
    steps = [
      { ph: `Round ${round} · setup`, t: round === 1 ? 'Meet the model' : 'New round, updated knobs', b: setupBody, vals: ['x', 'w', 'b', 'y'], focus: ['w', 'b'], code: round === 1 ? [] : [4] },
      { ph: 'Forward 1 of 4', t: 'Multiply', b: `The first node multiplies its two inputs: z = w × x = ${f2(w)} × ${X} = <b>${f2(z)}</b>. Values flow left to right (amber).`, vals: ['mul'], fe: ['xmul', 'wmul'], focus: ['mul'], code: [0] },
      { ph: 'Forward 2 of 4', t: 'Add the bias', b: `p = z + b = ${f2(z)} + ${f2(b)} = <b>${f2(p)}</b>. This is the model's <b>prediction</b>. The right answer is ${Y}.`, vals: ['add'], fe: ['muladd', 'badd'], focus: ['add'], code: [0] },
      { ph: 'Forward 3 of 4', t: 'How wrong are we?', b: `e = p − y = ${f2(p)} − ${Y} = <b>${f2(e)}</b>. ${e < 0 ? 'Negative means the prediction is too <b>low</b>.' : 'Positive means the prediction is too <b>high</b>.'}`, vals: ['sub'], fe: ['addsub', 'ysub'], focus: ['sub'], code: [0] },
      { ph: 'Forward 4 of 4', t: 'Turn the error into a loss', b: `L = e² = (${f2(e)})² = <b>${f2(L)}</b>. Squaring makes the error positive and punishes big misses much more than small ones. Training has one job: make this number small.`, vals: ['sq'], fe: ['subsq'], focus: ['sq'], code: [0] },
      { ph: 'Backward 1 of 5', t: 'Start at the end', b: 'Now we go right to left (plum, dashed). Each node answers one question: <i>"if my value wiggled up by a tiny bit, how much would L change?"</i> That is its gradient. L\'s answer about itself is simply <b>1</b>.', grads: ['sq'], focus: ['sq'], code: [1] },
      { ph: 'Backward 2 of 5', t: 'Square rule', b: `For L = e², the slope is 2e. So ∂L/∂e = 2 × ${f2(e)} = <b>${f2(g)}</b>. ${g < 0 ? 'Negative means that <i>increasing</i> e would <i>decrease</i> the loss. Makes sense, since e is too low.' : 'Positive means that decreasing e would decrease the loss.'}`, grads: ['sub'], be: ['subsq'], focus: ['sub'], code: [1] },
      { ph: 'Backward 3 of 5', t: 'Subtraction passes it straight through', b: `e = p − y, so raising p by 1 raises e by exactly 1. The gradient passes through unchanged: ∂L/∂p = <b>${f2(g)}</b>.`, grads: ['add'], be: ['addsub'], focus: ['add'], code: [1] },
      { ph: 'Backward 4 of 5', t: 'Addition copies it to both inputs', b: `p = z + b, and both inputs count equally, so both receive the same gradient: ∂L/∂z = ∂L/∂b = <b>${f2(g)}</b>. That's <code>b.grad</code> done.`, grads: ['mul', 'b'], be: ['muladd', 'badd'], focus: ['mul', 'b'], code: [1] },
      { ph: 'Backward 5 of 5', t: 'Multiplication uses the other input', b: `z = w × x, so raising w by 1 raises z by x (that is, ${X}). Chain rule: ∂L/∂w = ∂L/∂z × x = ${f2(g)} × ${X} = <b>${f2(gw)}</b>. That's <code>w.grad</code>. x and y are data, so they get no gradient.`, grads: ['w'], be: ['wmul'], focus: ['w'], nograd: true, code: [1] },
      { ph: `Round ${round} · update`, t: 'Nudge each knob against its gradient', b: `w ← w − lr × ∂L/∂w = ${f2(w)} − ${LR} × (${f2(gw)}) = <b>${f2(w2)}</b><br>b ← b − lr × ∂L/∂b = ${f2(b)} − ${LR} × (${f2(g)}) = <b>${f2(b2)}</b><br>${gw < 0 ? `The gradients are negative, so subtracting them <i>raises</i> w and b, which pushes the prediction up toward ${Y}.` : `The gradients are positive, so subtracting them lowers w and b, which pulls the prediction down toward ${Y}.`} Press Next to apply this and start round ${round + 1}.`, focus: ['w', 'b'], code: [2, 3] },
    ];
  }

  function render() {
    const s = steps[idx];
    const last = steps.length - 1;
    for (const k in N) {
      N[k].val!.textContent = '';
      N[k].grd!.textContent = '';
      N[k].grd!.classList.remove('nograd');
      N[k].g!.classList.remove('focus');
    }
    for (const k in edgeEl) edgeEl[k].setAttribute('class', 'edge');
    const backward = idx >= 5;
    for (let i = 0; i <= idx; i++) {
      const st = steps[i];
      (st.vals ?? []).forEach((k) => (N[k].val!.textContent = (N[k].v ? `${N[k].v} = ` : '= ') + f2(V[k])));
      (st.grads ?? []).forEach((k) => (N[k].grd!.textContent = `∂L/∂${N[k].v ?? N[k].lbl} = ${f2(G[k])}`));
      if (!backward) (st.fe ?? []).forEach((e) => edgeEl[e].setAttribute('class', 'edge fwd'));
      else (st.be ?? []).forEach((e) => edgeEl[e].setAttribute('class', 'edge bwd'));
      if (st.nograd || idx === last)
        ['x', 'y'].forEach((k) => {
          N[k].grd!.textContent = 'data, no grad';
          N[k].grd!.classList.add('nograd');
        });
    }
    if (idx === last) ['sq', 'sub', 'add', 'mul', 'b', 'w'].forEach((k) => (N[k].grd!.textContent = `∂L/∂${N[k].v ?? N[k].lbl} = ${f2(G[k])}`));
    (s.focus ?? []).forEach((k) => N[k].g!.classList.add('focus'));
    dirF.textContent = idx === 0 ? '' : backward ? '◀ gradients flow right to left' : 'values flow left to right ▶';
    codeLines.forEach((d, j) => d.classList.toggle('hl', (s.code ?? []).includes(j)));
    narr.set(s.ph, s.t, s.b);
    const hist = losses.slice(-7).map(f2);
    readout.innerHTML =
      `<span>round ${round}</span><span>w = ${f2(w)}, b = ${f2(b)}</span>` +
      (hist.length ? `<span>loss by round: ${hist.join(' → ')}${idx >= 4 ? ` → ${f2(R.L)}` : ''}</span>` : '');
  }

  function next() {
    if (idx < steps.length - 1) {
      idx++;
      return render();
    }
    losses.push(R.L);
    [w, b] = [R.w2, R.b2];
    round++;
    idx = 0;
    note = "Update applied, and both gradients were reset to zero (otherwise PyTorch would add the next round's gradients on top).";
    build();
    render();
  }

  const stepper = new Stepper({
    root: part(root, 'stepper'),
    next,
    back: () => {
      idx--;
      render();
    },
    canBack: () => idx > 0,
    progress: () => `step ${idx + 1} of ${steps.length}`,
    nextLabel: () => (idx === steps.length - 1 ? 'Apply update' : 'Next ▶︎'),
    autoMs: cfg.autoMs,
  });

  part(root, 'skip').addEventListener('click', () => {
    stepper.stopAuto();
    const from = computeRound(w, b).L;
    for (let i = 0; i < cfg.skipRounds; i++) {
      build();
      losses.push(R.L);
      [w, b] = [R.w2, R.b2];
      round++;
    }
    note = `Skipped ahead ${cfg.skipRounds} rounds of forward, backward and update. Loss went from ${f2(from)} to ${f2(losses[losses.length - 1])}, and it keeps shrinking as the prediction closes in on ${Y}.`;
    idx = 0;
    build();
    render();
    stepper.sync();
  });
  part(root, 'reset').addEventListener('click', () => {
    stepper.stopAuto();
    w = START.w;
    b = START.b;
    round = 1;
    idx = 0;
    losses = [];
    note = '';
    build();
    render();
    stepper.sync();
  });
  const unbindKeys = bindKeys(root, {
    toggle: () => stepper.toggleAuto(),
    next: () => stepper.next(),
    back: () => stepper.back(),
  });

  build();
  render();
  stepper.sync();
  return {
    destroy() {
      stepper.destroy();
      unbindKeys();
    },
  };
};
