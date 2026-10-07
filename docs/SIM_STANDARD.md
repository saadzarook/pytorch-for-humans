# The course sim standard

Every interactive simulation in PyTorch for Humans follows these rules. They come from user
testing: people couldn't follow the first version of the sims. Narration, step-through and
on-canvas labels fixed that.

The shared kit in [`site/src/sims/kit/`](../site/src/sims/kit/) does most of the work. If you use
it, rules 2 and 7 come almost for free.

## The rules

1. **Narrate every phase.** A caption (the kit's `Narrator`) always says what is happening *right
   now* and why it matters, in plain, casual language. It updates on every step or phase change.
2. **Slow by default, always steppable.** Nothing autoplays. Staged sims get **Back / Next**
   (`Stepper`). Continuous sims get **Play / Step** and a **Slow / Normal / Fast** control
   (`RunLoop`), and start on Slow.
3. **Show the numbers behind the picture.** Arithmetic strips, gradient values, loss charts. The
   picture builds intuition, and the numbers make it trustworthy.
4. **Label things on the canvas.** Arrows, reference lines, the decision line, rings around
   mistakes. Put labels where the eye already is (kit: `label()`, `arrow()`), not only in a legend.
5. **Log key moments** with their step or epoch number (`EventLog`): overshoots, the loss going
   up, zig-zags, divergence, accuracy milestones. Learners can look back after missing something.
6. **Change one thing at a time.** Highlight the one active element (node, window, phase) and keep
   everything else muted.
7. **Be accessible.**
   - The narrator is an `aria-live="polite"` region, so screen readers hear every phase.
   - Keyboard: <kbd>Space</kbd> play/pause (or auto-play), <kbd>→</kbd> step/next, <kbd>←</kbd>
     back, active while the sim has focus. Show the shortcuts in a small hint (`SimShell` does this).
   - `prefers-reduced-motion`: no tweening or CSS transitions; things jump to their new state.
   - Works at **375 px** wide with no horizontal page scroll, and in **light and dark** mode. Read
     colours from the `--sim-*` tokens, and re-render when the theme changes (`onThemeChange`).
   - Colour is never the only cue: label series directly on the canvas. Validate any new categorical
     palette with the data-viz palette checker (CVD ΔE ≥ 8).

## The correctness rule

Sims re-implement maths in TypeScript, so **every sim's maths is checked against real PyTorch**:

- Put the maths in a pure `math.ts` (no DOM), and the setup in a `config.json` that both the TS and
  the Python reference read.
- `golden.py` uses real PyTorch (via [`scripts/simref.py`](../scripts/simref.py)) to write
  `golden.json`. A Vitest test (`<sim>.test.ts`) checks the TS matches it within 1e-6 (relative).
  For iterative maths (optimizers), check **each step from PyTorch's exact state**, not just whole
  trajectories: chaotic or diverging runs amplify last-bit float differences ~10× per step.
- **Narration never hand-types a computed number, and every claim it makes is declared.**
  `facts.py` only *computes* numbers and evidence, writing `facts.json`; the sim renders
  numbers from it. Every claim the narration makes ("Momentum arrives first", "the middle of the
  crossbar stays 0") is declared in the sim's `config.json` under `"claims"`:

  ```json
  { "id": "arrival-order",
    "says": "at lr 0.1: Momentum, then Adam, then SGD",
    "check": "order == ['Momentum', 'Adam', 'SGD']" }
  ```

  `check` is a small Python expression over `facts.json` (its top-level keys are variables; only
  a few safe built-ins like `all`, `abs`, `zip` are available). The facts runner evaluates every
  claim after every `facts.py` run; a false claim fails CI and quotes the narration it would make
  wrong. A sim with a `facts.py` but no declared claims also fails. Lessons do the same in their
  `data.json`. A Vitest test checks the live JS logic reproduces the facts.
- `python scripts/test_snippets.py` regenerates-and-compares all of these (CI fails if any is
  stale); `npm test` runs the Vitest checks.

## Building a sim

```
site/src/sims/<name>/
  config.json        setup shared by TS and Python
  math.ts            pure maths, unit-tested
  mount.ts           export const mount: MountFn = (el, options) => { …; return { destroy() } }
  <Name>.astro       markup inside <SimShell>, elements tagged data-part="…"
  golden.py / .json  PyTorch reference values (generated)
  facts.py / .json   narration numbers + evidence (generated); claims live in config.json
  <name>.test.ts     Vitest: math vs golden.json, live logic vs facts.json
```

Register it in `kit/island.ts`. `<SimShell>` renders a lazy `<pfh-sim>` island: the sim's code
downloads only when it scrolls near the viewport (the framework-free equivalent of
`client:visible`), and `destroy()` runs when it leaves the page. Keep each lesson page under about
**100 KB** of JavaScript.

| Kit module | What it gives you |
| --- | --- |
| `SimShell.astro` | Card, title, narrator box, keyboard hint, lazy island, and the `layout`: `"stack"` (default), `"two-column"` (`slot="stage"` \| `slot="side"`), `"stage-chart"` (default slot, then `slot="chart"` \| `slot="side"`). Inside a stage, `.sim-panels` / `.sim-trio` lay out several canvases. Sims write no grid CSS of their own |
| `controls.ts` | `bindPresets(group, presets, onPick)` (preset toggle buttons) and `bindLogSlider(input, readout, {min, max, steps, …})` (log-scale learning-rate sliders) |
| `RunControls.astro` + `dom.ts › wireRunControls` | Play/Pause, Step, Reset, Slow·Normal·Fast, keys |
| `StepControls.astro` + `stepper.ts` | Back / Next / Auto-play, "step X of N", keys |
| `LogPanel.astro` + `eventLog.ts` | Key moments, newest first, max 5, empty-state hint |
| `narrator.ts` | Phase badge + title + body, aria-live, de-duplicated |
| `runLoop.ts` | Fixed steps/second (rAF + accumulator), pauses off-screen, reduced-motion aware |
| `theme.ts` | `--sim-*` tokens, `onThemeChange` (data-theme + OS scheme) |
| `canvas.ts` | HiDPI sizing, `arrow`, `label`, `chartFrame`, `dot`, resize observer |
| `random.ts`, `format.ts` | Seeded RNG, Gaussian samples, log sliders, number formatting |
| `lossChart.ts` | Single-series loss-vs-step chart with hover readout |
| `testing.ts` | The 1e-6 relative tolerance helpers for golden tests |

## Pre-merge checklist (copy into your PR)

- [ ] Narrator explains every phase (read it aloud: does it say *what* and *why*?)
- [ ] Nothing autoplays; Step works; staged sims have Back / Next; continuous sims start on Slow
- [ ] The numbers behind the picture are visible
- [ ] Key things are labelled on the canvas, not only in a legend
- [ ] Key moments are logged with step/epoch numbers
- [ ] Only the active element is highlighted
- [ ] Keyboard: Space / → / ← work when the sim is focused; the hint lists them
- [ ] Reduced motion: no tweening or transitions
- [ ] Checked at 375 px and in light + dark mode
- [ ] `golden.py` + Vitest: maths matches PyTorch within 1e-6
- [ ] Every number in the narration comes from `facts.json`, and every claim is declared in `config.json`
- [ ] The narration snapshot (`site/tests/e2e/narration/`) is updated on purpose, and the diff reviewed
- [ ] `python scripts/test_snippets.py`, `npm test` and `npm run test:e2e` pass; page JS under ~100 KB

## Testing

- `npm test` (Vitest): the sim's maths against `golden.json` (PyTorch), and the live logic
  against `facts.json`.
- `npm run test:e2e` (Playwright, after `npm run build`): behaviour of every sim in a real browser,
  plus **narration snapshots**: the exact narrator text for every preset, so refactors can't change
  what learners read by accident. Change narration on purpose with
  `UPDATE_NARRATION=1 npm run test:e2e -- narration`, then review the JSON diff.
- Tests find elements by **role and accessible name** only (never position or CSS class). If a
  test can't find a control that way, a screen-reader user probably can't either.

## Future

Ideas we've decided to postpone (not required yet):

- **Collision-avoiding label layout.** Canvas labels are placed by hand per sim, and overlaps were
  fixed case by case (e.g. moving the ball's slope label next to the ball). A shared
  `labelLayout()` pass that nudges labels apart would scale better as sims multiply.
- **Back for continuous sims.** Only staged sims (Back/Next) can step backwards today. Snapshotting
  the sim state every step would let ← undo a step in the ball, contour and optimizer-race sims too.
