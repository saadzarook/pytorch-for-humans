/**
 * Shared control wiring, so sims don't each re-implement preset buttons and
 * log-scale sliders.
 */
import { logSlider } from './format';

export interface PresetLike {
  label: string;
}

export interface PresetBinding<P> {
  /** Replace the preset buttons (e.g. when the sim switches curves). */
  setPresets(list: P[]): void;
  /** Un-press every preset (e.g. after the reader moves a slider by hand). */
  clearPressed(): void;
}

/**
 * Render one toggle button per preset into `group` (an element with
 * role="group" and an aria-label) and call `onPick` when one is pressed.
 * The pressed preset gets aria-pressed="true"; the others are cleared.
 */
export function bindPresets<P extends PresetLike>(
  group: HTMLElement,
  presets: P[],
  onPick: (preset: P) => void,
): PresetBinding<P> {
  let current = presets;
  const buttons = () => [...group.querySelectorAll<HTMLButtonElement>('button[data-preset]')];
  const render = () =>
    group.replaceChildren(
      ...current.map((p, i) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'pfh-btn';
        b.dataset.preset = String(i);
        b.setAttribute('aria-pressed', 'false');
        b.textContent = p.label;
        return b;
      }),
    );
  group.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest<HTMLButtonElement>('button[data-preset]');
    if (!btn || !group.contains(btn)) return;
    buttons().forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
    onPick(current[Number(btn.dataset.preset)]);
  });
  render();
  return {
    setPresets(list) {
      current = list;
      render();
    },
    clearPressed() {
      buttons().forEach((b) => b.setAttribute('aria-pressed', 'false'));
    },
  };
}

export interface LogSliderOptions {
  min: number;
  max: number;
  /** Slider positions (0..steps). */
  steps: number;
  /** Starting value, or a starting slider position (one of the two). */
  value?: number;
  position?: number;
  /** Round a dragged value (e.g. to 2 significant figures). Not applied to set(). */
  snap?: (v: number) => number;
  /** What the readout shows. May return HTML (e.g. a warning badge). Default: String(v). */
  display?: (v: number, position: number) => string;
  /** Called when the reader drags the slider. */
  onInput: (v: number, position: number) => void;
}

export interface LogSliderBinding {
  /** Current value. */
  readonly value: number;
  /** Move the slider to `v` and update the readout, without calling onInput. */
  set(v: number): void;
}

/**
 * Wire an <input type="range"> as a log-scale slider (learning rates span orders
 * of magnitude) with a readout element next to it.
 */
export function bindLogSlider(input: HTMLInputElement, readout: HTMLElement, o: LogSliderOptions): LogSliderBinding {
  const scale = logSlider(o.min, o.max, o.steps);
  const display = o.display ?? ((v: number) => String(v));
  input.min = '0';
  input.max = String(o.steps);
  input.step = '1';
  let value = 0;
  const show = (v: number, pos: number) => {
    value = v;
    input.value = String(pos);
    readout.innerHTML = display(v, pos);
  };
  if (o.position !== undefined) {
    const v = (o.snap ?? ((x: number) => x))(scale.toValue(o.position));
    show(v, o.position);
  } else {
    const v = o.value ?? o.min;
    show(v, scale.toPos(v));
  }
  input.addEventListener('input', () => {
    const pos = Number(input.value);
    const v = (o.snap ?? ((x: number) => x))(scale.toValue(pos));
    value = v;
    readout.innerHTML = display(v, pos);
    o.onInput(v, pos);
  });
  return {
    get value() {
      return value;
    },
    set(v: number) {
      show(v, scale.toPos(v));
    },
  };
}
