import { prefersReducedMotion } from './theme';

export type Speed = 'slow' | 'normal' | 'fast';

export interface RunLoopOptions {
  /** Element whose visibility pauses the loop when it's scrolled off-screen. */
  root: Element;
  /** Steps per second at each speed. */
  rates: Record<Speed, number>;
  /** Starting speed. Course standard: slow. */
  speed?: Speed;
  /** Cap on catch-up steps per frame (e.g. after a slow frame). */
  maxPerFrame?: number;
  /** Advance one step. Return false to stop playing (finished, exploded…). */
  step: () => boolean;
  /**
   * Draw. `t` (0→1) is the progress of the tween since the last step, for
   * smooth motion; always 1 with prefers-reduced-motion or tweenMs = 0.
   */
  render: (t: number) => void;
  /** Length of the between-step tween, ms (0 = none). Capped at 85% of the step interval. */
  tweenMs?: number;
  onRunningChange?: (running: boolean) => void;
}

/**
 * Play / Pause / Step at a fixed number of steps per second.
 *
 * Timing uses requestAnimationFrame plus an accumulator, so the step rate is
 * the same on 60 Hz and 120 Hz screens. Off-screen, the loop stops scheduling
 * frames (and resumes when it's visible again) without changing play state.
 * Nothing ever starts on its own: playback is always reader-initiated.
 */
export class RunLoop {
  running = false;
  speed: Speed;
  private readonly o: RunLoopOptions;
  private raf = 0;
  private acc = 0;
  private prevT = 0;
  private lastStepAt = -Infinity;
  private visible = true;
  private readonly io: IntersectionObserver;

  constructor(opts: RunLoopOptions) {
    this.o = opts;
    this.speed = opts.speed ?? 'slow';
    this.io = new IntersectionObserver(
      (entries) => {
        this.visible = entries.some((e) => e.isIntersecting);
        if (this.visible && this.running) this.kick();
      },
      { threshold: 0.05 },
    );
    this.io.observe(opts.root);
  }

  private get tweenMs(): number {
    if (prefersReducedMotion()) return 0;
    return Math.min(this.o.tweenMs ?? 0, (1000 / this.o.rates[this.speed]) * 0.85);
  }

  play(): void {
    if (this.running) return;
    this.running = true;
    this.acc = 1; // first step happens immediately
    this.o.onRunningChange?.(true);
    this.kick();
  }

  pause(): void {
    if (!this.running) return;
    this.running = false;
    this.o.onRunningChange?.(false);
  }

  toggle(): void {
    if (this.running) this.pause();
    else this.play();
  }

  /** Pause, then take exactly one step. Returns the step callback's result. */
  stepOnce(): boolean {
    this.pause();
    const more = this.o.step();
    this.lastStepAt = performance.now();
    // Draw right away (so the narrator/readout update synchronously for screen
    // readers), then let the next frames play the tween, if any.
    this.o.render(this.tweenMs === 0 ? 1 : 0);
    this.kick();
    return more;
  }

  setSpeed(speed: Speed): void {
    this.speed = speed;
  }

  /** Redraw now, without a tween (after reset, resize, theme change…). */
  redraw(): void {
    this.o.render(1);
  }

  destroy(): void {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.io.disconnect();
  }

  private kick(): void {
    cancelAnimationFrame(this.raf);
    this.prevT = performance.now();
    this.raf = requestAnimationFrame(this.frame);
  }

  private frame = (now: number): void => {
    if (this.running && this.visible) {
      this.acc += ((now - this.prevT) / 1000) * this.o.rates[this.speed];
      let n = Math.min(Math.floor(this.acc), this.o.maxPerFrame ?? 8);
      this.acc -= n;
      while (n-- > 0 && this.running) {
        if (!this.o.step()) this.pause();
        this.lastStepAt = now;
      }
    }
    this.prevT = now;
    const tween = this.tweenMs;
    const t = tween === 0 ? 1 : Math.min(1, (now - this.lastStepAt) / tween);
    this.o.render(1 - (1 - t) * (1 - t));
    if ((this.running && this.visible) || t < 1) this.raf = requestAnimationFrame(this.frame);
  };
}

/** Wire a Slow / Normal / Fast segmented control (buttons with data-speed) to a RunLoop. */
export function bindSpeedControl(group: HTMLElement, loop: { setSpeed(s: Speed): void; speed: Speed }): void {
  const buttons = [...group.querySelectorAll<HTMLButtonElement>('button[data-speed]')];
  const sync = () => buttons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.speed === loop.speed)));
  buttons.forEach((b) =>
    b.addEventListener('click', () => {
      loop.setSpeed(b.dataset.speed as Speed);
      sync();
    }),
  );
  sync();
}
