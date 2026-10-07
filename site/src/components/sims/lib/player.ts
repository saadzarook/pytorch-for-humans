import { prefersReducedMotion } from './theme';

export interface PlayerOptions {
  /** How many gradient-descent steps to take per second while playing. */
  stepsPerSecond: number;
  /** Advance the simulation by one step. Return `false` to stop playing (converged, exploded…). */
  step: () => boolean;
  /**
   * Draw the current frame. `t` (0→1) is how far we are through the animation
   * between the previous and current step, for smooth tweening. It's always 1
   * when the reader prefers reduced motion.
   */
  render: (t: number) => void;
  /** Called whenever play/pause state changes, so the UI can update its buttons. */
  onPlayingChange?: (playing: boolean) => void;
}

/**
 * Drives a step-based simulation: play / pause / single-step, at a fixed
 * step rate, with an optional tween between steps.
 *
 * Reduced motion: no tweening (things jump straight to their new position)
 * and autoplay runs at a calmer pace. Playback is always reader-initiated.
 */
export class Player {
  playing = false;
  private raf = 0;
  private lastStepAt = -Infinity;
  private readonly opts: PlayerOptions;

  constructor(opts: PlayerOptions) {
    this.opts = opts;
  }

  private get interval(): number {
    const sps = prefersReducedMotion()
      ? Math.min(this.opts.stepsPerSecond, 4)
      : this.opts.stepsPerSecond;
    return 1000 / sps;
  }

  private get tweenMs(): number {
    return prefersReducedMotion() ? 0 : Math.min(220, this.interval * 0.85);
  }

  play(): void {
    if (this.playing) return;
    this.playing = true;
    this.opts.onPlayingChange?.(true);
    this.lastStepAt = -Infinity; // take the first step immediately
    this.kick();
  }

  pause(): void {
    if (!this.playing) return;
    this.playing = false;
    this.opts.onPlayingChange?.(false);
  }

  toggle(): void {
    if (this.playing) this.pause();
    else this.play();
  }

  /** Take exactly one step (pauses playback first). Returns the step callback's result. */
  stepOnce(): boolean {
    this.pause();
    const more = this.opts.step();
    this.lastStepAt = performance.now();
    this.kick();
    return more;
  }

  /** Redraw immediately with no tween (after reset, resize or a theme change). */
  redraw(): void {
    this.opts.render(1);
  }

  private kick(): void {
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(this.frame);
  }

  private frame = (now: number): void => {
    if (this.playing && now - this.lastStepAt >= this.interval) {
      const more = this.opts.step();
      this.lastStepAt = now;
      if (!more) this.pause();
    }
    const tween = this.tweenMs;
    const t = tween === 0 ? 1 : Math.min(1, (now - this.lastStepAt) / tween);
    this.opts.render(easeOut(t));
    if (this.playing || t < 1) this.raf = requestAnimationFrame(this.frame);
  };
}

function easeOut(t: number): number {
  return 1 - (1 - t) * (1 - t);
}
