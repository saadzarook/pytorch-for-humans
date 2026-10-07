export interface StepperOptions {
  /** Element containing the Back / Next / Auto-play buttons and the progress readout. */
  root: HTMLElement;
  /** Go forward one stage (the sim decides what "past the last stage" means). */
  next: () => void;
  /** Go back one stage. */
  back: () => void;
  /** Can we go back from here? */
  canBack: () => boolean;
  /** "step 3 of 11" */
  progress: () => string;
  /** Label for the Next button right now (e.g. "Apply update" on the last stage). */
  nextLabel?: () => string;
  /** Auto-play interval, ms. */
  autoMs?: number;
}

/**
 * Back / Next / Auto-play for sims made of discrete stages.
 *
 * Markup (kit/StepControls.astro): buttons with data-step="back|next|auto"
 * and a [data-part="progress"] readout. Auto-play pauses itself when the sim
 * is scrolled off-screen and stops on any manual Back/Next.
 */
export class Stepper {
  private readonly o: StepperOptions;
  private readonly backBtn: HTMLButtonElement;
  private readonly nextBtn: HTMLButtonElement;
  private readonly autoBtn: HTMLButtonElement | null;
  private readonly progressEl: HTMLElement | null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private visible = true;
  private readonly io: IntersectionObserver;

  constructor(opts: StepperOptions) {
    this.o = opts;
    const q = <T extends Element>(s: string) => opts.root.querySelector<T>(s);
    this.backBtn = q<HTMLButtonElement>('[data-step="back"]')!;
    this.nextBtn = q<HTMLButtonElement>('[data-step="next"]')!;
    this.autoBtn = q<HTMLButtonElement>('[data-step="auto"]');
    this.progressEl = q<HTMLElement>('[data-part="progress"]');
    this.backBtn.addEventListener('click', () => this.back());
    this.nextBtn.addEventListener('click', () => this.next());
    this.autoBtn?.addEventListener('click', () => this.toggleAuto());
    this.io = new IntersectionObserver((es) => (this.visible = es.some((e) => e.isIntersecting)), { threshold: 0.05 });
    this.io.observe(opts.root);
  }

  get autoPlaying(): boolean {
    return this.timer !== null;
  }

  next(): void {
    this.stopAuto();
    this.o.next();
    this.sync();
  }

  back(): void {
    this.stopAuto();
    if (this.o.canBack()) this.o.back();
    this.sync();
  }

  toggleAuto(): void {
    if (this.timer) return this.stopAuto();
    if (this.autoBtn) this.autoBtn.textContent = 'Pause';
    this.o.next();
    this.sync();
    this.timer = setInterval(() => {
      if (!this.visible) return;
      this.o.next();
      this.sync();
    }, this.o.autoMs ?? 3200);
  }

  stopAuto(): void {
    if (!this.timer) return;
    clearInterval(this.timer);
    this.timer = null;
    if (this.autoBtn) this.autoBtn.textContent = 'Auto-play';
  }

  /** Refresh the progress readout and button states (call after any external change). */
  sync(): void {
    if (this.progressEl) this.progressEl.textContent = this.o.progress();
    this.backBtn.disabled = !this.o.canBack();
    if (this.o.nextLabel) this.nextBtn.textContent = this.o.nextLabel();
  }

  destroy(): void {
    this.stopAuto();
    this.io.disconnect();
  }
}
