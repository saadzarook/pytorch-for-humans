/**
 * Narrator: the caption that says what's happening right now and why it matters.
 *
 * Markup (from kit/SimShell.astro):
 *   <div data-part="narrator" class="sim-narr" aria-live="polite">
 *     <span class="ph"></span><b class="t"></b><p></p>
 *   </div>
 *
 * The region is aria-live="polite", so screen readers hear every phase. Setting
 * the same text twice is a no-op, so a sim can call set() on every frame
 * without spamming assistive tech.
 */
export class Narrator {
  private readonly ph: HTMLElement;
  private readonly title: HTMLElement;
  private readonly body: HTMLElement;
  private last = '';

  constructor(el: HTMLElement) {
    this.ph = el.querySelector('.ph')!;
    this.title = el.querySelector('.t')!;
    this.body = el.querySelector('p')!;
  }

  /**
   * @param phase small badge, e.g. "Phase 1 of 3"
   * @param title one-line headline
   * @param bodyHtml explanation. HTML is allowed (b, i, code, br): it always
   *   comes from the sim's own source, never from user input.
   */
  set(phase: string, title: string, bodyHtml: string): void {
    const key = `${phase}\u0000${title}\u0000${bodyHtml}`;
    if (key === this.last) return;
    this.last = key;
    this.ph.textContent = phase;
    this.title.textContent = title;
    this.body.innerHTML = bodyHtml;
  }
}
