/**
 * EventLog: "key moments", newest first, at most 5, each tagged with its
 * step/epoch so learners can look back after missing something.
 *
 * Markup: <ul data-part="log" class="sim-log"></ul>. The empty-state hint is
 * pure CSS (`.sim-log:empty::before`), so there's nothing to manage here.
 */
export class EventLog {
  private readonly el: HTMLUListElement;
  private readonly max: number;

  constructor(el: HTMLUListElement, max = 5) {
    this.el = el;
    this.max = max;
  }

  /** @param tag e.g. "step 7"  @param html the moment (from the sim's own source; may contain <b>/<i>) */
  add(tag: string, html: string): void {
    const li = document.createElement('li');
    const t = document.createElement('span');
    t.className = 'tag';
    t.textContent = tag;
    li.append(t);
    li.insertAdjacentHTML('beforeend', html);
    this.el.prepend(li);
    while (this.el.children.length > this.max) this.el.lastElementChild?.remove();
  }

  clear(): void {
    this.el.replaceChildren();
  }
}
