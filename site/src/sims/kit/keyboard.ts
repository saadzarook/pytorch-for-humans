export interface KeyHandlers {
  /** Space: play / pause (continuous sims) or auto-play (staged sims). */
  toggle?: () => void;
  /** →: step / next. */
  next?: () => void;
  /** ←: back. */
  back?: () => void;
}

/**
 * Keyboard shortcuts, active while focus is inside the sim.
 *
 * Form controls keep their own keys: sliders still use the arrows, selects and
 * text fields are left alone, and Space on a button still clicks it. Shortcuts
 * fire when focus is on the sim itself or on a canvas/diagram inside it
 * (those are focusable with tabindex="0").
 */
export function bindKeys(root: HTMLElement, h: KeyHandlers): () => void {
  const onKey = (e: KeyboardEvent) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const t = e.target as HTMLElement;
    if (t.closest('input, select, textarea, [contenteditable="true"]')) return;
    const onButton = !!t.closest('button, a, summary');
    let handler: (() => void) | undefined;
    if (e.key === ' ' && !onButton) handler = h.toggle;
    else if (e.key === 'ArrowRight') handler = h.next;
    else if (e.key === 'ArrowLeft') handler = h.back;
    if (!handler) return;
    e.preventDefault();
    handler();
  };
  root.addEventListener('keydown', onKey);
  return () => root.removeEventListener('keydown', onKey);
}
