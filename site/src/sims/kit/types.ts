/** What every sim module exports. */
export interface SimHandle {
  /** Stop timers/observers and drop listeners. Called when the sim leaves the page. */
  destroy(): void;
}

/**
 * Every sim is a module exporting `mount(el, options)`.
 * `el` is the <pfh-sim> element, already containing the sim's server-rendered
 * markup (title, narrator, canvases, controls); mount() wires it up.
 */
export type MountFn<Options = unknown> = (el: HTMLElement, options: Options) => SimHandle;

/** Find a `[data-part="name"]` element inside a sim (throws if the markup is missing it). */
export function part<T extends Element = HTMLElement>(root: ParentNode, name: string): T {
  const el = root.querySelector<T>(`[data-part="${name}"]`);
  if (!el) throw new Error(`sim markup is missing [data-part="${name}"]`);
  return el;
}
