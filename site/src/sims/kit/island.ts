/**
 * <pfh-sim data-sim="name" data-options='{…}'>: lazy-mounting sim island.
 *
 * Astro's `client:visible` only applies to framework components, so this is
 * the framework-free equivalent: the server renders the sim's markup, and the
 * sim's code is downloaded and mounted only when it scrolls near the viewport.
 * Each entry below becomes its own code-split chunk, sharing the kit.
 */
import type { MountFn, SimHandle } from './types';

const registry: Record<string, () => Promise<{ mount: MountFn<any> }>> = {
  ball: () => import('../ball/mount'),
  contour: () => import('../contour/mount'),
  'optimizer-race': () => import('../optimizer-race/mount'),
  backprop: () => import('../backprop/mount'),
  convolution: () => import('../convolution/mount'),
  'neural-network': () => import('../neural-network/mount'),
};

class PfhSim extends HTMLElement {
  private handle: SimHandle | null = null;
  private io: IntersectionObserver | null = null;
  private gone = false;

  connectedCallback(): void {
    this.gone = false;
    this.io = new IntersectionObserver(
      (entries) => {
        if (!entries.some((e) => e.isIntersecting)) return;
        this.io?.disconnect();
        void this.load();
      },
      { rootMargin: '300px 0px' },
    );
    this.io.observe(this);
  }

  disconnectedCallback(): void {
    this.gone = true;
    this.io?.disconnect();
    this.handle?.destroy();
    this.handle = null;
  }

  private async load(): Promise<void> {
    const name = this.dataset.sim ?? '';
    const loader = registry[name];
    if (!loader) {
      console.error(`Unknown sim "${name}"`);
      return;
    }
    try {
      const { mount } = await loader();
      if (this.gone) return;
      this.handle = mount(this, JSON.parse(this.dataset.options || '{}'));
      this.setAttribute('data-ready', '');
    } catch (err) {
      console.error(err);
      const msg = this.querySelector('[data-part="loading"]');
      if (msg) msg.textContent = "This sim couldn't load. Try refreshing the page.";
    }
  }
}

if (!customElements.get('pfh-sim')) customElements.define('pfh-sim', PfhSim);
