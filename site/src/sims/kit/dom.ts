import { RunLoop, bindSpeedControl, type RunLoopOptions } from './runLoop';
import { bindKeys } from './keyboard';
import { part } from './types';

/**
 * Wire RunControls.astro markup to a RunLoop: Play/Pause label, Step, Reset,
 * speed buttons and keyboard shortcuts. Returns the loop and a cleanup function.
 */
export function wireRunControls(
  root: HTMLElement,
  opts: Omit<RunLoopOptions, 'root' | 'onRunningChange'> & {
    onReset: () => void;
    /** Called before Play starts (e.g. auto-reset when the run already finished). */
    beforePlay?: () => void;
    /** Called before a manual Step (e.g. auto-reset when finished). Not called for Play. */
    beforeStep?: () => void;
    /** Called after a manual step (e.g. to announce the new state). */
    afterStep?: () => void;
    onRunningChange?: (running: boolean) => void;
  },
): { loop: RunLoop; cleanup: () => void } {
  const playBtn = root.querySelector<HTMLButtonElement>('[data-run="play"]')!;
  const loop = new RunLoop({
    ...opts,
    root,
    onRunningChange: (running) => {
      playBtn.textContent = running ? '⏸︎ Pause' : '▶︎ Play';
      opts.onRunningChange?.(running);
    },
  });
  const toggle = () => {
    if (!loop.running) opts.beforePlay?.();
    loop.toggle();
  };
  const step = () => {
    opts.beforeStep?.();
    loop.stepOnce();
    opts.afterStep?.();
  };
  playBtn.addEventListener('click', toggle);
  root.querySelector('[data-run="step"]')!.addEventListener('click', step);
  root.querySelector('[data-run="reset"]')!.addEventListener('click', () => {
    loop.pause();
    opts.onReset();
  });
  bindSpeedControl(part(root, 'speed'), loop);
  const unbindKeys = bindKeys(root, { toggle, next: step });
  return {
    loop,
    cleanup: () => {
      unbindKeys();
      loop.destroy();
    },
  };
}
