/**
 * Page-wide Pyodide client: one worker shared by every <PyRunner> on the page.
 *
 * - Lazy: nothing downloads until the first Run click.
 * - Serialised: runs are queued so two cells never interleave their output.
 * - Stoppable: `stop()` terminates the worker (the only reliable way to kill
 *   an infinite loop without cross-origin-isolation headers, which GitHub
 *   Pages can't set). The next run starts a fresh worker.
 */

export interface RunHandlers {
  onStdout: (text: string) => void;
  onStderr: (text: string) => void;
  onStatus: (text: string) => void;
}

export interface RunResult {
  ok: boolean;
  error?: string;
  plot?: number[] | null;
  stopped?: boolean;
}

type Pending = { resolve: (r: RunResult) => void; handlers: RunHandlers };

let worker: Worker | null = null;
let ready = false;
let nextId = 1;
const pending = new Map<number, Pending>();
let queue: Promise<unknown> = Promise.resolve();
/** Bumped by stop(); queued jobs from an older generation are cancelled. */
let generation = 0;
const statusListeners = new Set<(text: string) => void>();

function ensureWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL('./pyodide.worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (e: MessageEvent) => {
    const msg = e.data;
    const p = msg.id !== undefined ? pending.get(msg.id) : undefined;
    switch (msg.type) {
      case 'status':
        statusListeners.forEach((fn) => fn(msg.text));
        break;
      case 'ready':
        ready = true;
        statusListeners.forEach((fn) => fn(''));
        break;
      case 'stdout':
        p?.handlers.onStdout(msg.text);
        break;
      case 'stderr':
        p?.handlers.onStderr(msg.text);
        break;
      case 'done':
        pending.delete(msg.id);
        p?.resolve({ ok: true, plot: msg.plot });
        break;
      case 'error':
        pending.delete(msg.id);
        p?.resolve({ ok: false, error: msg.text });
        break;
      case 'fatal':
        if (p) {
          pending.delete(msg.id);
          p.resolve({ ok: false, error: msg.text });
        }
        statusListeners.forEach((fn) => fn(msg.text));
        break;
    }
  };
  worker.onerror = (e) => {
    const text =
      "Couldn't start Python. Are you offline? Python downloads the first time you press Run " +
      '(the sims work without it). Check your connection and try again.' +
      (e.message ? `
(details: ${e.message})` : '');
    pending.forEach((p) => p.resolve({ ok: false, error: text }));
    pending.clear();
    worker?.terminate();
    worker = null;
    ready = false;
  };
  return worker;
}

export function isReady(): boolean {
  return ready;
}

export function onStatus(fn: (text: string) => void): void {
  statusListeners.add(fn);
}

/** Queue a run. Resolves when the code (and optional check) finishes, errors, or is stopped. */
export function run(
  code: string,
  opts: { check?: string; plotVar?: string },
  handlers: RunHandlers,
): Promise<RunResult> {
  const gen = generation;
  const job = queue.then(
    () =>
      new Promise<RunResult>((resolve) => {
        if (gen !== generation) return resolve({ ok: false, stopped: true });
        const id = nextId++;
        const unsub = (text: string) => handlers.onStatus(text);
        statusListeners.add(unsub);
        pending.set(id, {
          handlers,
          resolve: (r) => {
            statusListeners.delete(unsub);
            resolve(r);
          },
        });
        ensureWorker().postMessage({ type: 'run', id, code, ...opts });
      }),
  );
  queue = job.catch(() => undefined);
  return job;
}

/** Kill whatever is running (and everything queued). Python restarts on the next run. */
export function stop(): void {
  generation++;
  if (!worker) return;
  worker.terminate();
  worker = null;
  ready = false;
  pending.forEach((p) => p.resolve({ ok: false, stopped: true }));
  pending.clear();
  queue = Promise.resolve();
}
