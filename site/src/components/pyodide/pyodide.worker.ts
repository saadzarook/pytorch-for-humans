/// <reference lib="webworker" />
/**
 * Runs Python (Pyodide + NumPy) off the main thread so the page never freezes,
 * and so a runaway loop can be killed by terminating the worker.
 *
 * Pyodide is the only thing the course loads from a CDN. It's fetched on the
 * first "Run" click, then cached by the browser.
 */
import { PYODIDE_INDEX_URL, PRELOAD_PACKAGES } from './config';

type Msg =
  | { type: 'init' }
  | { type: 'run'; id: number; code: string; check?: string; plotVar?: string };

// Pyodide's types aren't installed (it comes from the CDN), so keep this loose.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let pyodidePromise: Promise<any> | null = null;

function post(msg: unknown) {
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(msg);
}

function getPyodide() {
  pyodidePromise ??= (async () => {
    post({ type: 'status', text: 'Downloading Python…' });
    const { loadPyodide } = await import(/* @vite-ignore */ `${PYODIDE_INDEX_URL}pyodide.mjs`);
    const py = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });
    post({ type: 'status', text: 'Loading NumPy…' });
    await py.loadPackage(PRELOAD_PACKAGES, { messageCallback: () => {} });
    post({ type: 'ready' });
    return py;
  })();
  return pyodidePromise;
}

/** Drop Pyodide's internal frames so learners see only their own code in tracebacks. */
function cleanTraceback(message: string): string {
  const lines = message.split('\n');
  const first = lines.findIndex((l) => l.includes('File "<exec>"'));
  if (first === -1) return message.trim();
  return ['Traceback (most recent call last):', ...lines.slice(first)].join('\n').trim();
}

self.onmessage = async (event: MessageEvent<Msg>) => {
  const msg = event.data;
  if (msg.type === 'init') {
    try {
      await getPyodide();
    } catch (err) {
      post({ type: 'fatal', text: String(err) });
    }
    return;
  }

  const { id, code, check, plotVar } = msg;
  let py;
  try {
    py = await getPyodide();
  } catch (err) {
    post({
      type: 'fatal',
      id,
      text: `Couldn't download Python. Are you offline? Check your connection and press Run again.\n(details: ${String(err)})`,
    });
    pyodidePromise = null; // allow a retry once the connection is back
    return;
  }

  py.setStdout({ batched: (text: string) => post({ type: 'stdout', id, text }) });
  py.setStderr({ batched: (text: string) => post({ type: 'stderr', id, text }) });

  // Every run gets a fresh namespace: each code cell is self-contained.
  const ns = py.globals.get('dict')();
  try {
    // Silence the loader's "No new packages to load" chatter; status goes to the UI.
    await py.loadPackagesFromImports(code, {
      messageCallback: (m: string) => post({ type: 'status', text: m }),
    });
    await py.runPythonAsync(code, { globals: ns });
    if (check) await py.runPythonAsync(check, { globals: ns, filename: '<check>' });

    let plot: number[] | null = null;
    if (plotVar && ns.has(plotVar)) {
      const value = ns.get(plotVar);
      const js = value?.toJs ? value.toJs() : value;
      value?.destroy?.();
      if (js && typeof js.length === 'number') {
        // NaN / ∞ are kept on purpose: the chart shows them as "off the chart".
        plot = Array.from(js as ArrayLike<unknown>, Number);
      }
    }
    post({ type: 'done', id, plot });
  } catch (err) {
    const text = err instanceof Error ? err.message : String(err);
    post({ type: 'error', id, text: cleanTraceback(text) });
  } finally {
    ns.destroy();
  }
};
