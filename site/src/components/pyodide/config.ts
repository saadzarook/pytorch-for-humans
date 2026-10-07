/**
 * Pinned Pyodide version. Pyodide 314.x bundles CPython 3.14 and NumPy 2.4.
 * Check https://pyodide.org/en/stable/project/changelog.html before bumping,
 * and keep the lesson snippets runnable on both Pyodide and CI's CPython.
 */
export const PYODIDE_VERSION = '314.0.7';
export const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

/** Loaded up-front on first run. Other imports are auto-loaded from the Pyodide CDN when a cell imports them. */
export const PRELOAD_PACKAGES = ['numpy'];
