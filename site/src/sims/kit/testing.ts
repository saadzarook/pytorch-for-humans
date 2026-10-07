/**
 * Tolerance used by every sim test against PyTorch reference values:
 * |actual − expected| ≤ 1e-6·|expected| + 1e-12.
 * (Purely relative breaks down at exactly 0, hence the tiny absolute floor.)
 */
export const REL_TOL = 1e-6;
export const ABS_FLOOR = 1e-12;

export function close(actual: number, expected: number, rel = REL_TOL): boolean {
  if (!Number.isFinite(expected)) return Object.is(actual, expected) || actual === expected;
  return Math.abs(actual - expected) <= rel * Math.abs(expected) + ABS_FLOOR;
}

/** Compare nested arrays of numbers; returns a description of the first mismatch, or null. */
export function firstMismatch(actual: unknown, expected: unknown, path = ''): string | null {
  if (typeof expected === 'number') {
    if (typeof actual !== 'number') return `${path}: expected a number, got ${typeof actual}`;
    return close(actual, expected) ? null : `${path}: ${actual} ≠ ${expected} (PyTorch)`;
  }
  if (Array.isArray(expected)) {
    const a = actual as ArrayLike<unknown>;
    if (!a || a.length !== expected.length) return `${path}: length ${a?.length} ≠ ${expected.length}`;
    for (let i = 0; i < expected.length; i++) {
      const m = firstMismatch(a[i], expected[i], `${path}[${i}]`);
      if (m) return m;
    }
    return null;
  }
  return `${path}: unsupported value`;
}
