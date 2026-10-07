/**
 * Shared helpers for the e2e tests. Everything is found by ROLE and accessible
 * NAME (title / label), never by position or CSS class, so the tests read like
 * what a learner sees and survive layout refactors.
 */
import { expect, type Locator, type Page } from '@playwright/test';

/** A sim by its title (SimShell renders a region named "<title>: interactive simulation"). */
export function sim(page: Page, title: string): Locator {
  return page.getByRole('region', { name: `${title}: interactive simulation`, exact: true });
}

/** Scroll the sim into view and wait for its lazy island to mount. */
export async function ready(s: Locator): Promise<Locator> {
  await s.scrollIntoViewIfNeeded();
  await expect(s).toHaveAttribute('data-ready', '');
  return s;
}

export const narrator = (s: Locator) => s.getByRole('status', { name: 'Narrator' });
export const keyMoments = (s: Locator) => s.getByRole('list', { name: 'Key moments' });

/** Narrator text with whitespace collapsed (textContent: the phase badge is CSS-uppercased). */
export async function narration(s: Locator): Promise<string> {
  return ((await narrator(s).textContent()) ?? '').replace(/\s+/g, ' ').trim();
}
export async function moments(s: Locator): Promise<string> {
  return ((await keyMoments(s).textContent()) ?? '').replace(/\s+/g, ' ').trim();
}

export const button = (s: Locator, name: string | RegExp) => s.getByRole('button', { name });
/** Slow / Normal / Fast, inside the sim's "Speed" group (exact, so "Slow" ≠ "🐢 Too slow"). */
export const speed = (s: Locator, name: 'Slow' | 'Normal' | 'Fast') =>
  s.getByRole('group', { name: 'Speed' }).getByRole('button', { name, exact: true });
export const playButton = (s: Locator) => s.getByRole('button', { name: /^(▶︎?|⏸︎?)\s*(Play|Pause)$/ });
export const stepButton = (s: Locator, name: string | RegExp = 'Step') => s.getByRole('button', { name, exact: typeof name === 'string' });

/** Wait until the narrator matches. */
export async function waitForNarration(s: Locator, re: RegExp, timeout = 90_000): Promise<void> {
  await expect(narrator(s)).toHaveText(re, { timeout });
}

/** Collect uncaught errors and console errors on a page. */
export function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  return errors;
}

/** A PyRunner cell by its title. */
export function pyCell(page: Page, title: string): Locator {
  return page.getByRole('region', { name: `Python: ${title}` });
}

/** Run a PyRunner cell and return its output text (first run downloads Pyodide). */
export async function runCell(cell: Locator): Promise<string> {
  await cell.scrollIntoViewIfNeeded();
  await cell.getByRole('button', { name: /Run$/ }).click();
  await expect(cell.getByRole('status')).toHaveText(/Done|Error|Stopped/, { timeout: 180_000 });
  return (await cell.getByLabel('Output').textContent()) ?? '';
}
