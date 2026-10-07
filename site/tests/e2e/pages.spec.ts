/**
 * Page-level checks: in-browser Python (Pyodide), the quiz, facts-driven lesson
 * text, offline-after-load, and phone layout in both themes.
 */
import { expect, test } from '@playwright/test';
import { collectErrors, pyCell, ready, runCell } from './helpers';

const LESSON = '/foundations/gradient-descent/';

test('lesson loads nothing from other sites (works offline after load, except Pyodide)', async ({ page }) => {
  const external: string[] = [];
  page.on('request', (r) => {
    if (!r.url().startsWith('http://127.0.0.1')) external.push(r.url());
  });
  await page.goto(LESSON, { waitUntil: 'networkidle' });
  expect(external).toEqual([]);
});

test('Pyodide: from-scratch cell prints loss, w and b at the same moment', async ({ page }) => {
  await page.goto(LESSON);
  const out = await runCell(pyCell(page, 'Gradient descent from scratch'));
  expect(out).toMatch(/step\s+0\s+loss\s+353\.565\s+w\s+0\.000\s+b\s+0\.000/);
});

test('Pyodide: challenge fails as written, passes when solved', async ({ page }) => {
  await page.goto(LESSON);
  const cell = pyCell(page, 'Challenge: rainy-day chai');
  expect(await runCell(cell)).toContain('❌');
  const editor = cell.getByRole('textbox', { name: 'Python code, editable' });
  const code = (await editor.inputValue())
    .replace('grad_w = 0.0', 'grad_w = np.mean(2 * error * rain_hours)')
    .replace('grad_b = 0.0', 'grad_b = np.mean(2 * error)')
    .replace('w = w\n', 'w = w - lr * grad_w\n')
    .replace('b = b\n', 'b = b - lr * grad_b\n');
  await editor.fill(code);
  expect(await runCell(cell)).toContain('✅');
});

test('Pyodide: the valley nerd note runs and matches facts', async ({ page }) => {
  await page.goto(LESSON);
  await page.getByText('Nerd corner: exactly how lopsided is this valley?').click();
  const out = await runCell(pyCell(page, 'Measure the valley'));
  expect(out).toContain('36x steeper');
  expect(out).toContain('= 0.168');
});

test('Pyodide: Stop kills an infinite loop and Python restarts', async ({ page }) => {
  await page.goto(LESSON);
  const cell = pyCell(page, 'Score a guess');
  await cell.scrollIntoViewIfNeeded();
  await cell.getByRole('textbox', { name: 'Python code, editable' }).fill('while True:\n    pass\n');
  await cell.getByRole('button', { name: /Run$/ }).click();
  const stop = cell.getByRole('button', { name: /Stop/ });
  await expect(stop).toBeVisible({ timeout: 120_000 });
  await page.waitForTimeout(1000);
  await stop.click();
  await expect(cell.getByRole('status')).toHaveText(/Stopped/);
  await cell.getByRole('button', { name: /Reset code/ }).click();
  expect(await runCell(cell)).toContain('Loss (mean squared error) = 54.12');
});

test('quiz: wrong and right answers both explain themselves', async ({ page }) => {
  await page.goto(LESSON);
  const q1 = page.getByRole('region', { name: 'Quick quiz' }).getByRole('group', { name: /slope is negative/ });
  // Every option has its own explanation; only the picked one is shown.
  await q1.getByRole('radio', { name: 'Decreases it' }).check();
  await expect(q1.getByText(/That would walk uphill/)).toBeVisible();
  await expect(q1.getByText(/At the bottom the slope is/)).toBeHidden();
  await q1.getByRole('radio', { name: 'Increases it' }).check();
  await expect(q1.getByText(/A negative slope means the loss goes/)).toBeVisible();
  await expect(q1.getByText(/That would walk uphill/)).toBeHidden();
});

test('lesson text renders numbers from facts', async ({ page }) => {
  await page.goto(LESSON);
  const main = (await page.getByRole('main').textContent())!.replace(/\s+/g, ' ');
  expect(main).toContain('the arrival order is Momentum, then Adam, then SGD (46, 119 and 280 steps)');
  expect(main).toContain('why chaos starts at exactly 1 for this curve');
  expect(main).toContain('tune the learning rate right up to the edge');
  expect(main).toMatch(/lr 1\.0\s+SGD\s+blew up at step 9/);
});

const PAGES = ['/', LESSON, '/gallery/', '/gallery/optimizer-race/', '/gallery/backprop/', '/gallery/convolution/', '/gallery/neural-network/'];
for (const theme of ['light', 'dark'] as const) {
  for (const path of PAGES) {
    test(`375px ${theme} ${path}: no horizontal page scroll, no errors`, async ({ browser }) => {
      const ctx = await browser.newContext({ viewport: { width: 375, height: 812 }, colorScheme: theme });
      const page = await ctx.newPage();
      const errors = collectErrors(page);
      await page.goto(path);
      await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
      for (const s of await page.getByRole('region', { name: /: interactive simulation$/ }).all()) await ready(s);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      expect(overflow).toBeLessThanOrEqual(0);
      expect(errors).toEqual([]);
      await ctx.close();
    });
  }
}
