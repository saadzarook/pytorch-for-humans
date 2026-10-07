/**
 * Narration snapshots: the exact narrator strings (and key-moments log) for every
 * sim and every preset, walked through the same deterministic sequence:
 * starting state, after picking the preset, the first steps, and the end.
 *
 * Refactors must leave narration byte-identical; each test compares against its
 * committed file in tests/e2e/narration/. To change narration on purpose:
 *   UPDATE_NARRATION=1 npx playwright test narration
 * and review the JSON diff.
 */
import fs from 'node:fs';
import { expect, test, type Locator, type Page } from '@playwright/test';
import { button, moments, narration, playButton, ready, sim, speed, stepButton, waitForNarration } from './helpers';

type Snapshot = Record<string, string[]>;

function compare(name: string, actual: Snapshot) {
  const file = new URL(`./narration/${name}.json`, import.meta.url);
  if (process.env.UPDATE_NARRATION) {
    fs.mkdirSync(new URL('./narration/', import.meta.url), { recursive: true });
    fs.writeFileSync(file, JSON.stringify(actual, null, 2) + '\n');
    return;
  }
  expect(actual).toEqual(JSON.parse(fs.readFileSync(file, 'utf8')));
}

async function playToEnd(s: Locator, end: RegExp) {
  await speed(s, 'Fast').click();
  await playButton(s).click();
  await waitForNarration(s, end, 120_000);
}

/** Continuous sims with a preset group: start, then per preset: picked, 3 steps, end state, log. */
async function walkPresets(page: Page, title: string, group: string, end: RegExp): Promise<Snapshot> {
  const s = await ready(sim(page, title));
  const rec: Snapshot = { start: [await narration(s)] };
  const presets = s.getByRole('group', { name: group });
  for (const name of (await presets.getByRole('button').allTextContents()).map((n) => n.trim())) {
    await presets.getByRole('button', { name, exact: true }).click();
    await speed(s, 'Slow').click();
    const seq = [await narration(s)];
    for (let i = 0; i < 3; i++) {
      await stepButton(s).click();
      seq.push(await narration(s));
    }
    await playToEnd(s, end);
    seq.push(await narration(s), `log: ${await moments(s)}`);
    rec[name] = seq;
  }
  return rec;
}

const LESSON = '/foundations/gradient-descent/';
test.describe.configure({ timeout: 300_000 });

test('narration: ball (smooth bowl)', async ({ page }) => {
  await page.goto(LESSON);
  compare('ball-bowl', await walkPresets(page, 'Roll the ball downhill', 'Presets', /settled|exploded|stuck|gave up/));
});

test('narration: ball (bumpy hills)', async ({ page }) => {
  await page.goto(LESSON);
  compare('ball-bumpy', await walkPresets(page, 'Bumpy hills: local minima', 'Presets', /settled|exploded|stuck|gave up/));
});

test('narration: contour', async ({ page }) => {
  await page.goto(LESSON);
  compare('contour', await walkPresets(page, 'Fit the line, walk the bowl', 'Learning-rate presets', /converged|exploded|out of time/));
});

test('narration: optimizer race (default lr, then lr 1)', async ({ page }) => {
  await page.goto('/gallery/optimizer-race/');
  const race = await ready(sim(page, 'The optimizer race'));
  const seq = [await narration(race)];
  for (let i = 0; i < 3; i++) {
    await stepButton(race).click();
    seq.push(await narration(race));
  }
  await playToEnd(race, /Phase 3 of 3|Game over/);
  seq.push(await narration(race), `log: ${await moments(race)}`);
  await race.getByRole('slider', { name: /Learning rate/ }).focus();
  await page.keyboard.press('End');
  await button(race, /Reset/).click();
  seq.push(await narration(race));
  await playButton(race).click();
  await waitForNarration(race, /Phase 3 of 3|Game over/);
  seq.push(await narration(race), `log: ${await moments(race)}`);
  compare('optimizer-race', { 'default lr, then lr 1': seq });
});

test('narration: backprop', async ({ page }) => {
  await page.goto('/gallery/backprop/');
  const bp = await ready(sim(page, 'Backprop as a chain of blame'));
  const seq = [await narration(bp)];
  for (let i = 0; i < 11; i++) {
    await button(bp, /^(Next|Apply update)/).click();
    seq.push(await narration(bp));
  }
  await button(bp, /Skip 10 rounds/).click();
  seq.push(await narration(bp));
  compare('backprop', { 'round 1, round 2 setup, skip': seq });
});

test('narration: convolution (every preset kernel)', async ({ page }) => {
  await page.goto('/gallery/convolution/');
  const cv = await ready(sim(page, 'Convolution, one window at a time'));
  const kernel = cv.getByRole('combobox', { name: 'Kernel' });
  const rec: Snapshot = {};
  for (const option of (await kernel.getByRole('option').allTextContents()).map((o) => o.trim())) {
    if (option === 'custom') continue;
    await kernel.selectOption({ label: option });
    await button(cv, /Restart/).click();
    await speed(cv, 'Slow').click();
    const seq = [await narration(cv)];
    for (let i = 0; i < 3; i++) {
      await stepButton(cv).click();
      seq.push(await narration(cv));
    }
    await button(cv, 'Next interesting window').click();
    seq.push(await narration(cv));
    await playToEnd(cv, /Done/);
    seq.push(await narration(cv));
    rec[option] = seq;
  }
  compare('convolution', rec);
});

test('narration: neural network (epochs 0-12)', async ({ page }) => {
  await page.goto('/gallery/neural-network/');
  const nn = await ready(sim(page, 'Watch a network learn'));
  const seq = [await narration(nn)];
  for (let i = 0; i < 12; i++) {
    await stepButton(nn, '1 epoch').click();
    seq.push(await narration(nn));
  }
  seq.push(`log: ${await moments(nn)}`);
  compare('neural-network', { 'epochs 0-12': seq });
});
