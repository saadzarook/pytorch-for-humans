/**
 * Behaviour checks for every sim: presets reach their documented outcomes, the
 * narrator and key-moments log say the right things, step/keyboard/Back work.
 */
import { expect, test } from '@playwright/test';
import {
  button,
  speed,
  collectErrors,
  moments,
  narration,
  narrator,
  playButton,
  ready,
  sim,
  stepButton,
  waitForNarration,
} from './helpers';

const LESSON = '/foundations/gradient-descent/';

test.describe('lesson page sims', () => {
  test('sims mount lazily: none is ready before it scrolls into view', async ({ page }) => {
    await page.goto(LESSON);
    const sims = page.getByRole('region', { name: /: interactive simulation$/ });
    await expect(sims).toHaveCount(4);
    for (const s of await sims.all()) await expect(s).not.toHaveAttribute('data-ready', '');
  });

  test('ball: starts Ready on Slow, nothing autoplays, narrator is a live region', async ({ page }) => {
    await page.goto(LESSON);
    const ball = await ready(sim(page, 'Roll the ball downhill'));
    expect(await narration(ball)).toMatch(/^Ready/);
    await expect(playButton(ball)).toHaveText(/Play/);
    await expect(narrator(ball)).toHaveAttribute('aria-live', 'polite');
    await expect(speed(ball, 'Slow')).toHaveAttribute('aria-pressed', 'true');
  });

  test('ball: just right settles at the bottom after 11 steps', async ({ page }) => {
    await page.goto(LESSON);
    const ball = await ready(sim(page, 'Roll the ball downhill'));
    await button(ball, 'Just right').click();
    await speed(ball, 'Fast').click();
    await playButton(ball).click();
    await waitForNarration(ball, /settled/);
    expect(await narration(ball)).toContain('Settled at the bottom after 11 steps');
    expect(await moments(ball)).toContain('Settled at the bottom');
  });

  test('ball: chaos mode logs overshoot, loss going up and the explosion', async ({ page }) => {
    await page.goto(LESSON);
    const ball = await ready(sim(page, 'Roll the ball downhill'));
    await button(ball, 'Chaos mode').click();
    await speed(ball, 'Fast').click();
    await playButton(ball).click();
    await waitForNarration(ball, /exploded/);
    const log = await moments(ball);
    expect(log).toMatch(/past the bottom/);
    expect(log).toMatch(/Loss went up/);
    expect(log).toMatch(/Exploded/);
    expect(await narration(ball)).toMatch(/above lr = 1\b/); // the bowl's chaos lr, from facts.json
  });

  test('ball: → steps and Space toggles play when the sim has focus', async ({ page }) => {
    await page.goto(LESSON);
    const ball = await ready(sim(page, 'Roll the ball downhill'));
    await ball.getByRole('img', { name: /A loss curve/ }).focus();
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('ArrowRight');
    await expect(ball.getByText(/^step 2 ·/)).toBeVisible();
    await page.keyboard.press(' ');
    await expect(playButton(ball)).toHaveText(/Pause/);
    await page.keyboard.press(' ');
    await expect(playButton(ball)).toHaveText(/Play/);
  });

  test('bumpy: stuck in a dip ends in a local minimum', async ({ page }) => {
    await page.goto(LESSON);
    const bumpy = await ready(sim(page, 'Bumpy hills: local minima'));
    await button(bumpy, 'Stuck in a dip').click();
    await speed(bumpy, 'Fast').click();
    await playButton(bumpy).click();
    await waitForNarration(bumpy, /stuck/);
    expect(await narration(bumpy)).toMatch(/local minimum/);
    expect(await moments(bumpy)).toMatch(/local minimum/);
  });

  test('contour: just right converges in 230, zig-zag in 151, chaos explodes', async ({ page }) => {
    await page.goto(LESSON);
    const contour = await ready(sim(page, 'Fit the line, walk the bowl'));
    await speed(contour, 'Fast').click();

    await button(contour, 'Just right').click();
    await playButton(contour).click();
    await waitForNarration(contour, /converged/);
    expect(await narration(contour)).toContain('Converged after 230 steps');

    await button(contour, 'Zig-zag').click();
    await playButton(contour).click();
    await waitForNarration(contour, /converged/);
    expect(await narration(contour)).toContain('after 151 steps');
    expect(await moments(contour)).toMatch(/Zig-zagging/);

    await button(contour, 'Chaos mode').click();
    await playButton(contour).click();
    await waitForNarration(contour, /exploded/);
    const log = await moments(contour);
    expect(log).toMatch(/Exploded/);
    expect(log).toMatch(/Loss went up/);
  });

  test('optimizer race: phases, arrival order from facts, chaos threshold', async ({ page }) => {
    await page.goto(LESSON);
    const race = await ready(sim(page, 'The optimizer race'));
    expect(await narration(race)).toMatch(/^Ready/);
    await stepButton(race).click();
    expect(await narration(race)).toMatch(/Phase 1 of 3/);

    await speed(race, 'Fast').click();
    await playButton(race).click();
    await waitForNarration(race, /Phase 3 of 3/);
    expect(await narration(race)).toContain('at lr 0.1: Momentum, then Adam, then SGD');
    expect(await moments(race)).toContain('SGD reached the minimum (loss below 0.01) in 280 steps');

    const lr = race.getByRole('slider', { name: /Learning rate/ });
    await lr.focus();
    await page.keyboard.press('End');
    await expect(race.getByText(/chaos zone/)).toBeVisible();
    await button(race, /Reset/).click();
    await playButton(race).click();
    await expect(race.getByRole('list', { name: 'Key moments' })).toContainText('SGD blew up', { timeout: 30_000 });

    await lr.focus();
    for (let i = 0; i < 5; i++) await page.keyboard.press('ArrowLeft'); // below the facts.json threshold
    await expect(race.getByText(/chaos zone/)).toHaveCount(0);
  });
});

test.describe('gallery sims', () => {
  test('backprop: Back/Next walk the round, keys work, numbers match autograd', async ({ page }) => {
    await page.goto('/gallery/backprop/');
    const bp = await ready(sim(page, 'Backprop as a chain of blame'));
    await expect(bp.getByText('step 1 of 11')).toBeVisible();
    await expect(button(bp, /Back/)).toBeDisabled();
    for (let i = 0; i < 9; i++) await button(bp, /^Next/).click();
    expect(await narration(bp)).toMatch(/Backward 5 of 5.*= -18\.00/);
    const graph = bp.getByRole('img', { name: /Computation graph/ });
    await expect(graph).toContainText('∂L/∂w = -18.00');

    await graph.focus();
    await page.keyboard.press('ArrowLeft');
    expect(await narration(bp)).toMatch(/Backward 4 of 5/);
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('ArrowRight');
    await expect(button(bp, 'Apply update')).toBeVisible();
    await page.keyboard.press('ArrowRight');
    expect(await narration(bp)).toMatch(/Round 2 · setup.*reset to zero/);

    await button(bp, /Skip 10 rounds/).click();
    expect(await narration(bp)).toMatch(/Skipped ahead 10 rounds/);
  });

  test('convolution: beats, Back, interesting window, done narration from facts', async ({ page }) => {
    await page.goto('/gallery/convolution/');
    const cv = await ready(sim(page, 'Convolution, one window at a time'));
    expect(await narration(cv)).toMatch(/^Ready/);
    for (let i = 0; i < 3; i++) await stepButton(cv).click();
    expect(await narration(cv)).toMatch(/Window 1 of 100 · beat 3: add up/);
    await button(cv, /Back/).click();
    expect(await narration(cv)).toMatch(/beat 2: multiply/);
    await button(cv, 'Next interesting window').click();
    expect(await narration(cv)).toMatch(/Add the products/);

    await speed(cv, 'Fast').click();
    await playButton(cv).click();
    await waitForNarration(cv, /Done/);
    const n = await narration(cv);
    expect(n).toContain('12 − 3 + 1 = 10');
    expect(n).toContain('middle of the crossbar stayed 0');
  });

  test('neural network: first epoch logged, trains, milestones logged', async ({ page }) => {
    await page.goto('/gallery/neural-network/');
    const nn = await ready(sim(page, 'Watch a network learn'));
    expect(await narration(nn)).toMatch(/^Epoch 0/);
    await stepButton(nn, '1 epoch').click();
    expect(await moments(nn)).toContain('First weight update');
    await speed(nn, 'Fast').click();
    await playButton(nn).click();
    await expect(nn.getByText(/^epoch (\d{3,})$/)).toBeVisible({ timeout: 30_000 });
    await playButton(nn).click();
    await expect(nn.getByText(/^accuracy \d+%$/)).toBeVisible();
    expect(await moments(nn)).toMatch(/Accuracy passed|100% accuracy/);
  });

  test('reduced motion: sims still step', async ({ browser }) => {
    const ctx = await browser.newContext({ reducedMotion: 'reduce' });
    const page = await ctx.newPage();
    await page.goto('/gallery/optimizer-race/');
    const race = await ready(sim(page, 'The optimizer race'));
    await stepButton(race).click();
    await expect(race.getByText('step 1', { exact: true })).toBeVisible();
    await ctx.close();
  });
});

test('no errors on any sim page', async ({ page }) => {
  const errors = collectErrors(page);
  for (const path of [LESSON, '/gallery/optimizer-race/', '/gallery/backprop/', '/gallery/convolution/', '/gallery/neural-network/']) {
    await page.goto(path);
    for (const s of await page.getByRole('region', { name: /: interactive simulation$/ }).all()) await ready(s);
  }
  expect(errors).toEqual([]);
});
