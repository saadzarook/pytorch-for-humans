import { defineConfig, devices } from '@playwright/test';

/**
 * End-to-end tests against the BUILT site (run `npm run build` first; CI does).
 *   npm run test:e2e            run everything (headless Chromium)
 *   UPDATE_NARRATION=1 npm run test:e2e -- narration
 *                               rewrite tests/e2e/narration.snapshot.json on purpose
 *
 * The Pyodide checks need internet access (Pyodide comes from its CDN).
 */
const PORT = 4322;

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 180_000,
  expect: { timeout: 15_000 },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1280, height: 900 } } }],
  webServer: {
    command: `npx astro preview --port ${PORT} --host 127.0.0.1`,
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
