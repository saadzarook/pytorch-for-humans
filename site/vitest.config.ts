import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

// Unit tests for the sims' maths: each sim is checked against reference values
// computed with real PyTorch (golden.json) and against the claims its narration
// makes (facts.json). Lesson-level parity tests live next to the lesson.
export default defineConfig({
  resolve: {
    alias: { '~': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  test: {
    include: ['src/**/*.test.ts', '../course/**/*.test.ts'],
    environment: 'node',
  },
});
