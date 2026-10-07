// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';

// GitHub Pages project sites live at https://<user>.github.io/<repo>/.
// CI sets SITE_URL and BASE_PATH; locally we serve from the root.
const site = process.env.SITE_URL || 'http://localhost:4321';
const base = process.env.BASE_PATH || '/';

const courseDir = fileURLToPath(new URL('../course/', import.meta.url));

/**
 * Sidebar entries for one tier, read from the folder names, so adding
 * course/<tier>/NN-some-topic/lesson.mdx is all it takes to list a lesson.
 * Folders sort by their NN- prefix; the URL drops it (see content.config.ts).
 * Starlight takes each entry's label from the lesson's `sidebar.label` / title.
 * @param {string} tier
 */
function tierItems(tier) {
  const dir = courseDir + tier;
  const lessons = fs.existsSync(dir)
    ? fs
        .readdirSync(dir, { withFileTypes: true })
        .filter((d) => d.isDirectory() && /^\d+-/.test(d.name))
        .filter((d) => fs.existsSync(`${dir}/${d.name}/lesson.mdx`))
        .map((d) => d.name)
        .sort()
    : [];
  if (lessons.length === 0) return [{ label: 'Coming soon', slug: tier }];
  return lessons.map((name) => ({ slug: `${tier}/${name.replace(/^\d+-/, '')}` }));
}

export default defineConfig({
  site,
  base,
  trailingSlash: 'always',
  integrations: [
    starlight({
      title: 'PyTorch for Humans',
      description:
        'A free, friendly PyTorch & ML course that explains the *why* — with analogies, interactive sims and real code.',
      logo: { src: './src/assets/logo.svg', alt: '' },
      favicon: '/favicon.svg',
      customCss: ['./src/styles/course.css'],
      // Lessons live in ../course (next to their notebooks), so Starlight's own
      // `autogenerate` can't see them; tierItems() does the same job.
      sidebar: [
        { label: 'Start here', slug: 'index' },
        { label: 'Foundations', items: tierItems('foundations') },
        { label: 'Intermediate', collapsed: true, items: tierItems('intermediate') },
        { label: 'Advanced', collapsed: true, items: tierItems('advanced') },
        { label: 'How lessons work', slug: 'about/lesson-template' },
      ],
    }),
  ],
  vite: {
    // The Pyodide worker dynamically imports pyodide.mjs from the CDN, which
    // needs an ES-module worker (the default IIFE format can't do that).
    worker: { format: 'es' },
    resolve: {
      // Lets lessons in ../course import components as `~/components/...`.
      alias: { '~': fileURLToPath(new URL('./src', import.meta.url)) },
    },
    server: {
      // Lessons and their snippets live outside the Vite root.
      fs: { allow: ['..'] },
    },
  },
});
