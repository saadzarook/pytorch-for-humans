import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';
import { docsSchema } from '@astrojs/starlight/schema';

/**
 * All course content lives in ../course so that each lesson's prose
 * (lesson.mdx), notebook (notebook.py) and code snippets sit side by side.
 *
 * IDs (and therefore URLs) are derived from the folder path:
 *   foundations/01-gradient-descent/lesson.mdx  ->  foundations/gradient-descent
 * The numeric prefix only controls folder ordering and never leaks into URLs,
 * so lessons can be reordered without breaking links.
 */
function lessonId({ entry, data }: { entry: string; data: Record<string, unknown> }) {
  if (typeof data.slug === 'string') return data.slug;
  const id = entry
    .replace(/\.(md|mdx)$/, '')
    .replace(/(^|\/)(lesson|index)$/, '')
    .split('/')
    .map((seg) => seg.replace(/^\d+-/, ''))
    .join('/');
  return id || 'index';
}

export const collections = {
  docs: defineCollection({
    loader: glob({
      base: '../course',
      pattern: '**/[^_]*.{md,mdx}',
      generateId: lessonId,
    }),
    schema: docsSchema({
      extend: z.object({
        // Lesson metadata used by the lesson components.
        tier: z.enum(['foundations', 'intermediate', 'advanced']).optional(),
        minutes: z.number().optional(),
        kaggleSlug: z.string().optional(),
      }),
    }),
  }),
};
