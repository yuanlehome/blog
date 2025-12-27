import { defineCollection, z } from 'astro:content';

const blog = defineCollection({
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    tags: z.array(z.string()).default([]),
    cover: z.string().optional(), // Path to image in public or URL
    status: z.enum(['published', 'draft']).default('published'),
    notionId: z.string().optional(),
    comments: z.boolean().optional(),
    source: z.string().optional(),
    source_url: z.string().url().optional(),
    source_author: z.string().optional(),
    imported_at: z.string().optional(),
  }),
});

export const collections = { blog };
