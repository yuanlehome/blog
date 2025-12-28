import { defineCollection, z } from 'astro:content';

const dateField = z.coerce.date();

const sourceField = z
  .union([
    z.enum(['original', 'wechat', 'notion']),
    z.object({
      title: z.string(),
      url: z.string().url(),
    }),
  ])
  .optional();

const blog = defineCollection({
  schema: z.object({
    title: z.string(),
    date: dateField,
    updated: dateField.optional(),
    updatedAt: dateField.optional(),
    lastmod: dateField.optional(),
    lastEditedTime: dateField.optional(),
    tags: z.array(z.string()).default([]),
    cover: z.string().optional(), // Path to image in public or URL
    status: z.enum(['published', 'draft']).default('published'),
    notionId: z.string().optional(),
    comments: z.boolean().optional(),
    source_url: z.string().url().optional(),
    source_author: z.string().optional(),
    source: sourceField, // Content source: original (author's own), wechat (imported from WeChat), notion (imported from Notion)
    imported_at: z.string().optional(),
  }),
});

export const collections = { blog };
