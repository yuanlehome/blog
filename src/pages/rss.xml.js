import rss from "@astrojs/rss";
import { getCollection } from "astro:content";

export async function GET(context) {
  const posts = await getCollection("blog");
  return rss({
    title: "My Astro Blog",
    description: "A minimal blog built with Astro and Notion",
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.date,
      description: post.data.excerpt,
      link: `${post.slug}/`,
    })),
    customData: `<language>en-us</language>`,
  });
}
