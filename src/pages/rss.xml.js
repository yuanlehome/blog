import rss from "@astrojs/rss";
import { getCollection } from "astro:content";

export async function GET(context) {
  const posts = (await getCollection("blog"))
    .filter((post) => post.data.status === "published")
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
  return rss({
    title: "Yuanle Liu‘s Blog",
    description: "Engineering notes from Astro + Notion",
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.date,
      description: post.body.slice(0, 180),
      categories: post.data.tags,
      link: `${post.slug}/`,
    })),
    customData: `<language>en-us</language>`,
  });
}
