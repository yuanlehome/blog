import type { Plugin } from "unified";
import type { Image } from "mdast";
import { visit } from "unist-util-visit";

interface Options {
  base?: string;
}

const normalizeBase = (base: string) => {
  if (!base.startsWith("/")) return `/${base}`;
  return base;
};

export const remarkPrefixImages: Plugin<[Options?]> = (options = {}) => {
  const base = normalizeBase(options.base ?? "/");

  return (tree) => {
    visit(tree, "image", (node: Image) => {
      if (!node.url) return;
      if (!node.url.startsWith("/")) return;

      const trimmedBase = base.endsWith("/") ? base.slice(0, -1) : base;
      node.url = `${trimmedBase}${node.url}`;
    });
  };
};

export default remarkPrefixImages;
