import type { Plugin } from "unified";
import { visit } from "unist-util-visit";
import type { Root } from "mdast";

const highlightPattern = /\{([^}]+)\}/;
const titlePattern = /(?:title|file|filename)=((?:"[^"]+"|'[^']+'|\S+))/i;
const noLineNumbersPattern =
  /(?:no-?line-?numbers|line-?numbers\s*=\s*false|nolines)/i;

const parseHighlight = (meta?: string): number[] => {
  if (!meta) return [];
  const match = meta.match(highlightPattern);
  if (!match?.[1]) return [];
  return match[1]
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .flatMap((part) => {
      if (part.includes("-")) {
        const [start, end] = part
          .split("-")
          .map((v) => Number.parseInt(v.trim(), 10));
        if (Number.isFinite(start) && Number.isFinite(end)) {
          const low = Math.min(start, end);
          const high = Math.max(start, end);
          return Array.from({ length: high - low + 1 }, (_, idx) => low + idx);
        }
        return [];
      }
      const value = Number.parseInt(part, 10);
      return Number.isFinite(value) ? [value] : [];
    });
};

const cleanMeta = (
  meta?: string,
): {
  meta: string;
  title?: string;
  showLineNumbers: boolean;
  highlights: number[];
} => {
  if (!meta) return { meta: "", showLineNumbers: true, highlights: [] };

  const titleMatch = meta.match(titlePattern);
  const titleRaw = titleMatch?.[1]?.trim();
  const title = titleRaw?.replace(/^['"]|['"]$/g, "");

  const showLineNumbers = !noLineNumbersPattern.test(meta);
  const highlights = parseHighlight(meta);

  const cleaned = meta
    .replace(highlightPattern, "")
    .replace(titlePattern, "")
    .replace(noLineNumbersPattern, "")
    .trim();

  return { meta: cleaned, title, showLineNumbers, highlights };
};

const remarkCodeMeta: Plugin<[], Root> = () => {
  return (tree) => {
    visit(tree, "code", (node) => {
      const lang = node.lang?.trim() || "text";
      const { meta, title, showLineNumbers, highlights } = cleanMeta(
        node.meta || "",
      );

      node.lang = lang;
      node.meta = meta || undefined;

      node.data ||= {};
      const raw = typeof node.value === "string" ? node.value : "";
      // Pass data to HAST so rehype plugins can consume it later
      (node.data as any).hProperties = {
        ...(node.data as any).hProperties,
        dataLang: lang,
        dataMeta: meta,
        dataTitle: title,
        dataHighlightLines: highlights.length
          ? highlights.join(",")
          : undefined,
        dataLineNumbers: showLineNumbers ? "true" : "false",
        dataRawCode: raw,
      };
    });
  };
};

export default remarkCodeMeta;
