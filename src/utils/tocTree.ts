/**
 * TOC Tree Builder - Builds a forest structure from flat headings
 * Supports multiple H1 headings, each forming a tree root
 */

export interface TocNode {
  depth: number;
  slug: string;
  text: string;
  children: TocNode[];
}

export interface Heading {
  depth: number;
  slug: string;
  text: string;
}

/**
 * Build a forest (array of trees) from flat heading list
 * Each H1 becomes a root, with H2/H3/etc nested underneath
 */
export function buildTocForest(headings: Heading[]): TocNode[] {
  const forest: TocNode[] = [];
  const stack: TocNode[] = [];

  for (const heading of headings) {
    const node: TocNode = {
      ...heading,
      children: [],
    };

    // Remove nodes from stack that are same level or deeper
    while (stack.length > 0 && stack[stack.length - 1].depth >= heading.depth) {
      stack.pop();
    }

    if (stack.length === 0) {
      // This is a root node (typically H1)
      forest.push(node);
    } else {
      // Add as child to the parent
      stack[stack.length - 1].children.push(node);
    }

    stack.push(node);
  }

  return forest;
}

/**
 * Flatten the forest back to a list for linear operations (e.g., collecting all slugs)
 */
export function flattenForest(forest: TocNode[]): TocNode[] {
  const result: TocNode[] = [];

  function traverse(node: TocNode) {
    result.push(node);
    node.children.forEach(traverse);
  }

  forest.forEach(traverse);
  return result;
}
