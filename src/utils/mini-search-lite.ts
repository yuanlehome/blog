export type MiniSearchLiteOptions = {
  fields: string[];
  weights?: Record<string, number>;
};

export type MiniSearchLiteDocument = Record<string, any> & { id: string };

type SerializedIndex = Array<[string, Array<[string, number]>]>;

type SerializedPayload = {
  options: MiniSearchLiteOptions;
  index: SerializedIndex;
};

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function normalizeText(value: string): string {
  return value
    .toLowerCase()
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/[^\p{L}\p{N}\s'-]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(value: string): string[] {
  const normalized = normalizeText(value);
  if (!normalized) return [];

  const tokens: string[] = [];
  const wordMatches = normalized.match(/[\p{L}\p{N}][\p{L}\p{N}'-]*/gu);
  if (wordMatches) tokens.push(...wordMatches);

  const cjkMatches = normalized.match(/[\p{Script=Han}]/gu);
  if (cjkMatches) tokens.push(...cjkMatches);

  return tokens;
}

export class MiniSearchLite {
  private options: MiniSearchLiteOptions;
  private index: Map<string, Map<string, number>> = new Map();

  constructor(options: MiniSearchLiteOptions) {
    this.options = options;
  }

  add(document: MiniSearchLiteDocument): void {
    const { fields, weights = {} } = this.options;
    const id = document.id;

    fields.forEach((field) => {
      const rawValue = document[field];
      if (rawValue === undefined || rawValue === null) return;

      const value = Array.isArray(rawValue) ? rawValue.join(" ") : String(rawValue);
      const weight = weights[field] ?? 1;
      const tokens = tokenize(value);

      tokens.forEach((token) => {
        if (!token) return;
        const currentPosting = this.index.get(token) ?? new Map<string, number>();
        const currentScore = currentPosting.get(id) ?? 0;
        currentPosting.set(id, currentScore + weight);
        this.index.set(token, currentPosting);
      });
    });
  }

  addAll(documents: MiniSearchLiteDocument[]): void {
    documents.forEach((doc) => this.add(doc));
  }

  search(query: string): Array<{ id: string; score: number; terms: string[] }> {
    const searchTokens = tokenize(query);
    if (searchTokens.length === 0) return [];

    const scores = new Map<string, number>();
    const matchedTerms = new Map<string, Set<string>>();

    for (const token of searchTokens) {
      // Exact token matches
      const postings = this.index.get(token);
      if (postings) {
        postings.forEach((score, docId) => {
          scores.set(docId, (scores.get(docId) ?? 0) + score);
          const set = matchedTerms.get(docId) ?? new Set<string>();
          set.add(token);
          matchedTerms.set(docId, set);
        });
      }

      // Prefix matches to make typing forgiving
      for (const [indexedToken, postingsByDoc] of this.index.entries()) {
        if (indexedToken === token || !indexedToken.startsWith(token)) continue;
        postingsByDoc.forEach((score, docId) => {
          scores.set(docId, (scores.get(docId) ?? 0) + score * 0.5);
          const set = matchedTerms.get(docId) ?? new Set<string>();
          set.add(indexedToken);
          matchedTerms.set(docId, set);
        });
      }
    }

    return Array.from(scores.entries())
      .map(([id, score]) => ({ id, score, terms: Array.from(matchedTerms.get(id) ?? []) }))
      .sort((a, b) => b.score - a.score);
  }

  toJSON(): SerializedPayload {
    const index: SerializedIndex = Array.from(this.index.entries()).map(([token, postings]) => [token, Array.from(postings.entries())]);
    return { options: this.options, index };
  }

  static loadJSON(payload: SerializedPayload): MiniSearchLite {
    const instance = new MiniSearchLite(payload.options);
    instance.index = new Map(payload.index.map(([token, entries]) => [token, new Map(entries)]));
    return instance;
  }

  static highlight(text: string, terms: string[]): string {
    const uniqueTerms = Array.from(new Set(terms.filter(Boolean)));
    if (uniqueTerms.length === 0) return text;

    const pattern = uniqueTerms.map((t) => escapeRegExp(t)).join("|");
    const regex = new RegExp(`(${pattern})`, "gi");
    return text.replace(regex, "<mark>$1</mark>");
  }
}
