import crypto from 'crypto';
import slugify from 'slugify';

export function normalizeSlug(value: string): string {
  return slugify(value, { lower: true, strict: true }) || '';
}

export function deriveSlug({
  explicitSlug,
  title,
  fallbackId,
}: {
  explicitSlug?: string | null;
  title?: string;
  fallbackId: string;
}): string {
  const explicit = explicitSlug ? normalizeSlug(explicitSlug) : '';
  if (explicit) return explicit;
  const fromTitle = title ? normalizeSlug(title) : '';
  if (fromTitle) return fromTitle;
  return normalizeSlug(fallbackId) || fallbackId;
}

export function shortHash(input: string, length = 6): string {
  return crypto.createHash('sha256').update(input).digest('hex').slice(0, length);
}

export function ensureUniqueSlug(
  desired: string,
  ownerId: string,
  used: Map<string, string>,
): string {
  let slug = desired || normalizeSlug(ownerId);
  const existingOwner = used.get(slug);
  if (!existingOwner || existingOwner === ownerId) {
    used.set(slug, ownerId);
    return slug;
  }

  const candidate = `${slug}-${shortHash(ownerId)}`;
  const candidateOwner = used.get(candidate);
  if (!candidateOwner || candidateOwner === ownerId) {
    used.set(candidate, ownerId);
    return candidate;
  }

  let counter = 2;
  let finalSlug = candidate;
  while (used.has(finalSlug) && used.get(finalSlug) !== ownerId) {
    finalSlug = `${candidate}-${counter}`;
    counter += 1;
  }
  used.set(finalSlug, ownerId);
  return finalSlug;
}
