export function normalizeBase(base: string): string {
  if (!base) return '/';
  const prefixed = base.startsWith('/') ? base : `/${base}`;
  if (prefixed === '/') return '/';
  return prefixed.endsWith('/') ? prefixed : `${prefixed}/`;
}

export function withBase(path: string, base: string): string {
  if (!path) return path;
  if (path.startsWith('http://') || path.startsWith('https://') || path.startsWith('//')) {
    return path;
  }
  if (!path.startsWith('/')) return path;

  const normalizedBase = normalizeBase(base);
  if (normalizedBase === '/') return path;

  const basePrefix = normalizedBase.slice(0, -1);
  if (path === basePrefix || path.startsWith(`${basePrefix}/`)) {
    return path;
  }

  return `${basePrefix}${path}`;
}

export function withBaseFromEnv(path: string): string {
  return withBase(path, import.meta.env.BASE_URL);
}
