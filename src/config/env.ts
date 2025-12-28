export const stringFromEnv = (name: string, fallback?: string) => {
  const value = process.env[name];
  return value && value.trim().length > 0 ? value : fallback;
};

export const boolFromEnv = (name: string, fallback = false) => {
  const value = process.env[name];
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['true', '1', 'yes', 'y', 'on'].includes(normalized)) return true;
    if (['false', '0', 'no', 'n', 'off'].includes(normalized)) return false;
  }
  return fallback;
};

export const numberFromEnv = (name: string, fallback?: number) => {
  const value = process.env[name];
  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
};
