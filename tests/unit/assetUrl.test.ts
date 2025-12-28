import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { resolveAssetUrl } from '../../src/utils/assetUrl';

describe('resolveAssetUrl', () => {
  const originalEnv = import.meta.env.BASE_URL;

  beforeEach(() => {
    // Reset to production base URL for testing
    (import.meta.env as any).BASE_URL = '/blog/';
  });

  afterEach(() => {
    // Restore original value
    (import.meta.env as any).BASE_URL = originalEnv;
  });

  it('should return undefined for null or undefined input', () => {
    expect(resolveAssetUrl(null)).toBeUndefined();
    expect(resolveAssetUrl(undefined)).toBeUndefined();
  });

  it('should return undefined for empty string', () => {
    expect(resolveAssetUrl('')).toBeUndefined();
  });

  it('should not modify external URLs starting with http://', () => {
    const url = 'http://example.com/image.png';
    expect(resolveAssetUrl(url)).toBe(url);
  });

  it('should not modify external URLs starting with https://', () => {
    const url = 'https://example.com/image.png';
    expect(resolveAssetUrl(url)).toBe(url);
  });

  it('should not modify protocol-relative URLs', () => {
    const url = '//example.com/image.png';
    expect(resolveAssetUrl(url)).toBe(url);
  });

  it('should prefix absolute paths with BASE_URL', () => {
    const url = '/images/cover.png';
    expect(resolveAssetUrl(url)).toBe('/blog/images/cover.png');
  });

  it('should handle BASE_URL without trailing slash', () => {
    (import.meta.env as any).BASE_URL = '/blog';
    const url = '/images/cover.png';
    expect(resolveAssetUrl(url)).toBe('/blog/images/cover.png');
  });

  it('should handle BASE_URL with trailing slash', () => {
    (import.meta.env as any).BASE_URL = '/blog/';
    const url = '/images/cover.png';
    expect(resolveAssetUrl(url)).toBe('/blog/images/cover.png');
  });

  it('should handle root BASE_URL', () => {
    (import.meta.env as any).BASE_URL = '/';
    const url = '/images/cover.png';
    expect(resolveAssetUrl(url)).toBe('/images/cover.png');
  });

  it('should return relative paths as-is', () => {
    const url = 'images/cover.png';
    expect(resolveAssetUrl(url)).toBe(url);
  });

  it('should handle complex absolute paths', () => {
    const url = '/images/wechat/sglang-diffusion-wan2260percent/001-85495dc3.png';
    expect(resolveAssetUrl(url)).toBe(
      '/blog/images/wechat/sglang-diffusion-wan2260percent/001-85495dc3.png',
    );
  });
});
