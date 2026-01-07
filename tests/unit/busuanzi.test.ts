import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';
import { JSDOM } from 'jsdom';
import {
  isBusuanziEnabled,
  getBusuanziScriptUrl,
  isBusuanziDebugEnabled,
  loadBusuanzi,
  revealContainers,
  initBusuanzi,
  resetBusuanziState,
} from '../../src/lib/analytics/busuanzi';

// Mock the config loader
vi.mock('../../src/config/loaders', () => {
  let mockConfig = {
    busuanzi: {
      enabled: false,
      scriptUrl: 'https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js',
      debug: false,
    },
  };

  return {
    getSiteConfig: vi.fn(() => mockConfig),
    __setMockConfig: (config: any) => {
      mockConfig = { ...mockConfig, ...config };
    },
  };
});

describe('Busuanzi Analytics', () => {
  let dom: JSDOM;
  let document: Document;
  let window: Window & typeof globalThis;

  beforeEach(async () => {
    // Reset mock config
    const { __setMockConfig } = await import('../../src/config/loaders');
    __setMockConfig({
      busuanzi: {
        enabled: false,
        scriptUrl: 'https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js',
        debug: false,
      },
    });

    // Reset Busuanzi state
    resetBusuanziState();

    // Create a fresh DOM for each test
    dom = new JSDOM('<!doctype html><html><head></head><body></body></html>', {
      url: 'http://localhost/',
    });
    document = dom.window.document;
    window = dom.window as unknown as Window & typeof globalThis;

    // Set up global objects
    global.document = document;
    global.window = window;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('isBusuanziEnabled', () => {
    it('returns false when busuanzi.enabled is false', async () => {
      const { __setMockConfig } = await import('../../src/config/loaders');
      __setMockConfig({ busuanzi: { enabled: false } });
      expect(isBusuanziEnabled()).toBe(false);
    });

    it('returns true when busuanzi.enabled is true', async () => {
      const { __setMockConfig } = await import('../../src/config/loaders');
      __setMockConfig({ busuanzi: { enabled: true } });
      expect(isBusuanziEnabled()).toBe(true);
    });

    it('returns false when busuanzi config is missing', async () => {
      const { __setMockConfig } = await import('../../src/config/loaders');
      __setMockConfig({ busuanzi: undefined });
      expect(isBusuanziEnabled()).toBe(false);
    });
  });

  describe('getBusuanziScriptUrl', () => {
    it('returns default URL when not configured', async () => {
      const { __setMockConfig } = await import('../../src/config/loaders');
      __setMockConfig({
        busuanzi: {
          scriptUrl: 'https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js',
        },
      });
      expect(getBusuanziScriptUrl()).toBe(
        'https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js',
      );
    });

    it('returns custom URL when configured', async () => {
      const customUrl = 'https://example.com/busuanzi.js';
      const { __setMockConfig } = await import('../../src/config/loaders');
      __setMockConfig({ busuanzi: { scriptUrl: customUrl } });
      expect(getBusuanziScriptUrl()).toBe(customUrl);
    });
  });

  describe('isBusuanziDebugEnabled', () => {
    it('returns false when debug is false', async () => {
      const { __setMockConfig } = await import('../../src/config/loaders');
      __setMockConfig({ busuanzi: { debug: false } });
      expect(isBusuanziDebugEnabled()).toBe(false);
    });

    it('returns true when debug is true', async () => {
      const { __setMockConfig } = await import('../../src/config/loaders');
      __setMockConfig({ busuanzi: { debug: true } });
      expect(isBusuanziDebugEnabled()).toBe(true);
    });
  });

  describe('loadBusuanzi', () => {
    it('does not load script when Busuanzi is disabled', async () => {
      const { __setMockConfig } = await import('../../src/config/loaders');
      __setMockConfig({ busuanzi: { enabled: false } });

      await loadBusuanzi();

      const scripts = document.querySelectorAll('script[data-busuanzi]');
      expect(scripts.length).toBe(0);
    });

    it('loads script when enabled', async () => {
      const { __setMockConfig } = await import('../../src/config/loaders');
      __setMockConfig({ busuanzi: { enabled: true } });

      // Start loading
      const loadPromise = loadBusuanzi();

      // Script should be added to DOM immediately (before load completes)
      await new Promise((resolve) => setTimeout(resolve, 10));

      const scripts = document.querySelectorAll('script[data-busuanzi]');
      expect(scripts.length).toBe(1);

      // Trigger onload manually to complete the promise
      const script = scripts[0] as HTMLScriptElement;
      if (script.onload) {
        (script.onload as (event: Event) => void)(new Event('load'));
      }

      // Complete the load
      await loadPromise;
    });

    it('adds script tag with correct attributes', async () => {
      const { __setMockConfig } = await import('../../src/config/loaders');
      __setMockConfig({ busuanzi: { enabled: true } });

      const loadPromise = loadBusuanzi();

      // Wait for script to be added
      await new Promise((resolve) => setTimeout(resolve, 10));

      const script = document.querySelector('script[data-busuanzi]') as HTMLScriptElement;
      expect(script).toBeTruthy();
      expect(script.src).toContain('busuanzi');
      expect(script.async).toBe(true);
      expect(script.defer).toBe(true);

      // Trigger onload to complete
      if (script.onload) {
        (script.onload as (event: Event) => void)(new Event('load'));
      }

      await loadPromise;
    });

    it('handles script load error gracefully without throwing', async () => {
      const { __setMockConfig } = await import('../../src/config/loaders');
      __setMockConfig({ busuanzi: { enabled: true } });

      // Trigger script error by loading, then manually fire onerror
      const loadPromise = loadBusuanzi();

      // Find the script and trigger error
      await new Promise((resolve) => setTimeout(resolve, 10));
      const script = document.querySelector('script[data-busuanzi]') as HTMLScriptElement;
      if (script && script.onerror) {
        (script.onerror as (event: Event | string) => void)(new Event('error'));
      }

      // Should not throw
      await expect(loadPromise).resolves.toBeUndefined();
    });
  });

  describe('revealContainers', () => {
    it('reveals containers with valid values', async () => {
      // Create container elements
      const container = document.createElement('span');
      container.id = 'busuanzi_container_page_pv';
      container.style.display = 'none';
      document.body.appendChild(container);

      const value = document.createElement('span');
      value.id = 'busuanzi_value_page_pv';
      value.textContent = '123';
      container.appendChild(value);

      revealContainers();

      // Wait for setTimeout
      await new Promise((resolve) => setTimeout(resolve, 150));

      expect(container.style.display).toBe('');
    });

    it('does not reveal containers with empty values', async () => {
      const container = document.createElement('span');
      container.id = 'busuanzi_container_page_pv';
      container.style.display = 'none';
      document.body.appendChild(container);

      const value = document.createElement('span');
      value.id = 'busuanzi_value_page_pv';
      value.textContent = '';
      container.appendChild(value);

      revealContainers();

      await new Promise((resolve) => setTimeout(resolve, 150));

      expect(container.style.display).toBe('none');
    });
  });

  describe('initBusuanzi', () => {
    it('calls loadBusuanzi when in browser environment and enabled', async () => {
      const { __setMockConfig } = await import('../../src/config/loaders');
      __setMockConfig({ busuanzi: { enabled: true } });

      // Remove any existing scripts from previous tests
      document.querySelectorAll('script[data-busuanzi]').forEach((s) => s.remove());

      initBusuanzi();

      // Should add script tag after a short delay
      await new Promise((resolve) => setTimeout(resolve, 50));
      const scripts = document.querySelectorAll('script[data-busuanzi]');
      expect(scripts.length).toBeGreaterThan(0);

      // Trigger onload to complete
      const script = scripts[0] as HTMLScriptElement;
      if (script.onload) {
        (script.onload as (event: Event) => void)(new Event('load'));
      }
    });

    it('does not load script when disabled', async () => {
      const { __setMockConfig } = await import('../../src/config/loaders');
      __setMockConfig({ busuanzi: { enabled: false } });

      // Remove any existing scripts from previous tests
      document.querySelectorAll('script[data-busuanzi]').forEach((s) => s.remove());

      initBusuanzi();

      await new Promise((resolve) => setTimeout(resolve, 100));
      const newScripts = document.querySelectorAll('script[data-busuanzi]');
      expect(newScripts.length).toBe(0);
    });
  });
});
