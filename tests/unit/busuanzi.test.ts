import { describe, expect, it, beforeEach, afterEach } from 'vitest';
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

describe('Busuanzi Analytics', () => {
  let dom: JSDOM;
  let document: Document;
  let window: Window & typeof globalThis;
  let originalEnvEnabled: string | undefined;
  let originalEnvScriptUrl: string | undefined;
  let originalEnvDebug: string | undefined;

  beforeEach(() => {
    // Save original env values
    originalEnvEnabled = import.meta.env.PUBLIC_BUSUANZI_ENABLED;
    originalEnvScriptUrl = import.meta.env.PUBLIC_BUSUANZI_SCRIPT_URL;
    originalEnvDebug = import.meta.env.PUBLIC_BUSUANZI_DEBUG;

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

    // Reset env to defaults
    (import.meta.env as any).PUBLIC_BUSUANZI_ENABLED = '';
    (import.meta.env as any).PUBLIC_BUSUANZI_SCRIPT_URL = '';
    (import.meta.env as any).PUBLIC_BUSUANZI_DEBUG = '';
  });

  afterEach(() => {
    // Restore original env values
    (import.meta.env as any).PUBLIC_BUSUANZI_ENABLED = originalEnvEnabled;
    (import.meta.env as any).PUBLIC_BUSUANZI_SCRIPT_URL = originalEnvScriptUrl;
    (import.meta.env as any).PUBLIC_BUSUANZI_DEBUG = originalEnvDebug;
  });

  describe('isBusuanziEnabled', () => {
    it('returns false when PUBLIC_BUSUANZI_ENABLED is not set', () => {
      expect(isBusuanziEnabled()).toBe(false);
    });

    it('returns true when PUBLIC_BUSUANZI_ENABLED is "true"', () => {
      (import.meta.env as any).PUBLIC_BUSUANZI_ENABLED = 'true';
      expect(isBusuanziEnabled()).toBe(true);
    });

    it('returns true when PUBLIC_BUSUANZI_ENABLED is "1"', () => {
      (import.meta.env as any).PUBLIC_BUSUANZI_ENABLED = '1';
      expect(isBusuanziEnabled()).toBe(true);
    });

    it('returns false when PUBLIC_BUSUANZI_ENABLED is "false"', () => {
      (import.meta.env as any).PUBLIC_BUSUANZI_ENABLED = 'false';
      expect(isBusuanziEnabled()).toBe(false);
    });
  });

  describe('getBusuanziScriptUrl', () => {
    it('returns default URL when PUBLIC_BUSUANZI_SCRIPT_URL is not set', () => {
      expect(getBusuanziScriptUrl()).toBe(
        'https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js',
      );
    });

    it('returns custom URL when PUBLIC_BUSUANZI_SCRIPT_URL is set', () => {
      const customUrl = 'https://example.com/busuanzi.js';
      (import.meta.env as any).PUBLIC_BUSUANZI_SCRIPT_URL = customUrl;
      expect(getBusuanziScriptUrl()).toBe(customUrl);
    });
  });

  describe('isBusuanziDebugEnabled', () => {
    it('returns false when PUBLIC_BUSUANZI_DEBUG is not set', () => {
      expect(isBusuanziDebugEnabled()).toBe(false);
    });

    it('returns true when PUBLIC_BUSUANZI_DEBUG is "true"', () => {
      (import.meta.env as any).PUBLIC_BUSUANZI_DEBUG = 'true';
      expect(isBusuanziDebugEnabled()).toBe(true);
    });
  });

  describe('loadBusuanzi', () => {
    it('does not load script when Busuanzi is disabled', async () => {
      (import.meta.env as any).PUBLIC_BUSUANZI_ENABLED = 'false';

      await loadBusuanzi();

      const scripts = document.querySelectorAll('script[data-busuanzi]');
      expect(scripts.length).toBe(0);
    });

    it('loads script when enabled', async () => {
      (import.meta.env as any).PUBLIC_BUSUANZI_ENABLED = 'true';

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
      (import.meta.env as any).PUBLIC_BUSUANZI_ENABLED = 'true';

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
      (import.meta.env as any).PUBLIC_BUSUANZI_ENABLED = 'true';

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
      (import.meta.env as any).PUBLIC_BUSUANZI_ENABLED = 'true';

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
      (import.meta.env as any).PUBLIC_BUSUANZI_ENABLED = 'false';

      // Remove any existing scripts from previous tests
      document.querySelectorAll('script[data-busuanzi]').forEach((s) => s.remove());

      initBusuanzi();

      await new Promise((resolve) => setTimeout(resolve, 100));
      const newScripts = document.querySelectorAll('script[data-busuanzi]');
      expect(newScripts.length).toBe(0);
    });
  });
});
