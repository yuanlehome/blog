/**
 * Tests for client ID generation and persistence
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { getClientId } from '../../src/lib/views/client-id';

describe('Client ID Generation', () => {
  // Store original values
  let originalLocalStorage: Storage | undefined;
  let originalSessionStorage: Storage | undefined;

  beforeEach(() => {
    // Save originals
    originalLocalStorage = global.localStorage;
    originalSessionStorage = global.sessionStorage;

    // Mock localStorage
    const localStorageMock = (() => {
      let store: Record<string, string> = {};
      return {
        getItem: (key: string) => store[key] || null,
        setItem: (key: string, value: string) => {
          store[key] = value;
        },
        removeItem: (key: string) => {
          delete store[key];
        },
        clear: () => {
          store = {};
        },
        get length() {
          return Object.keys(store).length;
        },
        key: (index: number) => Object.keys(store)[index] || null,
      };
    })();

    // Mock sessionStorage
    const sessionStorageMock = (() => {
      let store: Record<string, string> = {};
      return {
        getItem: (key: string) => store[key] || null,
        setItem: (key: string, value: string) => {
          store[key] = value;
        },
        removeItem: (key: string) => {
          delete store[key];
        },
        clear: () => {
          store = {};
        },
        get length() {
          return Object.keys(store).length;
        },
        key: (index: number) => Object.keys(store)[index] || null,
      };
    })();

    Object.defineProperty(global, 'localStorage', {
      value: localStorageMock,
      writable: true,
      configurable: true,
    });

    Object.defineProperty(global, 'sessionStorage', {
      value: sessionStorageMock,
      writable: true,
      configurable: true,
    });
  });

  afterEach(() => {
    // Restore originals
    if (originalLocalStorage) {
      Object.defineProperty(global, 'localStorage', {
        value: originalLocalStorage,
        writable: true,
        configurable: true,
      });
    }
    if (originalSessionStorage) {
      Object.defineProperty(global, 'sessionStorage', {
        value: originalSessionStorage,
        writable: true,
        configurable: true,
      });
    }
    vi.restoreAllMocks();
  });

  it('should generate a valid UUID format', () => {
    const clientId = getClientId();
    expect(clientId).toBeTruthy();
    expect(typeof clientId).toBe('string');
    // UUID v4 format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    expect(clientId).toMatch(
      /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i,
    );
  });

  it('should persist client ID in localStorage', () => {
    const clientId1 = getClientId();
    const clientId2 = getClientId();

    expect(clientId1).toBe(clientId2);
    expect(localStorage.getItem('blog_views_client_id')).toBe(clientId1);
  });

  it('should use existing client ID from localStorage', () => {
    const existingId = 'existing-client-id-12345';
    localStorage.setItem('blog_views_client_id', existingId);

    const clientId = getClientId();
    expect(clientId).toBe(existingId);
  });

  it('should fallback to sessionStorage when localStorage fails', () => {
    // Make localStorage unavailable
    Object.defineProperty(global, 'localStorage', {
      value: {
        getItem: () => {
          throw new Error('localStorage not available');
        },
        setItem: () => {
          throw new Error('localStorage not available');
        },
        removeItem: () => {
          throw new Error('localStorage not available');
        },
        clear: () => {
          throw new Error('localStorage not available');
        },
        length: 0,
        key: () => null,
      },
      writable: true,
      configurable: true,
    });

    const clientId1 = getClientId();
    const clientId2 = getClientId();

    expect(clientId1).toBe(clientId2);
    expect(sessionStorage.getItem('blog_views_client_id')).toBe(clientId1);
  });

  it('should generate new ID when both storages fail', () => {
    // Make both storages unavailable
    Object.defineProperty(global, 'localStorage', {
      value: {
        getItem: () => {
          throw new Error('localStorage not available');
        },
        setItem: () => {
          throw new Error('localStorage not available');
        },
        removeItem: () => {
          throw new Error('localStorage not available');
        },
        clear: () => {
          throw new Error('localStorage not available');
        },
        length: 0,
        key: () => null,
      },
      writable: true,
      configurable: true,
    });

    Object.defineProperty(global, 'sessionStorage', {
      value: {
        getItem: () => {
          throw new Error('sessionStorage not available');
        },
        setItem: () => {
          throw new Error('sessionStorage not available');
        },
        removeItem: () => {
          throw new Error('sessionStorage not available');
        },
        clear: () => {
          throw new Error('sessionStorage not available');
        },
        length: 0,
        key: () => null,
      },
      writable: true,
      configurable: true,
    });

    const clientId = getClientId();
    expect(clientId).toBeTruthy();
    expect(typeof clientId).toBe('string');
  });
});
