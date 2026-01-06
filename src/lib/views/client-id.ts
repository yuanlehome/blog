/**
 * Client ID generation and persistence for view counting
 * Uses localStorage with fallback to sessionStorage
 */

const CLIENT_ID_KEY = 'blog_views_client_id';

/**
 * Generate a simple UUID v4
 */
function generateUUID(): string {
  // Use crypto.randomUUID if available (modern browsers)
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }

  // Fallback: simple UUID generation
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Check if localStorage is available
 */
function isLocalStorageAvailable(): boolean {
  try {
    const testKey = '__test__';
    localStorage.setItem(testKey, testKey);
    localStorage.removeItem(testKey);
    return true;
  } catch {
    return false;
  }
}

/**
 * Check if sessionStorage is available
 */
function isSessionStorageAvailable(): boolean {
  try {
    const testKey = '__test__';
    sessionStorage.setItem(testKey, testKey);
    sessionStorage.removeItem(testKey);
    return true;
  } catch {
    return false;
  }
}

/**
 * Get or create a persistent client ID
 * Priority: localStorage > sessionStorage > in-memory
 */
export function getClientId(): string {
  // Try localStorage first
  if (isLocalStorageAvailable()) {
    const stored = localStorage.getItem(CLIENT_ID_KEY);
    if (stored) {
      return stored;
    }

    const newId = generateUUID();
    try {
      localStorage.setItem(CLIENT_ID_KEY, newId);
    } catch {
      // Failed to store, but still return the ID
    }
    return newId;
  }

  // Fallback to sessionStorage
  if (isSessionStorageAvailable()) {
    const stored = sessionStorage.getItem(CLIENT_ID_KEY);
    if (stored) {
      return stored;
    }

    const newId = generateUUID();
    try {
      sessionStorage.setItem(CLIENT_ID_KEY, newId);
    } catch {
      // Failed to store, but still return the ID
    }
    return newId;
  }

  // Final fallback: generate a new ID each time
  // This is not ideal but prevents crashes
  return generateUUID();
}
