/**
 * Busuanzi (不蒜子) analytics integration for page view statistics
 * Provides functions to load the Busuanzi script and handle view counts
 */

import { getSiteConfig } from '../../config/loaders';

/**
 * Check if Busuanzi is enabled via configuration
 */
export function isBusuanziEnabled(): boolean {
  const siteConfig = getSiteConfig();
  return siteConfig.busuanzi?.enabled ?? false;
}

/**
 * Get the Busuanzi script URL from configuration
 */
export function getBusuanziScriptUrl(): string {
  const siteConfig = getSiteConfig();
  return (
    siteConfig.busuanzi?.scriptUrl ??
    'https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js'
  );
}

/**
 * Check if debug mode is enabled
 */
export function isBusuanziDebugEnabled(): boolean {
  const siteConfig = getSiteConfig();
  return siteConfig.busuanzi?.debug ?? false;
}

/**
 * Log debug messages if debug mode is enabled
 */
function debugLog(...args: unknown[]): void {
  if (isBusuanziDebugEnabled()) {
    console.log('[Busuanzi]', ...args);
  }
}

/**
 * Log warning messages
 */
function warnLog(...args: unknown[]): void {
  console.warn('[Busuanzi]', ...args);
}

// Track if script is already loaded or loading
let scriptLoadState: 'idle' | 'loading' | 'loaded' | 'error' = 'idle';
let scriptLoadPromise: Promise<void> | null = null;

/**
 * Reset the script load state (for testing purposes)
 * @internal
 */
export function resetBusuanziState(): void {
  scriptLoadState = 'idle';
  scriptLoadPromise = null;
}

/**
 * Load the Busuanzi script dynamically
 * This function is idempotent - calling it multiple times will only load the script once
 * @returns Promise that resolves when script is loaded or rejects on error
 */
export async function loadBusuanzi(): Promise<void> {
  // Check if Busuanzi is enabled
  if (!isBusuanziEnabled()) {
    debugLog('Busuanzi is disabled via configuration');
    return;
  }

  // If already loaded, return immediately
  if (scriptLoadState === 'loaded') {
    debugLog('Script already loaded');
    return;
  }

  // If currently loading, return the existing promise
  if (scriptLoadState === 'loading' && scriptLoadPromise) {
    debugLog('Script is already loading, waiting...');
    return scriptLoadPromise;
  }

  // If previous load failed, reset and try again
  if (scriptLoadState === 'error') {
    scriptLoadState = 'idle';
    scriptLoadPromise = null;
  }

  // Check if script already exists in DOM
  const existingScript = document.querySelector('script[data-busuanzi]');
  if (existingScript) {
    scriptLoadState = 'loaded';
    debugLog('Script already exists in DOM');
    revealContainers();
    return;
  }

  // Start loading
  scriptLoadState = 'loading';
  scriptLoadPromise = new Promise<void>((resolve, reject) => {
    const script = document.createElement('script');
    script.src = getBusuanziScriptUrl();
    script.async = true;
    script.defer = true;
    script.setAttribute('data-busuanzi', 'true');

    script.onload = () => {
      scriptLoadState = 'loaded';
      debugLog('Script loaded successfully');
      revealContainers();
      resolve();
    };

    script.onerror = (error) => {
      scriptLoadState = 'error';
      warnLog('Failed to load script:', error);
      // Don't reject - gracefully degrade
      resolve();
    };

    document.head.appendChild(script);
    debugLog('Script injection started');
  });

  return scriptLoadPromise;
}

/**
 * Reveal Busuanzi containers after values are populated
 * This function makes containers visible once the script has filled in the values
 */
export function revealContainers(): void {
  // Wait a bit for Busuanzi to populate values
  setTimeout(() => {
    const containers = [
      'busuanzi_container_page_pv',
      'busuanzi_container_site_pv',
      'busuanzi_container_site_uv',
    ];

    containers.forEach((containerId) => {
      const container = document.getElementById(containerId);
      const valueElement = document.getElementById(containerId.replace('_container_', '_value_'));

      if (container && valueElement) {
        // Check if value has been populated (not empty and not the default placeholder)
        const value = valueElement.textContent?.trim();
        if (value && value !== '' && value !== '--') {
          container.style.display = '';
          debugLog(`Revealed container: ${containerId}, value: ${value}`);
        } else {
          debugLog(`Container ${containerId} has no value yet`);
        }
      }
    });
  }, 100);
}

/**
 * Client-side initialization function
 * Call this when the page loads or when navigating to a new page (for SPA behavior)
 */
export function initBusuanzi(): void {
  if (typeof window === 'undefined') {
    return; // Skip on server-side
  }

  loadBusuanzi().catch((error) => {
    warnLog('Failed to initialize Busuanzi:', error);
  });
}
