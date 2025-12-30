/**
 * Adapter Registry
 *
 * Central registry for all site adapters with priority-based resolution
 */

import type { Adapter, AdapterRegistry } from './types.js';
import { zhihuAdapter } from './zhihu.js';
import { mediumAdapter } from './medium.js';
import { wechatAdapter } from './wechat.js';
import { othersAdapter } from './others.js';

/**
 * Default adapter registry implementation
 */
class DefaultAdapterRegistry implements AdapterRegistry {
  private adapters: Adapter[] = [];

  register(adapter: Adapter): void {
    // Skip if adapter is undefined (can happen during circular imports)
    if (!adapter || !adapter.id) {
      return;
    }

    // Remove existing adapter with same ID
    this.adapters = this.adapters.filter((a) => a.id !== adapter.id);

    // Add adapter (will be sorted by priority later)
    this.adapters.push(adapter);

    // Sort adapters: specific sites first, 'others' last
    this.adapters.sort((a, b) => {
      if (a.id === 'others') return 1;
      if (b.id === 'others') return -1;
      return 0;
    });
  }

  resolve(url: string): Adapter | null {
    for (const adapter of this.adapters) {
      if (adapter.canHandle(url)) {
        return adapter;
      }
    }
    return null;
  }

  getAll(): Adapter[] {
    return [...this.adapters];
  }

  getById(id: string): Adapter | null {
    return this.adapters.find((a) => a.id === id) || null;
  }
}

/**
 * Global adapter registry instance
 */
export const adapterRegistry = new DefaultAdapterRegistry();

/**
 * Register all default adapters
 */
function registerDefaultAdapters(): void {
  adapterRegistry.register(zhihuAdapter);
  adapterRegistry.register(mediumAdapter);
  adapterRegistry.register(wechatAdapter);
  adapterRegistry.register(othersAdapter);
}

// Auto-register default adapters on module load
registerDefaultAdapters();

/**
 * Resolve the best adapter for a given URL
 */
export function resolveAdapter(url: string): Adapter | null {
  return adapterRegistry.resolve(url);
}

/**
 * Register an adapter
 */
export function registerAdapter(adapter: Adapter): void {
  adapterRegistry.register(adapter);
}

/**
 * Get all registered adapters
 */
export function getAllAdapters(): Adapter[] {
  return adapterRegistry.getAll();
}

/**
 * Get adapter by ID
 */
export function getAdapterById(id: string): Adapter | null {
  return adapterRegistry.getById(id);
}

// Re-export adapters for direct use
export { zhihuAdapter, mediumAdapter, wechatAdapter, othersAdapter };
export * from './types.js';
