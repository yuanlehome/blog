/**
 * Base configuration loader utilities
 *
 * This module provides common utilities for loading and validating YAML configuration files.
 * All configuration loaders should use these utilities to ensure consistent error handling
 * and validation.
 *
 * @module src/config/loaders/base
 */

import { z, type ZodSchema } from 'zod';

/**
 * Configuration load result
 */
export interface ConfigLoadResult<T> {
  data: T;
  errors: string[];
}

/**
 * Load and validate a YAML configuration module
 *
 * @param configModule - The imported YAML module (must be imported with `import` statement)
 * @param schema - Zod schema for validation
 * @param configName - Name of the configuration for error messages
 * @returns Validated configuration data
 * @throws Error if validation fails
 */
export function loadConfig<T>(
  configModule: unknown,
  schema: ZodSchema<T>,
  configName: string,
): T {
  try {
    // Parse and validate the configuration
    const result = schema.safeParse(configModule);

    if (!result.success) {
      const errors = result.error.errors
        .map((err) => `  - ${err.path.join('.')}: ${err.message}`)
        .join('\n');

      throw new Error(
        `Invalid configuration in ${configName}:\n${errors}\n\nPlease check your YAML file and fix the errors above.`,
      );
    }

    return result.data;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error(`Failed to load ${configName}: ${String(error)}`);
  }
}

/**
 * Load and validate a configuration with fallback to defaults
 *
 * @param configModule - The imported YAML module
 * @param schema - Zod schema for validation
 * @param defaults - Default values to use if loading fails
 * @param configName - Name of the configuration for error messages
 * @returns Validated configuration data or defaults
 */
export function loadConfigWithDefaults<T>(
  configModule: unknown,
  schema: ZodSchema<T>,
  defaults: T,
  configName: string,
): T {
  try {
    return loadConfig(configModule, schema, configName);
  } catch (error) {
    console.warn(`Warning: ${configName} validation failed, using defaults:`, error);
    return defaults;
  }
}
