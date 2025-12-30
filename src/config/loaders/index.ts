/**
 * Configuration loaders index
 *
 * This module provides a unified interface to all configuration loaders.
 * Import configuration data from here to ensure type safety and validation.
 *
 * @example
 * ```typescript
 * import { getSiteConfig, getNavConfig } from '../config/loaders';
 *
 * const siteConfig = getSiteConfig();
 * const navConfig = getNavConfig();
 * ```
 *
 * @module src/config/loaders
 */

// Export all loader functions
export {
  loadSiteConfig,
  getSiteConfig,
  defaultSiteConfig,
  type SiteConfig,
} from './site';

export {
  loadNavConfig,
  getNavConfig,
  defaultNavConfig,
  type NavConfig,
  type NavMenuItem,
  type NavTheme,
} from './nav';

export {
  loadHomeConfig,
  getHomeConfig,
  defaultHomeConfig,
  type HomeConfig,
} from './home';

export {
  loadPostConfig,
  getPostConfig,
  defaultPostConfig,
  type PostConfig,
  type PostMetadata,
  type TocConfig,
  type FloatingActions,
  type ReadingProgress,
  type PrevNext,
  type RelatedPosts,
  type CommentsConfig,
  type GiscusConfig,
  type SourceAttribution,
} from './post';

export {
  loadThemeConfig,
  getThemeConfig,
  defaultThemeConfig,
  type ThemeConfig,
} from './theme';

export {
  loadProfileConfig,
  getProfileConfig,
  defaultProfileConfig,
  type ProfileConfig,
  type SocialLink,
  type WhatIDo,
  type TechStack,
  type Journey,
  type JourneyItem,
} from './profile';

// Export base utilities
export { loadConfig, loadConfigWithDefaults } from './base';
