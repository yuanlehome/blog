/**
 * Profile configuration loader
 *
 * This module loads and validates the profile/about page configuration from profile.yml.
 * It provides settings for the about page including personal information, social links,
 * skills, and journey timeline.
 *
 * @module src/config/loaders/profile
 */

import { z } from 'zod';
import { loadConfig } from './base';
import profileConfigData from '../yaml/profile.yml';

/**
 * Social link schema
 */
export const socialLinkSchema = z.object({
  name: z.string().min(1),
  url: z.string().url(),
  icon: z.string().optional(),
  colorClass: z.string().optional(),
});

export type SocialLink = z.infer<typeof socialLinkSchema>;

/**
 * What I Do section schema
 */
export const whatIDoSchema = z.object({
  title: z.string().default('What I Do'),
  description: z.string(),
});

export type WhatIDo = z.infer<typeof whatIDoSchema>;

/**
 * Tech Stack section schema
 */
export const techStackSchema = z.object({
  title: z.string().default('Tech Stack'),
  skills: z.array(z.string()).default([]),
});

export type TechStack = z.infer<typeof techStackSchema>;

/**
 * Journey item schema
 */
export const journeyItemSchema = z.object({
  year: z.string(),
  role: z.string(),
  description: z.string(),
  color: z.string().optional(),
});

export type JourneyItem = z.infer<typeof journeyItemSchema>;

/**
 * Journey section schema
 */
export const journeySchema = z.object({
  title: z.string().default('My Journey'),
  items: z.array(journeyItemSchema).default([]),
});

export type Journey = z.infer<typeof journeySchema>;

/**
 * Profile configuration schema
 */
export const profileConfigSchema = z.object({
  name: z.string().min(1).default('Yuanle Liu'),
  bio: z.string().default('A passionate developer'),
  socialLinks: z.array(socialLinkSchema).default([]),
  whatIDo: whatIDoSchema,
  techStack: techStackSchema,
  journey: journeySchema,
});

export type ProfileConfig = z.infer<typeof profileConfigSchema>;

/**
 * Default profile configuration
 */
export const defaultProfileConfig: ProfileConfig = {
  name: 'Yuanle Liu',
  bio: 'A passionate developer',
  socialLinks: [],
  whatIDo: {
    title: 'What I Do',
    description: 'Building software',
  },
  techStack: {
    title: 'Tech Stack',
    skills: [],
  },
  journey: {
    title: 'My Journey',
    items: [],
  },
};

/**
 * Load and validate profile configuration
 *
 * @returns Validated profile configuration
 * @throws Error if configuration is invalid
 */
export function loadProfileConfig(): ProfileConfig {
  return loadConfig(profileConfigData, profileConfigSchema, 'profile.yml');
}

/**
 * Cached profile configuration instance
 */
let cachedConfig: ProfileConfig | null = null;

/**
 * Get profile configuration (cached)
 *
 * @returns Profile configuration
 */
export function getProfileConfig(): ProfileConfig {
  if (!cachedConfig) {
    cachedConfig = loadProfileConfig();
  }
  return cachedConfig;
}
