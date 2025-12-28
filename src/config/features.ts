import { boolFromEnv } from './env';

export type FeatureFlags = {
  enableGiscus: boolean;
  enableToc: boolean;
  enableImportArtifacts: boolean;
};

export const features: FeatureFlags = {
  enableGiscus: boolFromEnv('FEATURE_GISCUS', true),
  enableToc: boolFromEnv('FEATURE_TOC', true),
  enableImportArtifacts: boolFromEnv('FEATURE_IMPORT_ARTIFACTS', true),
};
