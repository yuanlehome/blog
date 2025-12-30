/**
 * Example integration: Content Import with Logging
 *
 * This demonstrates how to wrap an existing script with the unified logging system.
 */

import { createScriptLogger, now, duration } from '../logger-helpers.js';

async function exampleContentImportWithLogging() {
  const scriptStart = now();
  const logger = createScriptLogger('content-import-example', {
    url: 'https://example.com/article',
    provider: 'example',
  });

  logger.info('Starting content import');

  // Example: Fetch article
  const fetchSpan = logger.time('fetch-article');
  try {
    // Simulate fetch
    await new Promise((resolve) => setTimeout(resolve, 1000));
    fetchSpan.end({ status: 'ok', fields: { size: 12345 } });
  } catch (error: any) {
    fetchSpan.end({ status: 'fail' });
    logger.error(error);
    throw error;
  }

  // Example: Process markdown
  const processSpan = logger.time('markdown-processing');
  try {
    await new Promise((resolve) => setTimeout(resolve, 500));
    processSpan.end({
      status: 'ok',
      fields: {
        translated: true,
        codeFencesFixed: 3,
        imageCaptionsFixed: 2,
      },
    });
  } catch (error: any) {
    processSpan.end({ status: 'fail' });
    logger.error(error);
  }

  // Example: Save file
  logger.info('Saving article', { filepath: 'content/blog/example.md' });

  // Summary
  logger.summary({
    status: 'ok',
    durationMs: duration(scriptStart),
    files: ['content/blog/example.md'],
    stats: {
      images: 5,
      codeBlocks: 3,
      translated: 1,
    },
  });
}

// Run example
if (import.meta.url === `file://${process.argv[1]}`) {
  exampleContentImportWithLogging().catch((error) => {
    console.error('Failed:', error);
    process.exit(1);
  });
}

export { exampleContentImportWithLogging };
