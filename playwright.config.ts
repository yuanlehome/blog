import { defineConfig, devices } from '@playwright/test';

const PORT = parseInt(process.env.PORT || '4173', 10);
const basePath = process.env.SITE_BASE ?? '/blog';
const normalizedBasePath = basePath.endsWith('/') ? basePath : `${basePath}/`;
const baseURL =
  process.env.PLAYWRIGHT_TEST_BASE_URL || `http://127.0.0.1:${PORT}${normalizedBasePath}`;

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 60_000,
  expect: {
    timeout: 10_000,
  },
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL,
    trace: 'on-first-retry',
    permissions: ['clipboard-read', 'clipboard-write'],
    viewport: { width: 1280, height: 720 },
    ignoreHTTPSErrors: true,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: `npm run preview -- --host --port ${PORT}`,
    url: baseURL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
