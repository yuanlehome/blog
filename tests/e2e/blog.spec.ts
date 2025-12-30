import { expect, test } from '@playwright/test';

const ensureHashNavigation = async (page: any, tocSelector: string) => {
  const tocLink = page.locator(`${tocSelector} a`).first();
  await expect(tocLink).toBeVisible();
  const rawHash = await tocLink.getAttribute('href');
  await tocLink.click();

  if (!rawHash) return;

  const targetHash = rawHash.startsWith('#') ? rawHash : `#${rawHash}`;
  const encodedHash = `#${encodeURIComponent(targetHash.slice(1))}`;
  const escapeRegex = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  await expect(page).toHaveURL(new RegExp(`${escapeRegex(encodedHash)}$`));
  await expect
    .poll(async () => decodeURIComponent((await page.evaluate(() => location.hash)) || ''))
    .toBe(targetHash);
};

test.describe('Blog smoke journey', () => {
  test('home to article with interactions', async ({ page }) => {
    await page.goto('/');
    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if (await notFoundHeading.count()) {
      const baseLink = page.locator('a[href="/blog/"]');
      if (await baseLink.count()) {
        await baseLink.first().click();
      }
    }

    const posts = page.locator('#post-list li');
    expect(await posts.count()).toBeGreaterThan(0);

    const firstLink = page.locator('#post-list li a').first();
    await firstLink.click();
    await expect(page.locator('[data-article]')).toBeVisible();

    await ensureHashNavigation(page, 'aside nav[aria-label="文章目录"]');

    const codeBlock = page.locator('.code-block').first();
    if (await codeBlock.count()) {
      await expect(codeBlock).toBeVisible();
      const rawCode = await codeBlock.getAttribute('data-raw-code');
      const copyButton = codeBlock.locator('button.code-block__copy');
      await expect(copyButton).toBeVisible();
      await copyButton.click();

      // Wait for the button state to change to 'copied' or 'error'
      await expect.poll(async () => copyButton.getAttribute('data-state')).toBe('copied');

      const clipboard = await page.evaluate(() => navigator.clipboard.readText());
      expect(clipboard).toBe(rawCode || '');
    } else {
      test
        .info()
        .annotations.push({ type: 'todo', description: 'Code block not available on page' });
    }

    const html = page.locator('html');
    const initialClass = await html.getAttribute('class');
    await page.locator('#theme-toggle').click();
    const toggledClass = await html.getAttribute('class');
    expect(toggledClass).not.toBe(initialClass);

    const expectedGiscusTheme = await page.evaluate(() => {
      const resolved = document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
      return resolved === 'dark' ? 'dark_dimmed' : 'light';
    });
    await expect
      .poll(async () =>
        page.evaluate(
          () => document.getElementById('comments')?.getAttribute('data-giscus-theme') || '',
        ),
      )
      .toBe(expectedGiscusTheme);

    await page.goto('/blog/archive/');
    const toggledResolved = await page.evaluate(() =>
      document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light',
    );
    await page.waitForFunction(
      (expected) => document.documentElement.dataset.theme === expected,
      toggledResolved,
    );
    const persistedDataTheme = await page.evaluate(
      () => document.documentElement.dataset.theme || '',
    );
    expect(persistedDataTheme).toBe(toggledResolved);
  });

  test('post cover shows on desktop and hides on mobile', async ({ browser }) => {
    const baseURL = test.info().project.use.baseURL;
    const basePath = baseURL ? new URL(baseURL).pathname : '/';
    const buildUrl = (value: string) =>
      baseURL ? new URL(value, baseURL).toString() : `/${value.replace(/^\//, '')}`;
    const normalizeSlug = (href: string) => {
      const normalized = baseURL ? new URL(href, baseURL).pathname : href;
      const trimmed =
        basePath !== '/' && normalized.startsWith(basePath)
          ? normalized.slice(basePath.length)
          : normalized.replace(/^\//, '');
      if (!trimmed) return '';
      return trimmed.endsWith('/') ? trimmed : `${trimmed}/`;
    };
    const candidates = new Set<string>();
    if (process.env.PLAYWRIGHT_COVER_SLUG) {
      candidates.add(normalizeSlug(process.env.PLAYWRIGHT_COVER_SLUG));
    }
    let targetUrl: string | null = null;

    const desktopContext = await browser.newContext({
      viewport: { width: 1280, height: 900 },
      baseURL,
    });
    const desktopPage = await desktopContext.newPage();
    await desktopPage.goto(buildUrl(''));
    const discoveredSlugs = await desktopPage
      .locator('#post-list li a')
      .evaluateAll((anchors) => anchors.map((a) => a.getAttribute('href') || ''));
    discoveredSlugs
      .map(normalizeSlug)
      .filter(Boolean)
      .forEach((slug) => candidates.add(slug));

    for (const candidate of candidates) {
      const candidateUrl = buildUrl(candidate);
      await desktopPage.goto(candidateUrl);
      const desktopCover = desktopPage.locator('[data-post-cover]');
      if ((await desktopCover.count()) === 0) continue;
      await expect(desktopCover).toBeVisible();
      targetUrl = candidateUrl;
      break;
    }

    if (!targetUrl) {
      test.info().annotations.push({
        type: 'todo',
        description: 'No post with cover found for cover visibility test',
      });
      await desktopContext.close();
      return;
    }

    await desktopContext.close();

    const mobileContext = await browser.newContext({
      viewport: { width: 375, height: 812 },
      baseURL,
    });
    const mobilePage = await mobileContext.newPage();
    await mobilePage.goto(targetUrl);
    await expect(mobilePage.locator('[data-post-cover]')).toBeHidden();
    await mobileContext.close();
  });

  test('mobile viewport avoids horizontal overflow on flashattention page', async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 375, height: 812 } });
    const mobilePage = await context.newPage();
    await mobilePage.goto('/flashattention/');

    const [scrollWidth, clientWidth] = await mobilePage.evaluate(() => [
      document.documentElement.scrollWidth,
      document.documentElement.clientWidth,
    ]);

    expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 1);
    await context.close();
  });

  test('mobile toc navigates to heading', async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 375, height: 812 } });
    const page = await context.newPage();

    await page.goto('/');
    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if (await notFoundHeading.count()) {
      const baseLink = page.locator('a[href="/blog/"]');
      if (await baseLink.count()) {
        await baseLink.first().click();
      }
    }

    const firstLink = page.locator('#post-list li a').first();
    await firstLink.click();
    await expect(page.locator('[data-article]')).toBeVisible();

    const tocButton = page.locator('[data-action="toc"]');
    await expect(tocButton).toBeVisible();
    await tocButton.click();

    const toc = page.locator('[data-mobile-toc][data-open="true"]');
    await expect(toc).toBeVisible();

    const tocLink = toc.locator('[data-mobile-toc-link]').first();
    await expect(tocLink).toBeVisible();
    const rawHash = await tocLink.getAttribute('href');
    if (!rawHash) {
      await context.close();
      return;
    }
    const targetHash = rawHash.startsWith('#') ? rawHash : `#${rawHash}`;
    const targetId = targetHash.replace(/^#/, '');
    const encodedHash = `#${encodeURIComponent(targetId)}`;
    const escapeRegex = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

    await tocLink.click();

    await expect(page).toHaveURL(new RegExp(`${escapeRegex(encodedHash)}$`));
    await expect
      .poll(async () => decodeURIComponent((await page.evaluate(() => location.hash)) || ''))
      .toBe(targetHash);

    const headingBox = await page.locator(`#${targetId}`).boundingBox();
    const viewportHeight = await page.evaluate(() => window.innerHeight);
    expect(headingBox).toBeTruthy();
    if (headingBox) {
      expect(headingBox.y).toBeGreaterThanOrEqual(0);
      expect(headingBox.y).toBeLessThanOrEqual(viewportHeight - 40);
    }

    await context.close();
  });

  test('search modal opens with keyboard shortcut and navigates', async ({ page }) => {
    await page.goto('/');
    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if (await notFoundHeading.count()) {
      const baseLink = page.locator('a[href="/blog/"]');
      if (await baseLink.count()) {
        await baseLink.first().click();
      }
    }

    // Wait for page to be fully loaded
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(500);

    // Open search with Ctrl+K (or Meta+K on Mac)
    await page.keyboard.press('Control+k');

    // Wait for search modal to open
    const searchModal = page.locator('#search-modal[data-open="true"]');
    await expect(searchModal).toBeVisible({ timeout: 5000 });

    // Check that input is focused
    const searchInput = page.locator('#search-input');
    await expect(searchInput).toBeVisible();

    // Wait for search engine to initialize (loads index)
    await page.waitForTimeout(1000);

    // Type a search query
    await searchInput.fill('flash');

    // Wait for debounce and results
    await page.waitForTimeout(500);

    // Check if results are visible (may be empty depending on content)
    const resultsOrRecent = page.locator(
      '.search-modal__result, .search-modal__recent-list .search-modal__result',
    );

    // If there are results, test navigation
    const resultCount = await resultsOrRecent.count();
    if (resultCount > 0) {
      // Navigate with arrow keys
      await searchInput.focus();
      await page.keyboard.press('ArrowDown');
      const selectedResult = page.locator('.search-modal__result.is-selected');
      await expect(selectedResult).toBeVisible();
    }

    // Close with ESC
    await page.keyboard.press('Escape');
    await expect(searchModal).toBeHidden();
  });

  test('search trigger button opens modal', async ({ page }) => {
    await page.goto('/');
    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if (await notFoundHeading.count()) {
      const baseLink = page.locator('a[href="/blog/"]');
      if (await baseLink.count()) {
        await baseLink.first().click();
      }
    }

    // Wait for page to be fully loaded
    await page.waitForLoadState('domcontentloaded');

    // Click search trigger button
    const searchTrigger = page.locator('#search-trigger');
    await expect(searchTrigger).toBeVisible();
    await searchTrigger.click();

    // Wait for search modal to open
    const searchModal = page.locator('#search-modal[data-open="true"]');
    await expect(searchModal).toBeVisible({ timeout: 5000 });

    // Check input is visible
    const searchInput = page.locator('#search-input');
    await expect(searchInput).toBeVisible();

    // Close by clicking close button instead of backdrop (more reliable)
    const closeButton = page.locator('.search-modal__close');
    await closeButton.click();
    await expect(searchModal).toBeHidden();
  });

  test('mobile floating actions are vertically ordered and non-overlapping', async ({
    browser,
  }) => {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
    const page = await context.newPage();

    await page.goto('/');
    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if (await notFoundHeading.count()) {
      const baseLink = page.locator('a[href="/blog/"]');
      if (await baseLink.count()) {
        await baseLink.first().click();
      }
    }

    const firstLink = page.locator('#post-list li a').first();
    await firstLink.click();
    await expect(page.locator('[data-article]')).toBeVisible();

    await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));

    const stack = page.locator('[data-floating-action-stack]');
    await expect(stack).toBeVisible();

    const topButton = stack.locator('[data-action="top"]');
    const tocButton = stack.locator('[data-action="toc"]');
    const bottomButton = stack.locator('[data-action="bottom"]');

    await expect(topButton).toBeVisible();
    await expect(tocButton).toBeVisible();
    await expect(bottomButton).toBeVisible();

    const [topBox, tocBox, bottomBox] = await Promise.all([
      topButton.boundingBox(),
      tocButton.boundingBox(),
      bottomButton.boundingBox(),
    ]);

    expect(topBox && tocBox && bottomBox).toBeTruthy();
    if (topBox && tocBox && bottomBox) {
      expect(topBox.y + topBox.height).toBeLessThanOrEqual(tocBox.y - 1);
      expect(tocBox.y + tocBox.height).toBeLessThanOrEqual(bottomBox.y - 1);
    }

    await tocButton.click();
    await expect(page.locator('[data-mobile-toc][data-open="true"]')).toBeVisible();
    await page.locator('[data-mobile-toc-close]').click();
    await expect(page.locator('[data-mobile-toc][data-open="true"]')).toHaveCount(0);

    await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
    const articleTop = await page.evaluate(() => {
      const article = document.querySelector('[data-article]');
      if (!article) return null;
      const rect = article.getBoundingClientRect();
      return rect.top + window.scrollY;
    });
    await topButton.click();
    await expect
      .poll(async () => page.evaluate(() => window.scrollY), {
        message: 'top button should scroll toward article start',
      })
      .toBeLessThanOrEqual((articleTop ?? 0) + 10);

    await context.close();
  });

  test('pagination navigation works correctly', async ({ page }) => {
    await page.goto('/');
    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if (await notFoundHeading.count()) {
      const baseLink = page.locator('a[href="/blog/"]');
      if (await baseLink.count()) {
        await baseLink.first().click();
      }
    }

    // Check if pagination is present (only if more than 5 posts)
    const pagination = page.locator('nav[aria-label="Pagination"]');
    if ((await pagination.count()) === 0) {
      test.info().annotations.push({
        type: 'skip',
        description: 'Not enough posts for pagination',
      });
      return;
    }

    // Verify pagination structure on first page
    await expect(pagination).toBeVisible();

    // Check that "Newer" is disabled on first page
    const newerButton = pagination.locator('span[aria-disabled="true"]', {
      hasText: /← Newer/,
    });
    await expect(newerButton).toBeVisible();

    // Check current page indicator
    const currentPageIndicator = pagination.locator('span[aria-current="page"]');
    await expect(currentPageIndicator).toBeVisible();
    await expect(currentPageIndicator).toHaveText('1');

    // Check page links exist
    const pageLinks = pagination.locator('a[aria-label^="Go to page"]');
    const pageLinkCount = await pageLinks.count();
    expect(pageLinkCount).toBeGreaterThan(0);

    // Click on page 2 if it exists
    if (pageLinkCount >= 1) {
      const page2Link = pagination.locator('a[aria-label="Go to page 2"]');
      if ((await page2Link.count()) > 0) {
        await page2Link.click();

        // Verify we're on page 2
        await expect(page).toHaveURL(/page\/2\//);
        await expect(pagination.locator('span[aria-current="page"]')).toHaveText('2');

        // Verify "Newer" is now enabled
        const newerLinkOnPage2 = pagination.locator('a[aria-label="Go to previous page"]', {
          hasText: /← Newer/,
        });
        await expect(newerLinkOnPage2).toBeVisible();

        // Click "Older" if available
        const olderButton = pagination.locator('a[aria-label="Go to next page"]', {
          hasText: /Older →/,
        });
        if ((await olderButton.count()) > 0) {
          await olderButton.click();
          await expect(page).toHaveURL(/page\/3\//);
          await expect(pagination.locator('span[aria-current="page"]')).toHaveText('3');
        }

        // Navigate back to homepage
        const page1Link = pagination.locator('a[aria-label="Go to page 1"]');
        if ((await page1Link.count()) > 0) {
          await page1Link.click();
          await expect(page).toHaveURL(/\/$|\/blog\/$/);
          await expect(pagination.locator('span[aria-current="page"]')).toHaveText('1');
        }
      }
    }
  });
});
