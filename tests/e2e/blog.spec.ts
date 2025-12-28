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
      const clipboard = await page.evaluate(() => navigator.clipboard.readText());
      expect(clipboard.trim()).toBe((rawCode || '').trim());
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

  test('search box is noted when available', async ({ page }) => {
    await page.goto('/');
    const searchInput = page.locator('input[type="search"]');
    if (await searchInput.count()) {
      await searchInput.first().fill('flash');
      await expect(searchInput.first()).toHaveValue(/flash/);
    } else {
      test.info().annotations.push({ type: 'todo', description: 'Search UI not yet implemented' });
    }
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
});
