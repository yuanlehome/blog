import { expect, test } from '@playwright/test';

const ensureHashNavigation = async (page: any, tocSelector: string) => {
  const tocLink = page.locator(`${tocSelector} a`).first();
  await expect(tocLink).toBeVisible();
  const hash = await tocLink.getAttribute('href');
  await tocLink.click();
  if (hash) {
    await expect(page).toHaveURL(new RegExp(`${hash.replace('#', '#')}$`));
  }
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
    await expect(codeBlock).toBeVisible();
    const rawCode = await codeBlock.getAttribute('data-raw-code');
    const copyButton = codeBlock.locator('button.code-block__copy');
    await expect(copyButton).toBeVisible();
    await copyButton.click();
    const clipboard = await page.evaluate(() => navigator.clipboard.readText());
    expect(clipboard.trim()).toBe((rawCode || '').trim());

    const html = page.locator('html');
    const initialClass = await html.getAttribute('class');
    await page.locator('#theme-toggle').click();
    const toggledClass = await html.getAttribute('class');
    expect(toggledClass).not.toBe(initialClass);

    await page.goto('/blog/archive/');
    const toggledResolved = toggledClass ? 'dark' : 'light';
    await page.waitForFunction((expected) => document.documentElement.dataset.theme === expected, toggledResolved);
    const persistedDataTheme = await page.evaluate(() => document.documentElement.dataset.theme || '');
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
});
