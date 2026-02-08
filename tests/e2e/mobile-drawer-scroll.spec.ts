import { expect, test } from '@playwright/test';

/**
 * E2E tests for mobile drawer scroll position behavior
 *
 * These tests verify that opening and closing mobile drawers (TOC and CommonLinks)
 * does not cause the main page scroll position to jump or animate unexpectedly.
 *
 * Tests cover:
 * - Scroll position preservation when opening/closing TOC drawer
 * - Scroll position preservation when opening/closing CommonLinks drawer
 * - Background scroll lock when drawer is open
 * - iOS Safari compatibility (via mobile viewport simulation)
 */

test.describe('Mobile drawer scroll behavior', () => {
  // Use mobile viewport to simulate iOS Safari
  test.use({
    viewport: { width: 375, height: 667 }, // iPhone SE dimensions
    isMobile: true,
    hasTouch: true,
  });

  test('TOC drawer preserves scroll position on open/close', async ({ page }) => {
    // Navigate to a post with TOC
    await page.goto('/');
    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if (await notFoundHeading.count()) {
      const baseLink = page.locator('a[href="/blog/"]');
      if (await baseLink.count()) {
        await baseLink.first().click();
      }
    }

    // Find and navigate to a post with TOC
    const postLinks = await page
      .locator('#post-list li h3 a')
      .evaluateAll((anchors: HTMLAnchorElement[]) =>
        anchors
          .map((anchor: HTMLAnchorElement) => anchor.getAttribute('href') || '')
          .filter(Boolean),
      );

    let foundPostWithToc = false;
    for (const href of postLinks) {
      await page.goto(href);
      await expect(page.locator('[data-article]')).toBeVisible();

      // Check if mobile TOC button exists
      const tocButton = page.locator('[data-mobile-toc-open]');
      if (await tocButton.count()) {
        foundPostWithToc = true;
        break;
      }
    }

    if (!foundPostWithToc) {
      test.info().annotations.push({
        type: 'skip',
        description: 'No post with TOC found for mobile drawer test',
      });
      return;
    }

    // Scroll down to a specific position
    const targetScrollY = 800;
    await page.evaluate((y) => window.scrollTo(0, y), targetScrollY);

    // Wait for scroll to complete
    await page.waitForTimeout(100);

    // Record scroll position before opening drawer
    const scrollBeforeOpen = await page.evaluate(() => window.scrollY);
    expect(scrollBeforeOpen).toBeGreaterThan(500); // Ensure we actually scrolled

    // Open TOC drawer
    const tocButton = page.locator('[data-mobile-toc-open]');
    await tocButton.click();

    // Wait for drawer to open
    await expect(page.locator('[data-mobile-toc-drawer][data-open="true"]')).toBeVisible();

    // Verify scroll position hasn't changed after opening
    const scrollAfterOpen = await page.evaluate(() => window.scrollY);
    expect(Math.abs(scrollAfterOpen - scrollBeforeOpen)).toBeLessThanOrEqual(2); // Allow 2px tolerance

    // Close TOC drawer
    const closeButton = page.locator('[data-mobile-toc-close]');
    await closeButton.click();

    // Wait for drawer to close
    await expect(page.locator('[data-mobile-toc-drawer][data-open="false"]')).toBeVisible();

    // Wait for any potential scroll animations
    await page.waitForTimeout(100);

    // Verify scroll position is preserved after closing
    const scrollAfterClose = await page.evaluate(() => window.scrollY);
    expect(Math.abs(scrollAfterClose - scrollBeforeOpen)).toBeLessThanOrEqual(2); // Allow 2px tolerance
  });

  test('CommonLinks drawer preserves scroll position on open/close', async ({ page }) => {
    // Navigate to home page
    await page.goto('/');
    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if (await notFoundHeading.count()) {
      const baseLink = page.locator('a[href="/blog/"]');
      if (await baseLink.count()) {
        await baseLink.first().click();
      }
    }

    // Check if CommonLinks button exists
    const commonLinksButton = page.locator('[data-mobile-common-links-open]');
    if (!(await commonLinksButton.count())) {
      test.info().annotations.push({
        type: 'skip',
        description: 'CommonLinks drawer not available',
      });
      return;
    }

    // Scroll down to a specific position
    const targetScrollY = 600;
    await page.evaluate((y) => window.scrollTo(0, y), targetScrollY);

    // Wait for scroll to complete
    await page.waitForTimeout(100);

    // Record scroll position before opening drawer
    const scrollBeforeOpen = await page.evaluate(() => window.scrollY);
    expect(scrollBeforeOpen).toBeGreaterThan(400); // Ensure we actually scrolled

    // Open CommonLinks drawer
    await commonLinksButton.click();

    // Wait for drawer to open
    await expect(page.locator('[data-mobile-common-links-drawer][data-open="true"]')).toBeVisible();

    // Verify scroll position hasn't changed after opening
    const scrollAfterOpen = await page.evaluate(() => window.scrollY);
    expect(Math.abs(scrollAfterOpen - scrollBeforeOpen)).toBeLessThanOrEqual(2); // Allow 2px tolerance

    // Close CommonLinks drawer
    const closeButton = page.locator('[data-mobile-common-links-close]');
    await closeButton.click();

    // Wait for drawer to close
    await expect(
      page.locator('[data-mobile-common-links-drawer][data-open="false"]'),
    ).toBeVisible();

    // Wait for any potential scroll animations
    await page.waitForTimeout(100);

    // Verify scroll position is preserved after closing
    const scrollAfterClose = await page.evaluate(() => window.scrollY);
    expect(Math.abs(scrollAfterClose - scrollBeforeOpen)).toBeLessThanOrEqual(2); // Allow 2px tolerance
  });

  test('Background scroll is locked when TOC drawer is open', async ({ page }) => {
    // Navigate to a post with TOC
    await page.goto('/');
    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if (await notFoundHeading.count()) {
      const baseLink = page.locator('a[href="/blog/"]');
      if (await baseLink.count()) {
        await baseLink.first().click();
      }
    }

    // Find and navigate to a post with TOC
    const postLinks = await page
      .locator('#post-list li h3 a')
      .evaluateAll((anchors: HTMLAnchorElement[]) =>
        anchors
          .map((anchor: HTMLAnchorElement) => anchor.getAttribute('href') || '')
          .filter(Boolean),
      );

    let foundPostWithToc = false;
    for (const href of postLinks) {
      await page.goto(href);
      await expect(page.locator('[data-article]')).toBeVisible();

      const tocButton = page.locator('[data-mobile-toc-open]');
      if (await tocButton.count()) {
        foundPostWithToc = true;
        break;
      }
    }

    if (!foundPostWithToc) {
      test.info().annotations.push({
        type: 'skip',
        description: 'No post with TOC found for scroll lock test',
      });
      return;
    }

    // Scroll down
    await page.evaluate(() => window.scrollTo(0, 800));
    await page.waitForTimeout(100);

    // Open TOC drawer
    const tocButton = page.locator('[data-mobile-toc-open]');
    await tocButton.click();
    await expect(page.locator('[data-mobile-toc-drawer][data-open="true"]')).toBeVisible();

    // Record scroll position when drawer is open
    const scrollWhenOpen = await page.evaluate(() => window.scrollY);

    // Try to scroll the page (should be locked)
    await page.evaluate(() => window.scrollBy(0, 200));
    await page.waitForTimeout(100);

    // Verify scroll position hasn't changed (background is locked)
    const scrollAfterAttempt = await page.evaluate(() => window.scrollY);
    expect(scrollAfterAttempt).toBe(scrollWhenOpen);

    // Close drawer
    const closeButton = page.locator('[data-mobile-toc-close]');
    await closeButton.click();
    await expect(page.locator('[data-mobile-toc-drawer][data-open="false"]')).toBeVisible();
  });

  test('Drawer can scroll internally while background is locked', async ({ page }) => {
    // Navigate to a post with TOC
    await page.goto('/');
    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if (await notFoundHeading.count()) {
      const baseLink = page.locator('a[href="/blog/"]');
      if (await baseLink.count()) {
        await baseLink.first().click();
      }
    }

    // Find and navigate to a post with long TOC
    const postLinks = await page
      .locator('#post-list li h3 a')
      .evaluateAll((anchors: HTMLAnchorElement[]) =>
        anchors
          .map((anchor: HTMLAnchorElement) => anchor.getAttribute('href') || '')
          .filter(Boolean),
      );

    let foundPostWithToc = false;
    for (const href of postLinks) {
      await page.goto(href);
      await expect(page.locator('[data-article]')).toBeVisible();

      const tocButton = page.locator('[data-mobile-toc-open]');
      if (await tocButton.count()) {
        foundPostWithToc = true;
        break;
      }
    }

    if (!foundPostWithToc) {
      test.info().annotations.push({
        type: 'skip',
        description: 'No post with TOC found for drawer scroll test',
      });
      return;
    }

    // Open TOC drawer
    const tocButton = page.locator('[data-mobile-toc-open]');
    await tocButton.click();
    await expect(page.locator('[data-mobile-toc-drawer][data-open="true"]')).toBeVisible();

    // Check if drawer content is scrollable
    const drawerScrollContainer = page.locator('[data-mobile-toc-scroll]');
    const isScrollable = await drawerScrollContainer.evaluate(
      (el) => el.scrollHeight > el.clientHeight,
    );

    if (isScrollable) {
      // Get initial scroll position of drawer
      const initialDrawerScroll = await drawerScrollContainer.evaluate((el) => el.scrollTop);

      // Scroll within drawer
      await drawerScrollContainer.evaluate((el) => el.scrollBy(0, 50));
      await page.waitForTimeout(100);

      // Verify drawer scrolled
      const newDrawerScroll = await drawerScrollContainer.evaluate((el) => el.scrollTop);
      expect(newDrawerScroll).toBeGreaterThan(initialDrawerScroll);
    }

    // Close drawer
    const closeButton = page.locator('[data-mobile-toc-close]');
    await closeButton.click();
  });

  test('Esc key closes drawer without scroll jump', async ({ page }) => {
    // Navigate to a post with TOC
    await page.goto('/');
    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if (await notFoundHeading.count()) {
      const baseLink = page.locator('a[href="/blog/"]');
      if (await baseLink.count()) {
        await baseLink.first().click();
      }
    }

    // Find and navigate to a post with TOC
    const postLinks = await page
      .locator('#post-list li h3 a')
      .evaluateAll((anchors: HTMLAnchorElement[]) =>
        anchors
          .map((anchor: HTMLAnchorElement) => anchor.getAttribute('href') || '')
          .filter(Boolean),
      );

    let foundPostWithToc = false;
    for (const href of postLinks) {
      await page.goto(href);
      await expect(page.locator('[data-article]')).toBeVisible();

      const tocButton = page.locator('[data-mobile-toc-open]');
      if (await tocButton.count()) {
        foundPostWithToc = true;
        break;
      }
    }

    if (!foundPostWithToc) {
      test.info().annotations.push({
        type: 'skip',
        description: 'No post with TOC found for Esc key test',
      });
      return;
    }

    // Scroll down
    const targetScrollY = 700;
    await page.evaluate((y) => window.scrollTo(0, y), targetScrollY);
    await page.waitForTimeout(100);

    const scrollBeforeOpen = await page.evaluate(() => window.scrollY);

    // Open TOC drawer
    const tocButton = page.locator('[data-mobile-toc-open]');
    await tocButton.click();
    await expect(page.locator('[data-mobile-toc-drawer][data-open="true"]')).toBeVisible();

    // Close with Esc key
    await page.keyboard.press('Escape');
    await expect(page.locator('[data-mobile-toc-drawer][data-open="false"]')).toBeVisible();

    // Wait for any potential scroll animations
    await page.waitForTimeout(100);

    // Verify scroll position is preserved
    const scrollAfterClose = await page.evaluate(() => window.scrollY);
    expect(Math.abs(scrollAfterClose - scrollBeforeOpen)).toBeLessThanOrEqual(2);
  });
});
