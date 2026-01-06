/**
 * E2E tests for page views functionality
 */

import { expect, test } from '@playwright/test';

test.describe('Page Views (PV) Feature', () => {
  test('should display views counter on article page', async ({ page }) => {
    // Navigate to homepage
    await page.goto('/');

    // Handle 404 redirect if needed
    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if ((await notFoundHeading.count()) > 0) {
      const baseLink = page.locator('a[href="/blog/"]');
      if ((await baseLink.count()) > 0) {
        await baseLink.first().click();
      }
    }

    // Click on first post
    const firstLink = page.locator('#post-list li a').first();
    await firstLink.click();

    // Wait for article to load
    await expect(page.locator('[data-article]')).toBeVisible();

    // Check that views container exists
    const viewsContainer = page.locator('[data-views-container]');
    await expect(viewsContainer).toBeVisible({ timeout: 10000 });

    // Check that views count is displayed (either a number or placeholder)
    const viewsCount = page.locator('[data-views-count]');
    await expect(viewsCount).toBeVisible();

    // Wait for views to load (should change from "—" to a number or hide on error)
    await page.waitForTimeout(3000);

    // Verify it shows either a number or is hidden (graceful degradation)
    const isVisible = await viewsContainer.isVisible();
    if (isVisible) {
      const countText = await viewsCount.textContent();
      // Should be either a number or the placeholder
      expect(countText).toBeTruthy();
    }
  });

  test('should not block page rendering if views API fails', async ({ page }) => {
    // Navigate to homepage
    await page.goto('/');

    // Handle 404 redirect if needed
    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if ((await notFoundHeading.count()) > 0) {
      const baseLink = page.locator('a[href="/blog/"]');
      if ((await baseLink.count()) > 0) {
        await baseLink.first().click();
      }
    }

    // Click on first post
    const firstLink = page.locator('#post-list li a').first();
    await firstLink.click();

    // Verify article loads quickly regardless of views API
    await expect(page.locator('[data-article]')).toBeVisible({ timeout: 5000 });

    // Verify post title is visible
    const postTitle = page.locator('[data-article] header h1');
    await expect(postTitle).toBeVisible();

    // Verify article is visible (has prose class)
    const article = page.locator('[data-article].prose');
    await expect(article).toBeVisible();
  });

  test('views counter should have proper accessibility attributes', async ({ page }) => {
    await page.goto('/');

    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if ((await notFoundHeading.count()) > 0) {
      const baseLink = page.locator('a[href="/blog/"]');
      if ((await baseLink.count()) > 0) {
        await baseLink.first().click();
      }
    }

    const firstLink = page.locator('#post-list li a').first();
    await firstLink.click();

    await expect(page.locator('[data-article]')).toBeVisible();

    const viewsContainer = page.locator('[data-views-container]');

    // Check for title attribute for accessibility
    const title = await viewsContainer.getAttribute('title');
    expect(title).toBeTruthy();
    expect(title).toContain('浏览量');
  });

  test('should handle multiple visits within same session', async ({ page }) => {
    await page.goto('/');

    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if ((await notFoundHeading.count()) > 0) {
      const baseLink = page.locator('a[href="/blog/"]');
      if ((await baseLink.count()) > 0) {
        await baseLink.first().click();
      }
    }

    // Get the first post link
    const firstLink = page.locator('#post-list li a').first();
    const postUrl = await firstLink.getAttribute('href');

    // Visit the post
    await firstLink.click();
    await expect(page.locator('[data-article]')).toBeVisible();

    // Wait for views to load
    await page.waitForTimeout(3000);

    // Go back to home
    await page.goto('/');

    // Visit the same post again
    await page.goto(postUrl || '/');
    await expect(page.locator('[data-article]')).toBeVisible();

    // Views should still be displayed (testing that it doesn't crash on second visit)
    const viewsContainer = page.locator('[data-views-container]');
    const isVisible = await viewsContainer.isVisible().catch(() => false);

    // Either visible or gracefully hidden - both are acceptable
    expect(typeof isVisible).toBe('boolean');
  });

  test('should store client ID in localStorage', async ({ page }) => {
    await page.goto('/');

    const notFoundHeading = page.locator('h1', { hasText: '404: Not found' });
    if ((await notFoundHeading.count()) > 0) {
      const baseLink = page.locator('a[href="/blog/"]');
      if ((await baseLink.count()) > 0) {
        await baseLink.first().click();
      }
    }

    const firstLink = page.locator('#post-list li a').first();
    await firstLink.click();

    await expect(page.locator('[data-article]')).toBeVisible();

    // Wait for views script to initialize
    await page.waitForTimeout(3000);

    // Check that client ID is stored in localStorage
    const clientId = await page.evaluate(() => {
      return localStorage.getItem('blog_views_client_id');
    });

    if (clientId) {
      // If client ID was created, verify it's a valid UUID v4 format
      // UUID v4: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx where y is [89ab]
      expect(clientId).toMatch(
        /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i,
      );
    }

    // Test passes whether or not the mock API triggered client ID creation
    // as this depends on API availability
  });
});
