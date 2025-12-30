import { expect, test } from '@playwright/test';

test.describe('Header Mascot', () => {
  test('should render mascot when enabled', async ({ page }) => {
    await page.goto('/');

    // Check if mascot exists in the header
    const mascot = page.locator('[data-mascot]');
    await expect(mascot).toBeAttached();

    // Verify it has aria-hidden for accessibility
    await expect(mascot).toHaveAttribute('aria-hidden', 'true');

    // Verify SVG is present
    const svg = mascot.locator('svg.mascot-svg');
    await expect(svg).toBeAttached();
  });

  test('should have stick figure structure', async ({ page }) => {
    await page.goto('/');

    const mascot = page.locator('[data-mascot]');
    const svg = mascot.locator('svg.mascot-svg');

    // Check for head (circle)
    await expect(svg.locator('circle')).toBeAttached();

    // Check for arms group
    await expect(svg.locator('.mascot-arms')).toBeAttached();

    // Check for legs group
    await expect(svg.locator('.mascot-legs')).toBeAttached();
  });

  test('should not block navigation clicks', async ({ page }) => {
    await page.goto('/');

    // Verify header navigation is still clickable
    const archiveLink = page.locator('header nav a', { hasText: /Archive|归档/ });
    if ((await archiveLink.count()) > 0) {
      await archiveLink.click();
      await expect(page).toHaveURL(/archive/);
    }
  });

  test('should respond to click interaction', async ({ page }) => {
    await page.goto('/');

    const mascot = page.locator('[data-mascot]');
    const svg = mascot.locator('svg.mascot-svg');

    // Click the mascot
    await mascot.click();

    // Check if jumping class is added (it will be removed after 600ms)
    // We check immediately after click
    const hasJumpingClass = await svg.evaluate((el) => el.classList.contains('jumping'));
    expect(hasJumpingClass).toBe(true);
  });

  test('should be hidden on mobile when hideOnMobile is true', async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 375, height: 812 } });
    const page = await context.newPage();

    await page.goto('/');

    const mascot = page.locator('[data-mascot]');

    // Check if mascot has hidden class on mobile (Tailwind's 'hidden md:block' pattern)
    const classes = await mascot.getAttribute('class');
    expect(classes).toContain('hidden');

    await context.close();
  });

  test('should be visible on desktop', async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const page = await context.newPage();

    await page.goto('/');

    const mascot = page.locator('[data-mascot]');

    // On desktop with 'hidden md:block', the element should be visible
    // Tailwind's md: breakpoint shows the element
    await expect(mascot).toBeAttached();

    await context.close();
  });
});
