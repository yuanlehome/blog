/**
 * PDF VL OCR Multi-Page Tests
 *
 * Tests for multi-page PDF merging functionality
 * Note: These tests run serially to avoid race conditions when manipulating fixtures
 */

import { describe, it, expect, vi } from 'vitest';
import fs from 'fs/promises';
import path from 'path';

// Mock logger
const mockLogger = {
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
};

describe.sequential('PDF VL OCR Multi-Page Support', () => {
  it('should merge multiple pages correctly', async () => {
    // Dynamically import to avoid issues with module resolution
    const { callLocalMockOcr } = await import('../../scripts/import/adapters/pdf_vl_ocr.js');

    // Temporarily replace the fixture file with our multi-page version
    const fixtureDir = path.join(process.cwd(), 'tests/fixtures/ocr');
    const originalPath = path.join(fixtureDir, 'paddle_mock.json');
    const multiPagePath = path.join(fixtureDir, 'paddle_mock_multipage.json');
    const backupPath = path.join(fixtureDir, 'paddle_mock.json.backup');

    // Backup original
    await fs.copyFile(originalPath, backupPath);

    try {
      // Replace with multi-page fixture
      await fs.copyFile(multiPagePath, originalPath);

      // Call the OCR function
      const result = await callLocalMockOcr(mockLogger as any);

      // Verify pages were processed
      expect(result.pagesProcessed).toBe(3);

      // Verify markdown was merged
      expect(result.markdown).toContain('Page 1 Title');
      expect(result.markdown).toContain('Page 2 Title');
      expect(result.markdown).toContain('Page 3 Title');

      // Verify images were merged
      expect(Object.keys(result.images).length).toBe(3);
      expect(result.images['img1.png']).toBe('https://example.com/images/page1-img1.png');
      expect(result.images['img2.png']).toBe('https://example.com/images/page2-img2.png');

      // Verify conflicting image was renamed
      expect(result.images['img1_page3.png']).toBe('https://example.com/images/page3-img1.png');

      // Verify the markdown references the renamed image
      expect(result.markdown).toContain('img1_page3.png');

      // Verify outputImages were merged
      expect(result.outputImages).toHaveLength(2);
      expect(result.outputImages).toContain('https://example.com/output/page1.png');
      expect(result.outputImages).toContain('https://example.com/output/page2.png');
    } finally {
      // Restore original
      await fs.copyFile(backupPath, originalPath);
      await fs.unlink(backupPath);
    }
  });

  it('should handle single page as before', async () => {
    const { callLocalMockOcr } = await import('../../scripts/import/adapters/pdf_vl_ocr.js');

    const result = await callLocalMockOcr(mockLogger as any);

    // Verify single page
    expect(result.pagesProcessed).toBe(1);

    // Verify content exists
    expect(result.markdown).toContain('Mock PDF Document Title');
    expect(result.markdown.length).toBeGreaterThan(100);

    // Verify images exist (or are empty)
    expect(result.images).toBeDefined();
  });

  it('should handle pages with no images', async () => {
    const { callLocalMockOcr } = await import('../../scripts/import/adapters/pdf_vl_ocr.js');

    // This test uses the original single-page fixture which has no images
    const result = await callLocalMockOcr(mockLogger as any);

    expect(result.pagesProcessed).toBe(1);
    expect(result.images).toEqual({});
    expect(result.outputImages).toEqual([]);
  });
});
