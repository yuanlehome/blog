/**
 * Dynamic Header Height Tracking
 *
 * This script measures the site header height and updates a CSS variable
 * that can be used by sticky elements (like TOC) to align with the header bottom.
 *
 * - Uses ResizeObserver to handle dynamic header height changes
 * - Fallback to window resize events for broader browser support
 * - Sets --site-header-height CSS variable on document root
 */

function initHeaderHeightTracking() {
  const header = document.querySelector('[data-site-header]') as HTMLElement | null;

  if (!header) {
    // Fallback: set to 0 if header not found
    document.documentElement.style.setProperty('--site-header-height', '0px');
    return;
  }

  const updateHeaderHeight = () => {
    const height = header.getBoundingClientRect().height;
    document.documentElement.style.setProperty('--site-header-height', `${height}px`);
  };

  // Initial measurement
  updateHeaderHeight();

  // Use ResizeObserver for better tracking of content changes
  if ('ResizeObserver' in window) {
    const resizeObserver = new ResizeObserver(() => {
      updateHeaderHeight();
    });
    resizeObserver.observe(header);
  }

  // Fallback: also listen to window resize
  window.addEventListener('resize', updateHeaderHeight, { passive: true });

  // Update on load to ensure accuracy after all resources loaded
  window.addEventListener('load', updateHeaderHeight, { once: true });
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initHeaderHeightTracking);
} else {
  initHeaderHeightTracking();
}
