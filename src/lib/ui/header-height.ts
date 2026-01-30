/**
 * Dynamic Header Height Tracking
 *
 * This script measures the site header height and updates a CSS variable
 * that can be used by sticky elements (like TOC) to align with the header bottom.
 *
 * - Uses ResizeObserver to handle dynamic header height changes
 * - Fallback to window resize events for broader browser support
 * - Sets --site-header-height CSS variable on document root
 * - Dispatches custom event 'headerheightchange' when height changes
 */

function initHeaderHeightTracking() {
  const header = document.querySelector('[data-site-header]') as HTMLElement | null;

  if (!header) {
    // Fallback: set to 0 if header not found
    document.documentElement.style.setProperty('--site-header-height', '0px');
    return;
  }

  let lastHeight = 0;

  const updateHeaderHeight = () => {
    const height = header.getBoundingClientRect().height;
    if (Math.abs(height - lastHeight) > 0.5) {
      // Only update if changed significantly
      lastHeight = height;
      document.documentElement.style.setProperty('--site-header-height', `${height}px`);
      // Dispatch custom event for other components to listen
      window.dispatchEvent(
        new CustomEvent('headerheightchange', {
          detail: { height },
        }),
      );
    }
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
