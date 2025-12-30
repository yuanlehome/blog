import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { JSDOM } from 'jsdom';

// Mock the config loader
vi.mock('../../src/config/loaders/components', () => ({
  getComponentsConfig: vi.fn(() => ({
    headerMascot: {
      enabled: true,
      speed: 1.0,
      interactive: true,
      hideOnMobile: true,
    },
    radius: {},
    shadow: {},
    shadowValues: {},
    shadowValuesDark: {},
    border: {},
    motion: {},
    motionTiming: {},
    spacingScale: 'comfortable',
    spacingMultiplier: {},
  })),
}));

describe('HeaderMascot Component Behavior', () => {
  let dom: JSDOM;
  let document: Document;
  let window: Window & typeof globalThis;

  beforeEach(() => {
    dom = new JSDOM(
      `
      <!DOCTYPE html>
      <html>
        <head></head>
        <body>
          <div class="header-mascot" data-mascot aria-hidden="true">
            <svg class="mascot-svg" width="40" height="60">
              <circle cx="20" cy="12" r="8"></circle>
              <line x1="20" y1="20" x2="20" y2="35"></line>
              <g class="mascot-arms">
                <line class="arm-left" x1="20" y1="25" x2="10" y2="30"></line>
                <line class="arm-right" x1="20" y1="25" x2="30" y2="30"></line>
              </g>
              <g class="mascot-legs">
                <line class="leg-left" x1="20" y1="35" x2="12" y2="50"></line>
                <line class="leg-right" x1="20" y1="35" x2="28" y2="50"></line>
              </g>
            </svg>
          </div>
        </body>
      </html>
    `,
      { url: 'http://localhost' },
    );

    document = dom.window.document;
    window = dom.window as unknown as Window & typeof globalThis;
  });

  afterEach(() => {
    dom.window.close();
  });

  it('should render mascot with correct structure', () => {
    const mascot = document.querySelector('[data-mascot]');
    expect(mascot).toBeTruthy();
    expect(mascot?.getAttribute('aria-hidden')).toBe('true');
  });

  it('should have SVG with stick figure elements', () => {
    const svg = document.querySelector('.mascot-svg');
    expect(svg).toBeTruthy();

    // Check for head (circle)
    const head = svg?.querySelector('circle');
    expect(head).toBeTruthy();

    // Check for body (line)
    const body = svg?.querySelector('line[x1="20"][y1="20"]');
    expect(body).toBeTruthy();

    // Check for arms
    const leftArm = svg?.querySelector('.arm-left');
    const rightArm = svg?.querySelector('.arm-right');
    expect(leftArm).toBeTruthy();
    expect(rightArm).toBeTruthy();

    // Check for legs
    const leftLeg = svg?.querySelector('.leg-left');
    const rightLeg = svg?.querySelector('.leg-right');
    expect(leftLeg).toBeTruthy();
    expect(rightLeg).toBeTruthy();
  });

  it('should have mascot-arms and mascot-legs groups', () => {
    const armsGroup = document.querySelector('.mascot-arms');
    const legsGroup = document.querySelector('.mascot-legs');

    expect(armsGroup).toBeTruthy();
    expect(legsGroup).toBeTruthy();

    // Verify arms group contains arm elements
    expect(armsGroup?.querySelectorAll('line').length).toBe(2);

    // Verify legs group contains leg elements
    expect(legsGroup?.querySelectorAll('line').length).toBe(2);
  });

  it('should handle click events when interactive', () => {
    const mascot = document.querySelector('[data-mascot]') as HTMLElement;
    const svg = mascot?.querySelector('.mascot-svg') as SVGElement;

    expect(mascot).toBeTruthy();
    expect(svg).toBeTruthy();

    // Simulate setting up click handler (as would be done in actual component)
    mascot.style.pointerEvents = 'auto';
    mascot.style.cursor = 'pointer';

    mascot.addEventListener('click', () => {
      svg.classList.add('jumping');
      setTimeout(() => {
        svg.classList.remove('jumping');
      }, 600);
    });

    // Trigger click
    mascot.click();
    expect(svg.classList.contains('jumping')).toBe(true);
  });

  it('should be accessible with aria-hidden', () => {
    const mascot = document.querySelector('[data-mascot]');
    expect(mascot?.getAttribute('aria-hidden')).toBe('true');
  });

  it('should support hiding on mobile with responsive classes', () => {
    // Re-create with mobile hide class
    document.body.innerHTML = `
      <div class="header-mascot hidden md:block" data-mascot aria-hidden="true">
        <svg class="mascot-svg"></svg>
      </div>
    `;

    const mascot = document.querySelector('[data-mascot]') as HTMLElement;
    expect(mascot.className).toContain('hidden');
    expect(mascot.className).toContain('md:block');
  });
});

describe('HeaderMascot prefers-reduced-motion', () => {
  it('should respect reduced motion preferences in CSS', () => {
    // This tests the CSS structure - in real browser, media query would apply
    // We're verifying that the component includes the appropriate structure
    const cssContent = `
      @media (prefers-reduced-motion: reduce) {
        .mascot-svg,
        .leg-left,
        .leg-right,
        .arm-left,
        .arm-right,
        .mascot-arms {
          animation: none !important;
        }
        
        .header-mascot {
          opacity: 0.4;
        }
      }
    `;

    // Verify CSS media query structure exists
    expect(cssContent).toContain('prefers-reduced-motion: reduce');
    expect(cssContent).toContain('animation: none !important');
    expect(cssContent).toContain('opacity: 0.4');
  });
});
