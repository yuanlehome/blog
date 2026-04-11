import { describe, expect, it } from 'vitest';
import { createTouchClickGuard } from '../../src/lib/ui/touchClickGuard';

const createPointerEvent = (overrides: Partial<PointerEvent> = {}) =>
  ({
    pointerType: 'touch',
    pointerId: 1,
    clientX: 0,
    clientY: 0,
    timeStamp: 0,
    ...overrides,
  }) as PointerEvent;

const createMouseEvent = (overrides: Partial<MouseEvent> = {}) =>
  ({
    timeStamp: 0,
    ...overrides,
  }) as MouseEvent;

describe('createTouchClickGuard', () => {
  it('allows a simple tap-derived click', () => {
    const guard = createTouchClickGuard();

    guard.handlePointerDown(createPointerEvent({ timeStamp: 10 }));
    guard.handlePointerUp(createPointerEvent({ timeStamp: 30 }));

    expect(guard.shouldHandleClick(createMouseEvent({ timeStamp: 40 }))).toBe(true);
  });

  it('suppresses a click that follows a drag gesture', () => {
    const guard = createTouchClickGuard();

    guard.handlePointerDown(createPointerEvent({ timeStamp: 10, clientX: 12, clientY: 12 }));
    guard.handlePointerMove(createPointerEvent({ timeStamp: 20, clientX: 12, clientY: 28 }));
    guard.handlePointerUp(createPointerEvent({ timeStamp: 40, clientX: 12, clientY: 32 }));

    expect(guard.shouldHandleClick(createMouseEvent({ timeStamp: 50 }))).toBe(false);
  });

  it('suppresses a click when the page scrolled during the touch gesture', () => {
    const guard = createTouchClickGuard();

    guard.handlePointerDown(createPointerEvent({ timeStamp: 10 }));
    guard.handleScroll(10);
    guard.handlePointerUp(createPointerEvent({ timeStamp: 40 }));

    expect(guard.shouldHandleClick(createMouseEvent({ timeStamp: 50 }))).toBe(false);
  });

  it('allows regular clicks when there is no preceding touch gesture', () => {
    const guard = createTouchClickGuard();

    expect(guard.shouldHandleClick(createMouseEvent({ timeStamp: 20 }))).toBe(true);
  });
});
