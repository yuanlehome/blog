export interface TouchClickGuardOptions {
  moveThreshold?: number;
}

interface ActiveTouchGesture {
  pointerId: number;
  startX: number;
  startY: number;
  moved: boolean;
}

interface PendingTouchClick {
  finishedAt: number;
  moved: boolean;
}

const DEFAULT_MOVE_THRESHOLD = 12;
const PENDING_CLICK_WINDOW_MS = 800;

export const createTouchClickGuard = ({
  moveThreshold = DEFAULT_MOVE_THRESHOLD,
}: TouchClickGuardOptions = {}) => {
  let activeGesture: ActiveTouchGesture | null = null;
  let pendingTouchClick: PendingTouchClick | null = null;

  const resetPendingTouchClickIfStale = (timeStamp: number) => {
    if (!pendingTouchClick) return;
    if (timeStamp - pendingTouchClick.finishedAt > PENDING_CLICK_WINDOW_MS) {
      pendingTouchClick = null;
    }
  };

  return {
    handlePointerDown(event: PointerEvent) {
      if (event.pointerType !== 'touch') return;
      activeGesture = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        moved: false,
      };
      pendingTouchClick = null;
    },

    handlePointerMove(event: PointerEvent) {
      if (!activeGesture || event.pointerId !== activeGesture.pointerId) return;
      const deltaX = event.clientX - activeGesture.startX;
      const deltaY = event.clientY - activeGesture.startY;
      if (Math.hypot(deltaX, deltaY) >= moveThreshold) {
        activeGesture.moved = true;
      }
    },

    handlePointerUp(event: PointerEvent) {
      if (!activeGesture || event.pointerId !== activeGesture.pointerId) return;
      pendingTouchClick = {
        finishedAt: event.timeStamp,
        moved: activeGesture.moved,
      };
      activeGesture = null;
    },

    handlePointerCancel(event: PointerEvent) {
      if (!activeGesture || event.pointerId !== activeGesture.pointerId) return;
      pendingTouchClick = {
        finishedAt: event.timeStamp,
        moved: true,
      };
      activeGesture = null;
    },

    shouldHandleClick(event: MouseEvent) {
      resetPendingTouchClickIfStale(event.timeStamp);
      if (!pendingTouchClick) return true;
      const shouldHandle = !pendingTouchClick.moved;
      pendingTouchClick = null;
      return shouldHandle;
    },
  };
};
