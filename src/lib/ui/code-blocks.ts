const resetTimeouts = new WeakMap<HTMLButtonElement, number>();

const setState = (button: HTMLButtonElement, state: 'idle' | 'copied' | 'error') => {
  button.dataset.state = state;
  if (state === 'copied') {
    button.textContent = 'Copied';
  } else if (state === 'error') {
    button.textContent = 'Failed';
  } else {
    button.textContent = 'Copy';
  }

  const previous = resetTimeouts.get(button);
  if (previous) window.clearTimeout(previous);
  if (state !== 'idle') {
    const timer = window.setTimeout(() => setState(button, 'idle'), 2000);
    resetTimeouts.set(button, timer);
  }
};

const handleCopy = async (button: HTMLButtonElement) => {
  const wrapper = button.closest<HTMLElement>('.code-block');
  if (!wrapper) return;
  const raw = wrapper.getAttribute('data-raw-code') ?? '';
  try {
    await navigator.clipboard.writeText(raw);
    setState(button, 'copied');
  } catch (error) {
    console.error('Copy failed', error);
    setState(button, 'error');
  }
};

const registerCopyButtons = () => {
  document.addEventListener('click', (event) => {
    const target = event.target as HTMLElement | null;
    const button = target?.closest<HTMLButtonElement>('button.code-block__copy');
    if (!button) return;
    event.preventDefault();
    handleCopy(button);
  });
};

export const initCodeBlockCopy = () => {
  if (typeof document === 'undefined') return;
  registerCopyButtons();
};

// Auto-initialize when the script is loaded
initCodeBlockCopy();
