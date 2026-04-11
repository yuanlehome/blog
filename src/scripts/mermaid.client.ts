type MermaidModule = {
  initialize: (config: Record<string, unknown>) => void;
  render: (id: string, text: string, container?: Element) => Promise<{ svg: string }>;
};

type MermaidBlockElement = HTMLElement & {
  dataset: {
    mermaid?: string;
    idx?: string;
    rendered?: string;
    source?: string;
  };
};

const VIEWPORT_PRELOAD_PX = 1200;
const IDLE_TIMEOUT_MS = 1200;

const renderingBlocks = new Set<string>();
const queuedBlocks = new Map<MermaidBlockElement, boolean>();
let mermaidModulePromise: Promise<MermaidModule> | null = null;
let blockObserver: IntersectionObserver | null = null;
let renderQueueScheduled = false;

const isDarkTheme = () => document.documentElement.classList.contains('dark');

const getBlocks = (): MermaidBlockElement[] =>
  Array.from(document.querySelectorAll<MermaidBlockElement>('.mermaid-block'));

const runWhenIdle = (callback: () => void, timeout = IDLE_TIMEOUT_MS): void => {
  if (typeof window.requestIdleCallback === 'function') {
    window.requestIdleCallback(() => callback(), { timeout });
    return;
  }
  window.setTimeout(callback, 1);
};

const isNearViewport = (element: Element): boolean => {
  const rect = element.getBoundingClientRect();
  return rect.top < window.innerHeight + VIEWPORT_PRELOAD_PX && rect.bottom > -VIEWPORT_PRELOAD_PX;
};

const decodeMermaid = (encoded: string): string => {
  const normalized = encoded.replace(/\s/g, '');
  return decodeURIComponent(
    Array.from(atob(normalized))
      .map((char) => `%${char.charCodeAt(0).toString(16).padStart(2, '0')}`)
      .join(''),
  );
};

const getMermaidModule = async (): Promise<MermaidModule> => {
  if (!mermaidModulePromise) {
    mermaidModulePromise = import('mermaid').then((module) => module.default as MermaidModule);
  }
  return mermaidModulePromise;
};

const initMermaid = async () => {
  const mermaid = await getMermaidModule();
  const dark = isDarkTheme();

  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    theme: 'base',
    themeVariables: dark
      ? {
          background: 'transparent',
          primaryTextColor: '#f4f4f5',
          primaryBorderColor: '#d4d4d8',
          lineColor: '#d4d4d8',
          primaryColor: '#27272a',
          secondaryColor: '#3f3f46',
          tertiaryColor: '#52525b',
        }
      : {
          background: 'transparent',
          primaryTextColor: '#18181b',
          lineColor: '#27272a',
          primaryColor: '#f4f4f5',
          secondaryColor: '#e4e4e7',
          tertiaryColor: '#d4d4d8',
        },
  });

  return mermaid;
};

const updateFailure = (
  block: MermaidBlockElement,
  code: string,
  error: unknown,
  idx: string,
): void => {
  const loading = block.querySelector<HTMLElement>('.mermaid-loading');
  const source = block.querySelector<HTMLElement>('.mermaid-source');
  const render = block.querySelector<HTMLElement>('.mermaid-render');
  const errorBox = block.querySelector<HTMLElement>('.mermaid-error');

  if (loading) loading.hidden = true;
  if (render) render.innerHTML = '';

  if (source) {
    source.textContent = code;
    source.hidden = false;
  }

  if (errorBox) {
    errorBox.textContent = `Failed to render Mermaid diagram (${idx}).`;
    errorBox.hidden = false;
  }

  console.error(`[mermaid] render failed for block ${idx}`, error);
};

const updateSuccess = (block: MermaidBlockElement, svg: string): void => {
  const loading = block.querySelector<HTMLElement>('.mermaid-loading');
  const source = block.querySelector<HTMLElement>('.mermaid-source');
  const render = block.querySelector<HTMLElement>('.mermaid-render');
  const errorBox = block.querySelector<HTMLElement>('.mermaid-error');

  if (render) {
    render.innerHTML = svg;
    const renderedSvg = render.querySelector<SVGSVGElement>('svg');
    if (renderedSvg) {
      renderedSvg.setAttribute('tabindex', '0');
      renderedSvg.setAttribute('role', 'button');
      renderedSvg.setAttribute('aria-label', '点击查看 Mermaid 图表大图');
      renderedSvg.style.cursor = 'zoom-in';
    }
  }
  if (loading) loading.hidden = true;
  if (source) source.hidden = true;
  if (errorBox) {
    errorBox.textContent = '';
    errorBox.hidden = true;
  }

  block.dataset.rendered = 'true';
};

const renderBlock = async (block: MermaidBlockElement, force = false): Promise<void> => {
  const encoded = block.dataset.mermaid;
  const idx = block.dataset.idx || 'unknown';

  if (!encoded) return;
  if (!force && block.dataset.rendered === 'true') return;
  if (renderingBlocks.has(idx)) return;

  renderingBlocks.add(idx);

  const render = block.querySelector<HTMLElement>('.mermaid-render');
  if (render) render.innerHTML = '';

  try {
    const code = block.dataset.source || decodeMermaid(encoded);
    block.dataset.source = code;

    const mermaid = await initMermaid();
    const result = await mermaid.render(`mermaid-diagram-${idx}`, code);
    updateSuccess(block, result.svg);
  } catch (error) {
    const code = block.dataset.source || decodeMermaid(encoded);
    updateFailure(block, code, error, idx);
  } finally {
    renderingBlocks.delete(idx);
  }
};

const flushRenderQueue = (): void => {
  renderQueueScheduled = false;
  const blocks = Array.from(queuedBlocks.entries());
  queuedBlocks.clear();
  void Promise.all(blocks.map(([block, force]) => renderBlock(block, force)));
};

const queueBlockRender = (block: MermaidBlockElement, force = false): void => {
  if (!force && block.dataset.rendered === 'true') return;
  queuedBlocks.set(block, force || queuedBlocks.get(block) === true);
  blockObserver?.unobserve(block);
  if (renderQueueScheduled) return;
  renderQueueScheduled = true;
  runWhenIdle(flushRenderQueue);
};

const ensureObserver = (): IntersectionObserver | null => {
  if (blockObserver || typeof IntersectionObserver !== 'function') {
    return blockObserver;
  }

  blockObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        queueBlockRender(entry.target as MermaidBlockElement);
      });
    },
    {
      rootMargin: `${VIEWPORT_PRELOAD_PX}px 0px`,
    },
  );

  return blockObserver;
};

const observeBlocks = (blocks: MermaidBlockElement[]): void => {
  const observer = ensureObserver();
  if (!observer) {
    blocks.forEach((block) => queueBlockRender(block));
    return;
  }

  blocks.forEach((block) => {
    if (block.dataset.rendered === 'true') return;
    if (isNearViewport(block)) {
      queueBlockRender(block);
      return;
    }
    observer.observe(block);
  });
};

const startLazyRendering = (): void => {
  observeBlocks(getBlocks());
};

const handleThemeChange = (): void => {
  const blocks = getBlocks();
  const renderedBlocks = blocks.filter((block) => block.dataset.rendered === 'true');

  blocks.forEach((block) => {
    delete block.dataset.rendered;
  });

  observeBlocks(blocks);
  renderedBlocks.forEach((block) => queueBlockRender(block, true));
};

runWhenIdle(startLazyRendering);
window.addEventListener('themechange', () => {
  handleThemeChange();
});
