import mermaid from 'mermaid';

type MermaidBlockElement = HTMLElement & {
  dataset: {
    mermaid?: string;
    idx?: string;
    rendered?: string;
    source?: string;
  };
};

const renderingBlocks = new Set<string>();

const isDarkTheme = () => document.documentElement.classList.contains('dark');
const getThemeMode = (): 'dark' | 'light' => (isDarkTheme() ? 'dark' : 'light');

const getBlocks = (): MermaidBlockElement[] =>
  Array.from(document.querySelectorAll<MermaidBlockElement>('.mermaid-block'));

const decodeMermaid = (encoded: string): string => {
  const normalized = encoded.replace(/\s/g, '');
  return decodeURIComponent(
    Array.from(atob(normalized))
      .map((char) => `%${char.charCodeAt(0).toString(16).padStart(2, '0')}`)
      .join(''),
  );
};

const initMermaid = (themeMode: 'dark' | 'light') => {
  const dark = themeMode === 'dark';

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
  let needsRerender = false;

  if (!encoded) return;
  if (!force && block.dataset.rendered === 'true') return;
  if (renderingBlocks.has(idx)) return;

  renderingBlocks.add(idx);

  const render = block.querySelector<HTMLElement>('.mermaid-render');
  if (render) render.innerHTML = '';

  try {
    const code = block.dataset.source || decodeMermaid(encoded);
    block.dataset.source = code;

    const renderTheme = getThemeMode();
    const mermaidInstance = initMermaid(renderTheme);
    const result = await mermaidInstance.render(`mermaid-diagram-${idx}`, code);
    updateSuccess(block, result.svg);

    if (renderTheme !== getThemeMode()) {
      delete block.dataset.rendered;
      needsRerender = true;
    }
  } catch (error) {
    const code = block.dataset.source || decodeMermaid(encoded);
    updateFailure(block, code, error, idx);
  } finally {
    renderingBlocks.delete(idx);

    if (needsRerender) {
      void renderBlock(block, true);
    }
  }
};

const renderBlocks = async (blocks: MermaidBlockElement[], force = false): Promise<void> => {
  await Promise.all(blocks.map((block) => renderBlock(block, force)));
};

const handleThemeChange = (): void => {
  const blocks = getBlocks();

  blocks.forEach((block) => {
    delete block.dataset.rendered;
  });

  void renderBlocks(blocks, true);
};

void renderBlocks(getBlocks());
window.addEventListener('themechange', () => {
  handleThemeChange();
});
