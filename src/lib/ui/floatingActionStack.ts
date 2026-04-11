export type FloatingActionKind = 'toc';

export interface FloatingActionDescriptor {
  kind: FloatingActionKind;
  label: string;
  icon: string;
  ariaLabel: string;
  enabled: boolean;
  dataAttributes?: Record<string, string>;
}

export const STACK_CLASSNAMES =
  'floating-action-stack mobile-fab-stack fixed z-40 flex flex-col items-end gap-3 lg:hidden pointer-events-auto transition-opacity data-[toc-open=true]:opacity-0 data-[toc-open=true]:pointer-events-none';

export const BUTTON_CLASSNAMES =
  'floating-action-stack__button mobile-fab-button touch-pan-y inline-flex h-11 w-11 items-center justify-center rounded-full border border-zinc-200 bg-neutral-50/92 text-zinc-900 shadow-lg backdrop-blur-sm ring-offset-white transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 focus-visible:ring-offset-2 hover:-translate-y-0.5 hover:border-sky-200 hover:text-sky-600 data-[visible=false]:opacity-0 data-[visible=false]:pointer-events-none data-[visible=true]:opacity-100 data-[visible=true]:pointer-events-auto dark:border-zinc-800 dark:bg-zinc-900/88 dark:text-zinc-100 dark:ring-offset-zinc-900 dark:hover:border-sky-500/50 dark:hover:text-sky-300';

export const BUTTON_ICON_CLASSNAMES = 'text-base leading-none';

export const buildActions = ({
  enableToc = true,
}: {
  enableToc?: boolean;
}): FloatingActionDescriptor[] => {
  const actions: FloatingActionDescriptor[] = [
    {
      kind: 'toc',
      label: '目录',
      icon: '📑',
      ariaLabel: '打开目录抽屉',
      enabled: enableToc,
      dataAttributes: { 'data-mobile-toc-open': 'true' },
    },
  ];

  return actions.filter((action) => action.enabled);
};
