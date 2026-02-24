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
  'floating-action-stack fixed z-40 flex flex-col items-end gap-3 lg:hidden pointer-events-auto transition-opacity data-[toc-open=true]:opacity-0 data-[toc-open=true]:pointer-events-none';

export const BUTTON_CLASSNAMES =
  'floating-action-stack__button h-11 w-11 rounded-full border border-gray-200 bg-white/90 text-gray-900 shadow-lg backdrop-blur-sm ring-offset-white transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 hover:-translate-y-0.5 hover:border-blue-200 hover:text-blue-600 data-[visible=false]:opacity-0 data-[visible=false]:pointer-events-none data-[visible=true]:opacity-100 data-[visible=true]:pointer-events-auto dark:border-gray-800 dark:bg-gray-900/85 dark:text-gray-100 dark:ring-offset-gray-900 dark:hover:border-blue-500/50 dark:hover:text-blue-300';

export const BUTTON_ICON_CLASSNAMES = 'text-base';

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
