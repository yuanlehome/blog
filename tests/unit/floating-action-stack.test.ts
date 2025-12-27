import { describe, expect, it } from 'vitest';
import { JSDOM } from 'jsdom';
import {
  BUTTON_CLASSNAMES,
  STACK_CLASSNAMES,
  buildActions,
} from '../../src/utils/floatingActionStack';

describe('FloatingActionStack layout', () => {
  it('stacks mobile actions vertically with consistent spacing', () => {
    const dom = new JSDOM('<!doctype html><body></body>');
    const document = dom.window.document;

    const container = document.createElement('div');
    container.className = STACK_CLASSNAMES;
    container.dataset.floatingActionStack = 'true';

    const actions = buildActions({ enableTop: true, enableToc: true, enableBottom: true });
    actions.forEach((action, index) => {
      const button = document.createElement('button');
      button.dataset.action = action.kind;
      button.dataset.visible = action.kind === 'top' ? 'false' : 'true';
      button.className = BUTTON_CLASSNAMES;
      button.textContent = `${index}-${action.label}`;
      container.appendChild(button);
    });

    document.body.appendChild(container);

    expect(container.className).toContain('flex-col');
    expect(container.className).toContain('gap-3');
    expect(container.className).toContain('pointer-events-auto');

    const buttons: HTMLButtonElement[] = Array.from(
      container.querySelectorAll<HTMLButtonElement>('button'),
    );
    expect(buttons).toHaveLength(3);
    expect(buttons.map((button) => button.dataset.action)).toEqual(['top', 'toc', 'bottom']);

    buttons.forEach((button) => {
      expect(button.className).toContain('h-11');
      expect(button.className).toContain('w-11');
      expect(button.closest('[data-floating-action-stack]')).toBe(container);
    });
  });
});
