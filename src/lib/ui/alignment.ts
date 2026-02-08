/**
 * Alignment utility functions
 *
 * Maps config alignment values to Tailwind CSS classes.
 * This ensures config-driven alignment instead of hardcoded classes.
 *
 * @module src/lib/ui/alignment
 */

export type AlignmentValue = 'left' | 'center';

/**
 * Map alignment value to Tailwind text-align class
 *
 * @param align - Alignment value from config
 * @returns Tailwind class string
 *
 * @example
 * ```ts
 * alignToTextClass('left')   // 'text-left'
 * alignToTextClass('center') // 'text-center'
 * ```
 */
export function alignToTextClass(align: AlignmentValue): string {
  return align === 'center' ? 'text-center' : 'text-left';
}

/**
 * Map alignment value to Tailwind justify-content class
 *
 * @param align - Alignment value from config
 * @returns Tailwind class string
 *
 * @example
 * ```ts
 * alignToJustifyClass('left')   // 'justify-start'
 * alignToJustifyClass('center') // 'justify-center'
 * ```
 */
export function alignToJustifyClass(align: AlignmentValue): string {
  return align === 'center' ? 'justify-center' : 'justify-start';
}

/**
 * Map alignment value to Tailwind items class
 *
 * @param align - Alignment value from config
 * @returns Tailwind class string
 *
 * @example
 * ```ts
 * alignToItemsClass('left')   // 'items-start'
 * alignToItemsClass('center') // 'items-center'
 * ```
 */
export function alignToItemsClass(align: AlignmentValue): string {
  return align === 'center' ? 'items-center' : 'items-start';
}
