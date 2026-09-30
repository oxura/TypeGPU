import type { AnyData } from './dataTypes.ts';
import { isDecorated, type BaseData } from './wgslTypes.ts';
import { getLayoutInfo } from './schemaMemoryLayout.ts';

/** Includes any member reservation specified by @size. */
export function sizeOf(schema: BaseData): number {
  return getLayoutInfo(schema, 'size');
}

/** The size of a standalone type, where WGSL member attributes have no effect. */
export function sizeOfType(schema: BaseData): number {
  return sizeOf(isDecorated(schema) ? schema.inner : schema);
}

/**
 * Returns the size (in bytes) of data represented by the `schema`.
 */
export function PUBLIC_sizeOf(schema: AnyData): number {
  return sizeOfType(schema);
}
