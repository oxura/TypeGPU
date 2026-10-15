import { type AnyData, undecorate } from './dataTypes.ts';
import type { BaseData } from './wgslTypes.ts';
import { isDecorated } from './wgslTypes.ts';
import { getLayoutInfo } from './schemaMemoryLayout.ts';

export function sizeOf(schema: BaseData): number {
  return getLayoutInfo(schema, 'size');
}

/**
 * Returns the size (in bytes) of data represented by the `schema`.
 * Struct-member attributes on the schema itself do not change its WGSL type's size.
 * Attributes on members inside a struct are included.
 */
export function PUBLIC_sizeOf(schema: AnyData): number {
  return sizeOf(isDecorated(schema) ? undecorate(schema) : schema);
}
