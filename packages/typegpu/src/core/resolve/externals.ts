import { isLooseData } from '../../data/dataTypes.ts';
import { isDecorated, isPtr, isWgslArray, isWgslStruct } from '../../data/wgslTypes.ts';
import { getName, hasTinyestMetadata, isNamable, setName } from '../../shared/meta.ts';
import { logger } from '../../tgpuLogger.ts';
import { isWgsl, type ResolutionCtx } from '../../types.ts';
import type { FnExternals } from '../function/fnCore.ts';
import { extractArgs } from '../function/extractArgs.ts';

/**
 * A key-value mapping where keys represent identifiers within shader code,
 * and values can either be another ExternalMap, or be any type that can be resolved to a code string.
 */
export type ExternalMap = Record<string, unknown>;

function isResolvable(value: unknown) {
  return isWgsl(value) || isLooseData(value) || hasTinyestMetadata(value);
}

/**
 * Merges function externals into one map.
 */
export function mergeFunctionExternals(fnExternals: FnExternals): ExternalMap {
  const base = fnExternals.pluginProvided ?? fnExternals.userProvided ?? {};
  // avoid calling any of the getters
  const result: ExternalMap = Object.defineProperties({}, Object.getOwnPropertyDescriptors(base));
  for (const flatExternal of [fnExternals.args, fnExternals.out].filter((e) => e !== undefined)) {
    for (const [key, value] of Object.entries(flatExternal)) {
      if (key in result && result[key] !== value) {
        throw new Error(
          `Key '${key}' appears in externals despite already being used for argument/return type. Please rename this external.`,
        );
      }
      result[key] = value;
    }
  }
  return result;
}

/**
 * Signature aliases name struct leaves, even inside arrays and pointers.
 * The surrounding WGSL type syntax is preserved when replacing these aliases.
 */
function collectStructAliases(schema: unknown, typeName: string | undefined, aliases: ExternalMap) {
  if (!typeName) {
    return;
  }
  if (isDecorated(schema)) {
    collectStructAliases(schema.inner, typeName, aliases);
  } else if (isWgslStruct(schema)) {
    aliases[typeName] = schema;
  } else if (isWgslArray(schema)) {
    const elementName = typeName.match(/^array<(.+?)(?:,\d+)?>$/)?.[1];
    collectStructAliases(schema.elementType, elementName, aliases);
  } else if (isPtr(schema)) {
    const innerName = typeName.match(/^ptr<[^,]+,(.+?)(?:,read|,read_write)?>$/)?.[1];
    collectStructAliases(schema.inner, innerName, aliases);
  }
}

export function addArgTypesToExternals(
  implementation: string,
  argTypes: unknown[],
  core: { setExternals: (key: 'args', externals: ExternalMap) => void },
) {
  const signature = extractArgs(implementation);
  const args: ExternalMap = {};
  for (const [i, argType] of argTypes.entries()) {
    collectStructAliases(argType, signature.args[i]?.type, args);
  }

  core.setExternals('args', args);
}

export function addReturnTypeToExternals(
  implementation: string,
  returnType: unknown,
  core: { setExternals: (key: 'out', externals: ExternalMap) => void },
) {
  const outputName = implementation.match(/->\s*([^{}]+)\s*\{/)?.[1]?.replace(/\s/g, '');
  const output: ExternalMap = {};
  collectStructAliases(returnType, outputName, output);
  if (Object.keys(output).length > 0) {
    core.setExternals('out', output);
  }
}

export const anyIdent = /([$_\p{XID_Start}][$\p{XID_Continue}]*)/u; // WGSL ident, modified to include $
const anyPropChain = new RegExp(`(${anyIdent.source})(\\.${anyIdent.source})*`, 'ug');
export const boundedPropChain = new RegExp(
  `(?<![\\p{XID_Continue}\\$.])${anyPropChain.source}(?![\\p{XID_Continue}\\$])`,
  'ug',
);

/**
 * Replaces all occurrences of external names in WGSL code with their resolved values.
 * It adds all necessary definitions to the resolution context.
 * @param ctx - The resolution context.
 * @param externalMap - The external map. Assumes that keys don't contain dots.
 * @param wgsl - The WGSL code.
 *
 * @returns The WGSL code with all external names replaced with their resolved values.
 */
export function replaceExternalsInWgsl(
  ctx: ResolutionCtx,
  externalMap: ExternalMap,
  wgsl: string,
): string {
  const keys = Object.keys(externalMap);
  if (keys.length === 0) {
    return wgsl;
  }

  const maybeInvalidKey = keys.find((key) => key.includes('.'));
  if (maybeInvalidKey) {
    throw new Error(`External key '${maybeInvalidKey}' contains invalid character '.'`);
  }

  return wgsl.replaceAll(boundedPropChain, (match) => {
    const chain = match.split('.');

    if (!Object.hasOwn(externalMap, chain.at(0) as string)) {
      // this prop access does not start with an external
      return match;
    }

    let currentItem: unknown = externalMap;
    let suffix = '';

    for (const [i, elem] of chain.entries()) {
      currentItem = (currentItem as ExternalMap)[elem];
      if (isResolvable(currentItem) || typeof currentItem === 'string') {
        suffix = chain
          .slice(i + 1)
          .map((s) => `.${s}`)
          .join('');

        if (isNamable(currentItem) && getName(currentItem) === undefined) {
          setName(currentItem, chain.slice(0, i + 1).join('_'));
        }

        break;
      }

      if (typeof currentItem !== 'object' || currentItem === null || i === chain.length - 1) {
        logger.warn(
          'external-omitted',
          `During resolution, the external '${chain.slice(0, i + 1).join('.')}' has been omitted. Only TGPU resources, 'use gpu' functions, primitives, and plain JS objects can be used as externals.`,
        );
        return match;
      }
    }

    return ctx.resolve(currentItem).value + suffix;
  });
}
