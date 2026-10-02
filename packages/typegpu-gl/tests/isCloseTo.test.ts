import { describe, expect } from 'vitest';
import { tgpu, d, std } from 'typegpu';
import { glOptions } from '@typegpu/gl';
import { it } from './utils/extendedTest.ts';

describe.each([
  [d.f32, 'float', 'abs(lhs - rhs)', false],
  [d.i32, 'int', 'abs(float(lhs) - float(rhs))', false],
  [d.u32, 'uint', 'abs(float(lhs) - float(rhs))', false],
  [d.vec2f, 'vec2', 'abs(lhs - rhs)', true],
  [d.vec3f, 'vec3', 'abs(lhs - rhs)', true],
  [d.vec4f, 'vec4', 'abs(lhs - rhs)', true],
] as const)('GLSL isCloseTo for %s', (schema, glslType, difference, isVector) => {
  const comparison = (precision: string) =>
    isVector
      ? `all(lessThanEqual(${difference}, ${glslType}(${precision})))`
      : `(${difference} <= ${precision})`;

  it('uses the default precision', () => {
    const compare = tgpu.fn([schema, schema], d.bool)((lhs, rhs) => std.isCloseTo(lhs, rhs));

    expect(tgpu.resolve([compare], glOptions())).toBe(
      `bool compare(${glslType} lhs, ${glslType} rhs) {\n  return ${comparison('0.01')};\n}`,
    );
  });

  it('uses an explicit precision', () => {
    const compare = tgpu.fn([schema, schema], d.bool)((lhs, rhs) => std.isCloseTo(lhs, rhs, 0.1));

    expect(tgpu.resolve([compare], glOptions())).toBe(
      `bool compare(${glslType} lhs, ${glslType} rhs) {\n  return ${comparison(isVector ? '0.1' : 'float(0.1)')};\n}`,
    );
  });

  it.each([-1, 0, 1])('uses an integer precision literal (%s)', (precision) => {
    const compare = tgpu.fn(
      [schema, schema],
      d.bool,
    )((lhs, rhs) => std.isCloseTo(lhs, rhs, precision));
    const tolerance = isVector ? `${precision}` : `float(${precision})`;

    expect(tgpu.resolve([compare], glOptions())).toBe(
      `bool compare(${glslType} lhs, ${glslType} rhs) {\n  return ${comparison(tolerance)};\n}`,
    );
  });

  it.each([
    [d.i32, 'int'],
    [d.u32, 'uint'],
  ] as const)('uses a runtime %s precision', (precisionSchema, precisionType) => {
    const compare = tgpu.fn(
      [schema, schema, precisionSchema],
      d.bool,
    )((lhs, rhs, tolerance) => std.isCloseTo(lhs, rhs, tolerance));
    const tolerance = isVector ? 'tolerance' : 'float(tolerance)';

    expect(tgpu.resolve([compare], glOptions())).toBe(
      `bool compare(${glslType} lhs, ${glslType} rhs, ${precisionType} tolerance) {\n  return ${comparison(tolerance)};\n}`,
    );
  });

  it('uses a runtime precision', () => {
    const compare = tgpu.fn(
      [schema, schema, d.f32],
      d.bool,
    )((lhs, rhs, tolerance) => std.isCloseTo(lhs, rhs, tolerance));

    expect(tgpu.resolve([compare], glOptions())).toBe(
      `bool compare(${glslType} lhs, ${glslType} rhs, float tolerance) {\n  return ${comparison('tolerance')};\n}`,
    );
  });
});

describe.each([d.f32, d.vec3f])('GLSL isCloseTo side effects for %s', (schema) => {
  it('evaluates each operand and the precision once', () => {
    const counter = tgpu.privateVar(d.u32, 0);
    const getLhs = tgpu.fn(
      [],
      schema,
    )(() => {
      counter.$ += 1;
      return schema(1);
    });
    const getRhs = tgpu.fn(
      [],
      schema,
    )(() => {
      counter.$ += 2;
      return schema(2);
    });
    const getTolerance = tgpu.fn(
      [],
      d.f32,
    )(() => {
      counter.$ += 4;
      return 0.1;
    });
    const compare = tgpu.fn([], d.bool)(() => std.isCloseTo(getLhs(), getRhs(), getTolerance()));

    const code = tgpu.resolve([compare], glOptions());
    expect(code.match(/getLhs\(/g)).toHaveLength(2 /* declaration and call */);
    expect(code.match(/getRhs\(/g)).toHaveLength(2 /* declaration and call */);
    expect(code.match(/getTolerance\(/g)).toHaveLength(2 /* declaration and call */);
  });
});
