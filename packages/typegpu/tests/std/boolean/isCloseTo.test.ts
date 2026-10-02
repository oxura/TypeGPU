import { describe, expect, it } from 'vitest';
import { d, tgpu } from 'typegpu';
import { isCloseTo } from 'typegpu/std';

describe('isCloseTo', () => {
  it('returns true for close f32 containers', () => {
    expect(isCloseTo(d.vec2f(0, 0), d.vec2f(0.0012, -0.009))).toBe(true);
    expect(isCloseTo(d.vec3f(1.05, -2.18, 1.22), d.vec3f(1.05, -2.17777, 1.229))).toBe(true);
    expect(
      isCloseTo(d.vec4f(3.8, -4.87, -2.42, -1.97), d.vec4f(3.794, -4.861, -2.412, -1.971)),
    ).toBe(true);
  });

  it('returns true for close f16 containers', () => {
    expect(isCloseTo(d.vec2h(0, 0), d.vec2h(0.0012, -0.009))).toBe(true);
    expect(isCloseTo(d.vec3h(1.05, -2.18, 1.22), d.vec3h(1.05, -2.17777, 1.229))).toBe(true);
    expect(
      isCloseTo(d.vec4h(3.8, -4.87, -2.42, -1.97), d.vec4h(3.794, -4.861, -2.412, -1.971), 0.05),
    ).toBe(true);
  });

  it('returns true for close numbers', () => {
    expect(isCloseTo(0, 0.009)).toBe(true);
    expect(isCloseTo(0, 0.0009)).toBe(true);
  });

  it('returns false for distant f32 containers', () => {
    expect(isCloseTo(d.vec2f(0, 0), d.vec2f(0, 1))).toBe(false);
    expect(isCloseTo(d.vec3f(100, 100, 100), d.vec3f(101, 100, 100))).toBe(false);
    expect(isCloseTo(d.vec4f(1, 2, 3, 4), d.vec4f(1.02, 2.02, 3.02, 4.02))).toBe(false);
  });

  it('returns false for distant f16 containers', () => {
    expect(isCloseTo(d.vec2h(0, 0), d.vec2h(0, 1))).toBe(false);
    expect(isCloseTo(d.vec3h(100, 100, 100), d.vec3h(101, 100, 100))).toBe(false);
    expect(isCloseTo(d.vec4h(1, 2, 3, 4), d.vec4h(1.02, 2.02, 3.02, 4.02))).toBe(false);
  });

  it('returns false for distant numbers', () => {
    expect(isCloseTo(0, 0.9)).toBe(false);
    expect(isCloseTo(0, 0.09)).toBe(false);
  });

  it('applies precision correctly', () => {
    // default precision of 0.01
    expect(isCloseTo(d.vec2h(0, 0), d.vec2h(0, 0.009))).toBe(true);
    expect(isCloseTo(d.vec2h(0, 0), d.vec2h(0, 0.011))).toBe(false);

    expect(isCloseTo(d.vec2h(0, 0), d.vec2h(0, 0.09), 0.1)).toBe(true);
    expect(isCloseTo(d.vec2h(0, 0), d.vec2h(0, 0.11), 0.1)).toBe(false);

    expect(isCloseTo(d.vec2h(0, 0), d.vec2h(0, 0.0009), 0.001)).toBe(true);
    expect(isCloseTo(d.vec2h(0, 0), d.vec2h(0, 0.0011), 0.001)).toBe(false);

    expect(isCloseTo(d.vec2h(0, 0), d.vec2h(0, 9), 10)).toBe(true);
    expect(isCloseTo(d.vec2h(0, 0), d.vec2h(0, 11), 10)).toBe(false);
  });

  it('does not duplicate any of its sides', () => {
    const myVar = tgpu.privateVar(d.u32, 0);
    const modify1 = () => {
      'use gpu';
      const value = myVar.$;
      myVar.$ += 1;
      return d.vec2f(value);
    };
    const modify2 = () => {
      'use gpu';
      const value = myVar.$;
      myVar.$ += 2;
      return d.vec2f(value);
    };
    const main = () => {
      'use gpu';
      return isCloseTo(modify1(), modify2());
    };

    const code = tgpu.resolve([main]);
    expect(code).toMatchInlineSnapshot(`
      "var<private> myVar: u32;

      fn modify1() -> vec2f {
        let value = myVar;
        myVar += 1u;
        return vec2f(f32(value));
      }

      fn modify2() -> vec2f {
        let value = myVar;
        myVar += 2u;
        return vec2f(f32(value));
      }

      fn main() -> bool {
        return all((abs(modify1() - modify2()) <= vec2f(0.01f)));
      }"
    `);
    expect(code.match(/modify1/g)?.length).toBe(1 /* decl */ + 1 /* call */);
    expect(code.match(/modify2/g)?.length).toBe(1 /* decl */ + 1 /* call */);
  });
});

describe.each([
  [d.f32, 'abs(lhs - rhs)', false],
  [d.f16, 'abs(f32(lhs) - f32(rhs))', false],
  [d.i32, 'abs(f32(lhs) - f32(rhs))', false],
  [d.u32, 'abs(f32(lhs) - f32(rhs))', false],
  [d.vec2f, 'abs(lhs - rhs)', true],
  [d.vec3f, 'abs(lhs - rhs)', true],
  [d.vec4f, 'abs(lhs - rhs)', true],
  [d.vec2h, 'abs(lhs - rhs)', true],
  [d.vec3h, 'abs(lhs - rhs)', true],
  [d.vec4h, 'abs(lhs - rhs)', true],
] as const)('WGSL isCloseTo for %s', (schema, difference, isVector) => {
  const comparison = (precision: string) =>
    isVector
      ? `all((${difference} <= ${schema.type}(${precision})))`
      : `(${difference} <= ${precision})`;

  it('uses the default precision', () => {
    const compare = tgpu.fn([schema, schema], d.bool)((lhs, rhs) => isCloseTo(lhs, rhs));

    expect(tgpu.resolve([compare])).toContain(`return ${comparison('0.01f')};`);
  });

  it('uses an explicit precision', () => {
    const compare = tgpu.fn([schema, schema], d.bool)((lhs, rhs) => isCloseTo(lhs, rhs, 0.1));

    expect(tgpu.resolve([compare])).toContain(
      `return ${comparison(isVector ? '0.1' : 'f32(0.1)')};`,
    );
  });

  it.each([-1, 0, 1])('uses an integer precision literal (%s)', (precision) => {
    const compare = tgpu.fn([schema, schema], d.bool)((lhs, rhs) => isCloseTo(lhs, rhs, precision));
    const tolerance = isVector ? `${precision}` : `f32(${precision})`;

    expect(tgpu.resolve([compare])).toContain(`return ${comparison(tolerance)};`);
  });

  it.each([d.i32, d.u32])('uses a runtime %s precision', (precisionSchema) => {
    const compare = tgpu.fn(
      [schema, schema, precisionSchema],
      d.bool,
    )((lhs, rhs, tolerance) => isCloseTo(lhs, rhs, tolerance));
    const tolerance = isVector ? 'tolerance' : 'f32(tolerance)';

    expect(tgpu.resolve([compare])).toContain(`return ${comparison(tolerance)};`);
  });

  it('uses a runtime precision', () => {
    const compare = tgpu.fn(
      [schema, schema, d.f32],
      d.bool,
    )((lhs, rhs, tolerance) => isCloseTo(lhs, rhs, tolerance));

    expect(tgpu.resolve([compare])).toContain(`return ${comparison('tolerance')};`);
  });
});
