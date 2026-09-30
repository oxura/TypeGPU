import { describe, expect, expectTypeOf, it } from 'vitest';
import { tgpu, d, readFromArrayBuffer, writeToArrayBuffer } from 'typegpu';

describe('d.size', () => {
  it('adds @size attribute for the custom sized struct members', () => {
    const s1 = d.struct({
      a: d.u32,
      b: d.size(16, d.u32),
      c: d.u32,
    });

    expect(tgpu.resolve([s1])).toContain('@size(16) b: u32,');
  });

  it('changes size of the struct containing aligned member', () => {
    expect(
      d.sizeOf(
        d.struct({
          a: d.u32,
          b: d.u32,
          c: d.u32,
        }),
      ),
    ).toBe(12);

    expect(
      d.sizeOf(
        d.struct({
          a: d.u32,
          b: d.size(8, d.u32),
          c: d.u32,
        }),
      ),
    ).toBe(16);

    expect(
      d.sizeOf(
        d.struct({
          a: d.u32,
          b: d.size(8, d.u32),
          c: d.size(16, d.u32),
        }),
      ),
    ).toBe(28);

    // nested
    const NestedStruct = d.struct({
      a: d.u32,
      b: d.struct({
        c: d.size(20, d.f32),
      }),
    });
    expect(d.sizeOf(NestedStruct)).toBe(24);

    // taking alignment into account
    const AlignedStruct = d.struct({
      a: d.struct({
        c: d.size(17, d.f32),
      }),
      b: d.u32,
    });
    expect(d.sizeOf(AlignedStruct)).toBe(24);
  });

  it('throws for invalid size values', () => {
    expect(() => d.size(3, d.u32)).toThrow();
    d.size(4, d.u32);

    expect(() => d.size(11, d.vec3f)).toThrow();
    d.size(12, d.vec3f);

    expect(() => d.size(-2, d.u32)).toThrow();
  });

  it('changes size of loose array element', () => {
    const s1 = d.disarrayOf(d.size(11, d.u32), 10);

    expect(d.sizeOf(s1)).toBe(110);
    expectTypeOf(s1).toEqualTypeOf<d.Disarray<d.Decorated<d.U32, [d.Size<11>]>>>();
  });

  it('changes size of loose struct member of type loose array', () => {
    const s1 = d.unstruct({
      a: d.u32, // 4
      b: d.size(20, d.disarrayOf(d.u32, 4)), // 20
      c: d.u32, // 4
    });

    expect(d.sizeOf(s1)).toBe(28);
    expectTypeOf(s1).toEqualTypeOf<
      d.Unstruct<{
        a: d.U32;
        b: d.LooseDecorated<d.Disarray<d.U32>, [d.Size<20>]>;
        c: d.U32;
      }>
    >();
  });
});

describe('top-level size decorations', () => {
  it('measures the WGSL type rather than a member reservation', () => {
    const scalar = d.size(32, d.u32);
    expect(d.sizeOf(scalar)).toBe(4);
    expect(d.isContiguous(scalar)).toBe(true);
    expect(d.memoryLayoutOf(scalar)).toEqual({ offset: 0, contiguous: 4 });
    expect(d.memoryLayoutOf(scalar, (value) => value)).toEqual({ offset: 0, contiguous: 4 });
    expect(d.sizeOf(d.size(32, d.vec3f))).toBe(12);
    expect(d.sizeOf(d.size(64, d.arrayOf(d.vec3f, 2)))).toBe(32);
    expect(d.sizeOf(d.size(64, d.struct({ value: d.size(32, d.u32) })))).toBe(32);
  });

  it('preserves member reservations and array strides', () => {
    const schema = d.struct({ a: d.size(32, d.u32), b: d.u32 });
    expect(d.sizeOf(schema)).toBe(36);
    const bytes = new ArrayBuffer(36);
    writeToArrayBuffer(bytes, schema, { a: 7, b: 9 });
    expect(new DataView(bytes).getUint32(32, true)).toBe(9);
    expect(readFromArrayBuffer(bytes, schema)).toEqual({ a: 7, b: 9 });
    expect(d.sizeOf(d.vec3f)).toBe(12);
    expect(d.sizeOf(d.arrayOf(d.vec3f, 2))).toBe(32);
    expect(d.sizeOf(d.disarrayOf(d.size(11, d.u32), 10))).toBe(110);
  });

  it('round-trips an outer decorated scalar in its natural byte size', () => {
    const schema = d.size(32, d.u32);
    const bytes = new ArrayBuffer(4);
    writeToArrayBuffer(bytes, schema, 42);
    expect(readFromArrayBuffer(bytes, schema)).toBe(42);
    expect([...new Uint32Array(bytes)]).toEqual([42]);
  });
});
