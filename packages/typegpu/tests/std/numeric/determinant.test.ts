import { it } from 'typegpu-testing-utility';
import { describe, expect, expectTypeOf } from 'vitest';
import { d, std, tgpu } from 'typegpu';

type MatrixSchema = typeof d.mat2x2f | typeof d.mat3x3f | typeof d.mat4x4f;

function fromElements(schema: MatrixSchema, elements: number[]) {
  const matrix = schema();
  const dimension = matrix.columns.length;
  matrix.columns.forEach((column, col) => {
    for (let row = 0; row < dimension; row++) {
      column[row] = elements[col * dimension + row]!;
    }
  });
  return matrix;
}

const cases = [
  { schema: d.mat2x2f, dimension: 2 },
  { schema: d.mat3x3f, dimension: 3 },
  { schema: d.mat4x4f, dimension: 4 },
];

// The permutation formula is independent of the cofactor expansion used by the implementation.
function referenceDeterminant(columns: number[][]): number {
  const n = columns.length;
  function visit(rows: number[]): number {
    if (rows.length < n) {
      let sum = 0;
      for (let row = 0; row < n; row++) {
        if (!rows.includes(row)) sum += visit([...rows, row]);
      }
      return sum;
    }
    let product = 1;
    for (let col = 0; col < n; col++) {
      product *= columns[col]![rows[col]!]!;
      for (let next = col + 1; next < n; next++) {
        if (rows[col]! > rows[next]!) product = -product;
      }
    }
    return product;
  }
  return visit([]);
}

for (const { schema, dimension } of cases) {
  describe(schema.type, () => {
    it('computes identity, zero, triangular, and singular determinants', () => {
      expect(std.determinant(schema.identity())).toBe(1);
      expect(std.determinant(schema())).toBe(0);
      const triangular = Array.from({ length: dimension * dimension }, (_, i) => {
        const col = Math.floor(i / dimension);
        const row = i % dimension;
        return row > col ? 0 : row === col ? row + 2 : 7;
      });
      expect(std.determinant(fromElements(schema, triangular))).toBe(
        Array.from({ length: dimension }, (_, i) => i + 2).reduce((a, b) => a * b, 1),
      );
      expect(std.determinant(fromElements(schema, Array(dimension * dimension).fill(2)))).toBe(0);
    });

    it('computes fractional determinants', () => {
      const matrix = schema.identity();
      matrix.columns[0][0] = -0.5;
      matrix.columns[1][1] = 0.25;
      expect(std.determinant(matrix)).toBe(-0.125);
      expectTypeOf(std.determinant(matrix)).toEqualTypeOf<number>();
    });

    it('flips the sign when two columns are exchanged', () => {
      const columns = schema.identity().columns.map((column) => [...column]);
      [columns[0], columns[1]] = [columns[1]!, columns[0]!];
      expect(std.determinant(fromElements(schema, columns.flat()))).toBe(-1);
    });

    it('agrees with an independent formula on dense matrices without mutating them', () => {
      let seed = 12345;
      for (let sample = 0; sample < 32; sample++) {
        const columns = Array.from({ length: dimension }, () =>
          Array.from({ length: dimension }, () => {
            seed = (1664525 * seed + 1013904223) >>> 0;
            return (seed % 9) - 4;
          }),
        );
        const matrix = fromElements(schema, columns.flat());
        expect(std.determinant(matrix)).toBe(referenceDeterminant(columns));
        expect(matrix.columns.map((column) => [...column])).toEqual(columns);
      }
    });

    it('keeps determinant calls in shaders when their argument is runtime-known', () => {
      const main = tgpu.fn(
        [schema],
        d.f32,
      )((value) => {
        'use gpu';
        return std.determinant(value);
      });
      expect(tgpu.resolve([main])).toBe(
        `fn main(value: ${schema.type}) -> f32 {\n  return determinant(value);\n}`,
      );
    });

    it('evaluates constant matrix determinants while resolving a shader', () => {
      const matrix = schema.identity();
      matrix.columns[0][0] = -2;
      matrix.columns[1][1] = 3;
      const main = tgpu.fn(
        [],
        d.f32,
      )(() => {
        'use gpu';
        return std.determinant(matrix);
      });
      expect(tgpu.resolve([main])).toBe('fn main() -> f32 {\n  return -6f;\n}');
    });
  });
}

it('computes a fractional determinant', () => {
  expect(std.determinant(d.mat2x2f(0.5, 0.25, -2, 3))).toBe(2);
});

it('rejects scalar and vector arguments on the CPU', () => {
  // @ts-expect-error
  expect(() => std.determinant(1)).toThrow('Unsupported signature');
  // @ts-expect-error
  expect(() => std.determinant(d.vec3f())).toThrow('Unsupported signature');
});
