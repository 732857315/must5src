import { dimensions } from "./geometry.mjs";

// Four symmetric reference points. Only odd-by-odd boards have an intersection
// at the geometric center. Each axis chooses its own inset for small boards.
export function starPoints(size) {
  const { rows, cols } = dimensions(size);
  const inset = length => length >= 13 ? 3 : length >= 9 ? 2 : 1;
  const r = inset(rows), c = inset(cols), marks = [];
  for (const row of [r, rows - 1 - r]) {
    for (const col of [c, cols - 1 - c]) {
      marks.push({ row, col, point: row * cols + col, kind: "star" });
    }
  }
  if (rows % 2 === 1 && cols % 2 === 1) {
    const row = (rows - 1) / 2, col = (cols - 1) / 2;
    marks.push({ row, col, point: row * cols + col, kind: "tianyuan" });
  }
  return marks;
}
