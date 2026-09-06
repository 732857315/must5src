/** Canonical browser board dimensions; numeric sizes retain square compatibility. */
const cache = new Map();
export function dimensions(size) {
  const rows = typeof size === "number" ? size : size?.rows,
    cols = typeof size === "number" ? size : size?.cols;
  if (!(typeof size === "number" || (size !== null && typeof size === "object" && !Array.isArray(size))) ||
      !Number.isInteger(rows) || !Number.isInteger(cols) ||
      rows < 5 || rows > 32 || cols < 5 || cols > 32)
    throw Error("棋盘行数和列数必须是 5 至 32 的整数");
  const key = `${rows}x${cols}`;
  if (!cache.has(key)) cache.set(key, Object.freeze({ rows, cols }));
  return cache.get(key);
}
