import test from "node:test";
import assert from "node:assert/strict";
import { starPoints } from "../../web/browser/star-points.mjs";

test("standard odd and even boards have symmetric corner stars and only a real center", () => {
  assert.deepEqual(starPoints(15).map(p => [p.row, p.col, p.kind]), [
    [3,3,"star"], [3,11,"star"], [11,3,"star"], [11,11,"star"], [7,7,"tianyuan"],
  ]);
  assert.deepEqual(starPoints(16).map(p => [p.row, p.col]), [[3,3],[3,12],[12,3],[12,12]]);
  assert.deepEqual(starPoints({rows:17,cols:9}).find(p => p.kind === "tianyuan"),
    {row:8,col:4,point:76,kind:"tianyuan"});
});

test("every supported rectangle keeps unique interior marks under both reflections", () => {
  for (let rows = 5; rows <= 32; rows++) for (let cols = 5; cols <= 32; cols++) {
    const marks = starPoints({rows,cols});
    const keys = new Set(marks.map(p => `${p.row},${p.col},${p.kind}`));
    const odd = rows % 2 === 1 && cols % 2 === 1;
    assert.equal(marks.length, odd ? 5 : 4);
    assert.equal(keys.size, marks.length);
    assert.equal(marks.filter(p => p.kind === "tianyuan").length, odd ? 1 : 0);
    for (const p of marks) {
      assert.ok(p.row > 0 && p.row < rows-1 && p.col > 0 && p.col < cols-1);
      assert.equal(p.point, p.row * cols + p.col);
      assert.ok(keys.has(`${rows-1-p.row},${p.col},${p.kind}`));
      assert.ok(keys.has(`${p.row},${cols-1-p.col},${p.kind}`));
    }
  }
});
