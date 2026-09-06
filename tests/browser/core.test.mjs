import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dimensions } from "../../web/browser/geometry.mjs";
import {
  validate,
  validateSeconds,
  segments,
  ordered,
  facts,
  legal,
  centers,
  windowAt,
  encodeRGB,
  maskedSoftmax,
  normalizeFusion,
  globalInput,
  forcing,
  quietThreat,
} from "../../web/browser/core.mjs";
function replay(board, n, side, line) {
  const b = board.slice();
  for (const step of line) {
    assert.equal(facts(b, n, side).winner, 0);
    assert.equal(step.side, side);
    assert.equal(b[step.move], 0);
    b[step.move] = side;
    side = 3 - side;
  }
  return facts(b, n, side).winner;
}
test("2-bit boundary mask and complete coverage on 5,16,19,32 boards", () => {
  for (const n of [5, 16, 19, 32]) {
    const b = new Uint8Array(n * n);
    b[n + 1] = 1;
    b[n * n - 1] = 2;
    for (let c = 2; c < n; c++) b[c] = 3;
    validate(b, n);
    const coverage = new Uint32Array(b.length);
    const cs = centers(b, n);
    assert.equal(new Set(cs).size, cs.length);
    assert.ok(cs.includes(n + 1));
    assert.ok(cs.includes(n * n - 1));
    for (const c of cs) {
      const w = windowAt(b, n, c);
      for (const p of legal(w.cells)) {
        assert.ok(w.indices[p] >= 0);
        coverage[w.indices[p]]++;
      }
    }
    const fused = normalizeFusion(Float64Array.from(coverage), coverage, b);
    for (const p of legal(b)) {
      assert.ok(coverage[p] > 0);
      assert.ok(Math.abs(fused[p] - 1 / legal(b).length) < 1e-12);
    }
    const corner = windowAt(b, n, 0);
    assert.equal(corner.cells[0], 3);
    assert.equal(corner.indices[0], -1);
  }
});
test("probabilities mask stones and forbidden cells; red input remains continuous", () => {
  const b = new Uint8Array(25);
  b[0] = 1;
  b[1] = 2;
  b[2] = 3;
  const probabilities = maskedSoftmax(new Float32Array(25), b);
  assert.deepEqual(Array.from(probabilities.slice(0, 3)), [0, 0, 0]);
  assert.ok(Math.abs(probabilities.reduce((a, b) => a + b) - 1) < 1e-12);
  const rgb = encodeRGB(b, probabilities);
  assert.ok(
    Math.abs(rgb[3] - (1 - probabilities[3] + (230 / 255) * probabilities[3])) <
      1e-7,
  );
  const g = globalInput(b, 5, 2, probabilities, probabilities);
  assert.ok(Math.abs(g[0] - 1) < 1e-7);
  assert.ok(Math.abs(g[1] - 24 / 255) < 1e-7);
  assert.equal(g[6 * 25], -1);
});
test("time settings reject invalid values and preserve seconds", () => {
  assert.equal(validateSeconds("1"), 1);
  assert.equal(validateSeconds(0.1), 0.1);
  assert.equal(validateSeconds(30), 30);
  for (const x of [0, -1, NaN, Infinity, 31, "bad", true, null])
    assert.throws(() => validateSeconds(x));
});
test("all legal replies of a cross threat are retained for either color", () => {
  for (const side of [1, 2]) {
    const b = new Uint8Array(49).fill(3);
    for (let i = 0; i < 7; i++) {
      b[3 * 7 + i] = 0;
      b[i * 7 + 3] = 0;
    }
    b[3 * 7 + 1] = b[3 * 7 + 2] = b[7 + 3] = b[14 + 3] = side;
    b[48] = 0;
    const before = b.slice(),
      r = quietThreat(b, 7, side, { milliseconds: 1000, width: 1, quiet: 1 });
    assert.equal(r.value, 1);
    assert.equal(r.move, 24);
    assert.equal(r.certifiedReplies, 9);
    assert.equal(replay(b, 7, side, r.line), side);
    assert.deepEqual(b, before);
    b[24] = side;
    for (const move of legal(b)) {
      b[move] = 3 - side;
      const proof = forcing(b, 7, side, { milliseconds: 100 });
      assert.equal(proof.value, 1);
      assert.equal(replay(b, 7, side, proof.line), side);
      b[move] = 0;
    }
  }
});
test("double four is a threat for either color, not a forbidden-move rule", () => {
  for (const side of [1, 2]) {
    const b = new Uint8Array(49);
    for (let c = 1; c < 5; c++) b[c] = 3 - side;
    const r = forcing(b, 7, side, { milliseconds: 100 });
    assert.equal(r.value, -1);
    assert.equal(replay(b, 7, side, r.line), 3 - side);
  }
});
test("unknown and exhausted proofs preserve input and node caps", () => {
  const b = new Uint8Array(256);
  b[136] = 1;
  const before = b.slice();
  for (const maxNodes of [0, 1, 3, 10]) {
    const r = quietThreat(b, 16, 2, { milliseconds: 500, maxNodes });
    assert.equal(r.value, null);
    assert.ok(r.nodes <= maxNodes);
    assert.equal(r.iterations.reduce((sum, pass) => sum + pass.nodes, 0), r.nodes);
    assert.equal(r.iterations.at(-1).completed, false);
    assert.deepEqual(b, before);
  }
  assert.equal(forcing(b, 16, 2, { milliseconds: 0 }).value, null);
});
test("deadline-interrupted quiet iterations retain their consumed nodes", () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, "performance");
  let clock = 0;
  Object.defineProperty(globalThis, "performance", {
    configurable: true,
    value: { now: () => ++clock },
  });
  const board = new Uint8Array(256);
  board[136] = 1;
  const before = board.slice();
  try {
    const result = quietThreat(board, 16, 2, { milliseconds: 6, maxNodes: 100 });
    assert.equal(result.value, null);
    assert.equal(result.exhausted, true);
    assert.ok(result.nodes > 0);
    assert.equal(result.iterations.reduce((sum, pass) => sum + pass.nodes, 0), result.nodes);
    assert.equal(result.iterations.at(-1).completed, false);
    assert.deepEqual(board, before);
  } finally {
    Object.defineProperty(globalThis, "performance", original);
  }
});
test("browser loss prefixes retain forcing-to-quiet lines and zero-quiet mandatory defenses", async () => {
  const fixture = JSON.parse(await readFile(new URL("./fixtures/layout_sequence.json", import.meta.url), "utf8"));
  for (const prefix of [20, 22]) {
    for (const swap of [false, true]) {
      const board = new Uint8Array(256);
      for (let i = 0; i < prefix; i++) {
        let [r, c] = fixture.history[i];
        if (swap) [r, c] = [c, 15 - r];
        board[r * 16 + c] = swap ? 2 - (i % 2) : 1 + (i % 2);
      }
      const side = swap ? 2 : 1, before = board.slice();
      const proof = quietThreat(board, 16, side, {milliseconds: 2500, maxNodes: 60000, width: 16, quiet: 1, total: 21});
      assert.equal(proof.value, 1, `prefix ${prefix}, swapped ${swap}`);
      assert.equal(proof.certifiedReplies, 255 - prefix);
      assert.ok(proof.line.length <= 21);
      assert.equal(replay(board, 16, side, proof.line), side);
      assert.deepEqual(board, before);
      assert.ok(proof.nodes <= 60000);
      assert.equal(proof.iterations[0].quiet, 1);
    }
  }
});
test("WebAssembly uses the shared native rules, mask and host deadline", async () => {
  const { instance } = await WebAssembly.instantiate(
    await readFile(new URL("../../web/browser/search.wasm", import.meta.url)),
    { env: { now: () => performance.now() } },
  );
  const e = instance.exports;
  const b = new Uint8Array(e.memory.buffer, e.browser_board(), 256),
    priors = new Float64Array(e.memory.buffer, e.browser_priors(), 256);
  for (const side of [1, 2]) {
    b.fill(0);
    priors.fill(1 / 256);
    for (let i = 0; i < 4; i++) b[8 * 16 + i + 4] = side;
    b[8 * 16 + 3] = 3;
    assert.equal(e.browser_select(16, 16, side, 100, 5000, 9, 16), 0);
    const out = Array.from(
      new Int32Array(e.memory.buffer, e.browser_output(), 7),
    );
    assert.equal(out[0], 8 * 16 + 8);
    assert.equal(out[4], 1);
  }
  b.fill(0);
  b[136] = 1;
  const t = performance.now();
  assert.equal(e.browser_select(16, 16, 2, 20, 150000, 9, 16), 0);
  const out = Array.from(
    new Int32Array(e.memory.buffer, e.browser_output(), 7),
  );
  assert.ok(out[0] >= 0);
  assert.equal(b[out[0]], 0);
  assert.ok(out[1] <= 150000);
  assert.ok(performance.now() - t < 250);
});

test("dimensions accepts square numbers and strict rectangular integer sizes", () => {
  assert.deepEqual(dimensions(16), { rows: 16, cols: 16 });
  assert.deepEqual(dimensions({ rows: 5, cols: 19 }), { rows: 5, cols: 19 });
  assert.deepEqual(dimensions({ rows: 32, cols: 5 }), { rows: 32, cols: 5 });
  for (const size of [null, undefined, "16", true, 4, 33, 5.5, NaN,
    {}, [], { rows: 5 }, { rows: 5, cols: 33 }, { rows: 5.5, cols: 16 },
    { rows: "5", cols: 16 }, { rows: 16, cols: false }])
    assert.throws(() => dimensions(size));
  assert.throws(() => validate(new Uint8Array(95), { rows: 19, cols: 6 }));
});

const RECTANGLES = [
  { rows: 5, cols: 19 }, { rows: 19, cols: 5 },
  { rows: 15, cols: 16 }, { rows: 16, cols: 15 },
];
function independentWinner(board, { rows, cols }) {
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const value = board[r * cols + c];
    if (value !== 1 && value !== 2) continue;
    for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
      if (r + dr * 4 >= rows || c + dc * 4 < 0 || c + dc * 4 >= cols) continue;
      if ([1, 2, 3, 4].every(k => board[(r + dr * k) * cols + c + dc * k] === value))
        return value;
    }
  }
  return 0;
}
function replayRectangle(board, size, actor, line) {
  const position = board.slice();
  for (const step of line) {
    assert.equal(independentWinner(position, size), 0, "no terminal continuation");
    assert.equal(step.side, actor);
    assert.ok(Number.isInteger(step.move) && step.move >= 0 && step.move < position.length);
    assert.equal(position[step.move], 0, "only real empty cells");
    position[step.move] = actor;
    actor = 3 - actor;
  }
  return independentWinner(position, size);
}

test("rectangular line caches include both axes and never wrap rows", () => {
  for (const size of RECTANGLES) {
    const { rows, cols } = size;
    const lines = segments(size);
    assert.equal(lines.length, rows * (cols - 4) + cols * (rows - 4) + 2 * (rows - 4) * (cols - 4));
    assert.equal(new Set(lines.map(line => line.join(","))).size, lines.length);
    for (const line of lines) {
      const coords = line.map(p => [Math.floor(p / cols), p % cols]);
      const dr = coords[1][0] - coords[0][0], dc = coords[1][1] - coords[0][1];
      assert.ok([[0, 1], [1, 0], [1, 1], [1, -1]].some(x => x[0] === dr && x[1] === dc));
      coords.forEach(([r, c], k) => {
        assert.ok(r >= 0 && r < rows && c >= 0 && c < cols);
        assert.deepEqual([r, c], [coords[0][0] + dr * k, coords[0][1] + dc * k]);
      });
    }
    const wrapped = new Uint8Array(rows * cols);
    for (let p = cols - 2; p < cols + 3; p++) wrapped[p] = 1;
    assert.equal(facts(wrapped, size, 1).winner, 0);
  }
  assert.notStrictEqual(segments({ rows: 15, cols: 16 }), segments({ rows: 16, cols: 15 }));
});

test("gray edge windows cover every real empty cell on narrow and transposed rectangles", () => {
  for (const size of RECTANGLES) {
    const { rows, cols } = size, board = new Uint8Array(rows * cols);
    for (let c = 0; c < cols; c++) board[c] = 3;
    for (let r = 0; r < rows; r++) board[r * cols + cols - 1] = 3;
    board[cols + 1] = 1;
    board[(rows - 2) * cols + cols - 2] = 2;
    validate(board, size, 2);
    const selected = centers(board, size), coverage = new Uint32Array(board.length);
    assert.ok(selected.includes(cols + 1));
    assert.ok(selected.includes((rows - 2) * cols + cols - 2));
    assert.equal(new Set(selected).size, selected.length);
    for (const center of selected) {
      const view = windowAt(board, size, center), row = Math.floor(center / cols), col = center % cols;
      for (let i = 0; i < 25; i++) {
        const r = row + Math.floor(i / 5) - 2, c = col + i % 5 - 2;
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
          assert.equal(view.indices[i], -1);
          assert.equal(view.cells[i], 3);
        } else {
          assert.equal(view.indices[i], r * cols + c);
          assert.equal(view.cells[i], board[r * cols + c]);
          if (view.cells[i] === 0) coverage[view.indices[i]]++;
        }
      }
    }
    for (const corner of [0, cols - 1, (rows - 1) * cols, rows * cols - 1]) {
      const view = windowAt(board, size, corner);
      assert.equal(view.indices[12], corner);
      assert.ok(view.indices.includes(-1));
    }
    const probabilities = normalizeFusion(Float64Array.from(coverage), coverage, board);
    for (let p = 0; p < board.length; p++) {
      if (board[p] === 0) {
        assert.ok(coverage[p] > 0);
        assert.ok(Math.abs(probabilities[p] - 1 / legal(board).length) < 1e-12);
      } else assert.equal(probabilities[p], 0);
    }
    const empty = new Uint8Array(rows * cols);
    assert.equal(centers(empty, size)[0], Math.floor(rows / 2) * cols + Math.floor(cols / 2));
  }
});

test("global row and column coordinates use their independent rectangular denominators", () => {
  for (const size of RECTANGLES) {
    const { rows, cols } = size, board = new Uint8Array(rows * cols), count = board.length;
    board[0] = 3; board[cols + 1] = 1; board[count - 2] = 2;
    const probabilities = maskedSoftmax(new Float32Array(count), board);
    const input = globalInput(board, size, 2, probabilities, probabilities);
    assert.equal(input.length, 9 * count);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const p = r * cols + c;
      assert.ok(Math.abs(input[7 * count + p] - (2 * r / (rows - 1) - 1)) < 1e-6);
      assert.ok(Math.abs(input[8 * count + p] - (2 * c / (cols - 1) - 1)) < 1e-6);
      assert.equal(input[5 * count + p], board[p] === 0 ? 1 : 0);
    }
  }
});

test("rectangle immediate wins in all four directions have independently legal proof lines", () => {
  for (const size of RECTANGLES) for (const side of [1, 2]) {
    const { rows, cols } = size;
    for (const cells of [
      [0, 1, 2, 3, 4].map(c => 2 * cols + c),
      [0, 1, 2, 3, 4].map(r => r * cols + 2),
      [0, 1, 2, 3, 4].map(r => r * cols + r),
      [0, 1, 2, 3, 4].map(r => r * cols + 4 - r),
    ]) {
      const board = new Uint8Array(rows * cols).fill(3);
      for (const p of cells.slice(0, 4)) board[p] = side;
      board[cells[4]] = 0;
      const before = board.slice();
      assert.deepEqual(facts(board, size, side).wins, [cells[4]]);
      assert.deepEqual(facts(board, size, 3 - side).blocks, [cells[4]]);
      for (const proof of [
        forcing(board, size, side, { milliseconds: 1000, maxNodes: 10, depth: 0 }),
        quietThreat(board, size, side, { milliseconds: 1000, maxNodes: 10, quiet: 0, total: 1 }),
      ]) {
        assert.equal(proof.value, 1);
        assert.equal(proof.move, cells[4]);
        assert.equal(replayRectangle(board, size, side, proof.line), side);
        assert.deepEqual(board, before);
      }
      board[cells[2]] = 3;
      assert.deepEqual(facts(board, size, side).wins, []);
    }
  }
});

test("rectangular forcing setups retain every defender reply and caller board", () => {
  for (const size of RECTANGLES) for (const side of [1, 2]) {
    const { rows, cols } = size, board = new Uint8Array(rows * cols).fill(3);
    const axis = [0, 1, 2, 3, 4, 5].map(k => cols >= 6 ? 2 * cols + k : k * cols + 2);
    for (const k of [1, 2, 3]) board[axis[k]] = side;
    for (const k of [0, 4, 5]) board[axis[k]] = 0;
    board[board.length - 1] = 0;
    const before = board.slice(), proof = quietThreat(board, size, side, {
      milliseconds: 1000, maxNodes: 1000, quiet: 1, total: 5, width: 16,
    });
    assert.equal(proof.value, 1);
    assert.equal(proof.move, axis[4]);
    assert.equal(replayRectangle(board, size, side, proof.line), side);
    assert.deepEqual(board, before);
    const child = board.slice(); child[proof.move] = side;
    assert.equal(proof.certifiedReplies, legal(child).length);
    for (const response of legal(child)) {
      const after = child.slice(); after[response] = 3 - side;
      assert.equal(independentWinner(after, size), 0, "no ignored counter-five");
      assert.ok(legal(after).some(move => {
        const final = after.slice(); final[move] = side;
        return independentWinner(final, size) === side;
      }), "every defender response leaves a real immediate win");
    }
  }
});

test("rectangle budget interruption preserves both attacker and defender cells", () => {
  for (const size of [{ rows: 15, cols: 16 }, { rows: 16, cols: 15 }]) {
    const board = new Uint8Array(size.rows * size.cols);
    board[Math.floor(size.rows / 2) * size.cols + Math.floor(size.cols / 2)] = 1;
    const before = board.slice();
    for (const maxNodes of [0, 1, 3, 10]) {
      const proof = quietThreat(board, size, 2, { milliseconds: 1000, maxNodes });
      assert.equal(proof.value, null);
      assert.ok(proof.nodes <= maxNodes);
      assert.equal(proof.nodes, proof.iterations.reduce((sum, pass) => sum + pass.nodes, 0));
      assert.deepEqual(board, before);
    }
  }
});

test("numeric 16x16 API remains identical to explicit square dimensions", () => {
  const size = { rows: 16, cols: 16 }, board = new Uint8Array(256);
  board[3] = 3; for (let c = 4; c < 8; c++) board[8 * 16 + c] = 1;
  const probability = maskedSoftmax(new Float32Array(256), board);
  assert.strictEqual(segments(16), segments(size));
  assert.deepEqual(facts(board, 16, 1), facts(board, size, 1));
  assert.deepEqual(ordered(board, 16, 1, probability), ordered(board, size, 1, probability));
  assert.deepEqual(centers(board, 16), centers(board, size));
  assert.deepEqual(windowAt(board, 16, 0), windowAt(board, size, 0));
  assert.deepEqual(globalInput(board, 16, 1, probability, probability),
    globalInput(board, size, 1, probability, probability));
  for (const solve of [forcing, quietThreat]) {
    const a = solve(board, 16, 1), b = solve(board, size, 1);
    assert.equal(a.value, b.value);
    assert.equal(a.move, b.move);
    assert.deepEqual(a.line, b.line);
  }
});


let positiveCacheHarnessPromise;
function positiveCacheHarness() {
  if (!positiveCacheHarnessPromise) positiveCacheHarnessPromise = (async () => {
    let source = await readFile(new URL("../../web/browser/core.mjs", import.meta.url), "utf8");
    source = source.replace(/\r\n/g, "\n");
    source = source.replace('"./geometry.mjs"', JSON.stringify(new URL("../../web/browser/geometry.mjs", import.meta.url).href));
    const once = (before, after) => {
      assert.equal(source.split(before).length, 2, `audit marker: ${before}`);
      source = source.replace(before, after);
    };
    // Record already completed defender loops without changing their choices.
    // Cache entries must retain these opaque ids when reusing a full proof.
    once('const BUDGET = Symbol("budget");', 'export const completedAuditRecords = [];\nconst BUDGET = Symbol("budget");');
    once('          used = 0;', '          used = 0, auditReplies = [];');
    once('            used++;', '            used++;\n            auditReplies.push({move:reply,line:structuredClone(proof.line),auditId:proof.auditId??null});');
    once(`        if (complete && used === replies.length)
          return {
            value: 1,
            move,
            line: [{ side, move }, ...longest],
            certifiedReplies: used,
          };`, `        if (complete && used === replies.length) {
          const auditId=completedAuditRecords.length;
          completedAuditRecords.push({board:Array.from(board),side,move,replies:auditReplies,certifiedReplies:used});
          return {value:1,move,line:[{side,move},...longest],certifiedReplies:used,auditId};
        }`);
    once('    iterations,\n    cacheHits:', '    iterations,\n    auditId: proof?.auditId ?? null,\n    cacheHits:');
    source += '\nexport { forcingWithCache };';
    return import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
  })();
  return positiveCacheHarnessPromise;
}

function withStoppedClock(work) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "performance");
  Object.defineProperty(globalThis, "performance", {configurable:true,value:{now:()=>0}});
  try { return work(); } finally { Object.defineProperty(globalThis, "performance", descriptor); }
}

function forcingCacheBoard() {
  const board = new Uint8Array(35).fill(3);
  for (let i=0;i<6;i++) board[i]=0;
  board[1]=board[2]=board[3]=1;
  return board;
}

test("forcing positive hits consume a node, honor deadlines, and isolate horizon/actor/shape/gray", async () => {
  const {forcingWithCache} = await positiveCacheHarness();
  const board=forcingCacheBoard(), cache=new Map(), size={rows:5,cols:7};
  const options={milliseconds:1000,maxNodes:100,depth:1};
  const first=forcingWithCache(board,size,1,options,cache);
  assert.equal(first.value,1);
  assert.equal(first.cacheHits,0);
  const originalLine=structuredClone(first.line);
  first.line[0].move=-99; // Returning a result must not expose the stored proof.
  const hit=forcingWithCache(board,size,1,options,cache);
  assert.deepEqual(hit.line,originalLine);
  assert.equal(hit.cacheHits,1);
  assert.equal(hit.nodes,1);
  assert.equal(hit.cacheHitNodes,1);
  for (const limits of [{maxNodes:0},{milliseconds:0}]) {
    const blocked=forcingWithCache(board,size,1,{...options,...limits},cache);
    assert.equal(blocked.value,null);
    assert.equal(blocked.nodes,0);
    assert.equal(blocked.cacheHits,0);
  }
  const shallow=forcingWithCache(board,size,1,{...options,depth:0},cache);
  assert.equal(shallow.value,null);
  assert.equal(shallow.cacheHits,0);
  assert.equal(forcingWithCache(board,size,2,options,cache).cacheHits,0);
  assert.equal(forcingWithCache(board,{rows:7,cols:5},1,options,cache).cacheHits,0);
  const changed=board.slice();changed[0]=3;
  assert.equal(forcingWithCache(changed,size,1,options,cache).cacheHits,0);
  assert.deepEqual(board,forcingCacheBoard());
});

test("unknown forcing attempts never occupy the positive cache or poison a later deeper proof", async () => {
  const {forcingWithCache} = await positiveCacheHarness();
  const board=forcingCacheBoard(), cache=new Map(), size={rows:5,cols:7};
  for (let i=0;i<2;i++) {
    const unknown=forcingWithCache(board,size,1,{milliseconds:1000,maxNodes:100,depth:0},cache);
    assert.equal(unknown.value,null);
    assert.equal(unknown.cacheHits,0);
    assert.equal(cache.size,0);
  }
  assert.equal(forcingWithCache(board,size,1,{milliseconds:1000,maxNodes:100,depth:1},cache).value,1);
});

test("public calls never inherit another request's positive proofs", () => {
  const board=forcingCacheBoard(), options={milliseconds:1000,maxNodes:100,depth:1};
  const first=forcing(board,{rows:5,cols:7},1,options);
  const second=forcing(board,{rows:5,cols:7},1,options);
  assert.equal(first.value,1);
  assert.equal(second.value,1);
  assert.equal(second.cacheHits,first.cacheHits);
  assert.equal(second.nodes,first.nodes);
  assert.equal(forcing(board,{rows:5,cols:7},1,{...options,maxNodes:0}).value,null);
  for (const solve of [forcing,quietThreat]) assert.throws(()=>solve(board,{rows:5,cols:7},1,{cacheProofs:"true"}));
});

function v4Prefix11Successor() {
  const board=new Uint8Array(256);
  const moves=[[8,8],[9,7],[10,8],[11,8],[9,9],[11,7],[8,7],[11,9],[11,10],[10,10],[7,8],[10,9]];
  moves.forEach(([r,c],i)=>{board[r*16+c]=1+i%2;});
  return board;
}

// Independent straight-line rules for unlinked forcing leaves. A legal winning
// PV alone is insufficient: the defender must be forced, or already face two
// distinct immediate completions with no counter-win (Python _replay_line).
function certificateWinner(board, size) {
  const {rows,cols}=dimensions(size);
  for (let r=0;r<rows;r++) for (let c=0;c<cols;c++) {
    const side=board[r*cols+c];if (side!==1 && side!==2) continue;
    for (const [dr,dc] of [[0,1],[1,0],[1,1],[1,-1]]) {
      if (r+4*dr>=rows || c+4*dc<0 || c+4*dc>=cols) continue;
      let complete=true;
      for (let k=1;k<5;k++) if (board[(r+k*dr)*cols+c+k*dc]!==side) {complete=false;break;}
      if (complete) return side;
    }
  }
  return 0;
}
function certificateWins(board,size,side) {
  const {rows,cols}=dimensions(size),wins=[];
  for (let p=0;p<board.length;p++) if (board[p]===0) {
    const r=Math.floor(p/cols),c=p%cols;
    for (const [dr,dc] of [[0,1],[1,0],[1,1],[1,-1]]) {
      let count=1;
      for (const sign of [-1,1]) for (let k=1;k<5;k++) {
        const rr=r+sign*k*dr,cc=c+sign*k*dc;
        if (rr<0 || rr>=rows || cc<0 || cc>=cols || board[rr*cols+cc]!==side) break;
        count++;
      }
      if (count>=5) {wins.push(p);break;}
    }
  }
  return wins;
}
function checkForcingContinuation(board,size,winner,line) {
  assert.ok(Array.isArray(line) && line.length>0 && line.length<=board.length);
  const working=board.slice();let actor=winner,forcedDouble=false;
  for (const step of line) {
    assert.equal(certificateWinner(working,size),0,"a continuation cannot move after a terminal position");
    assert.equal(step.side,actor);
    assert.ok(Number.isInteger(step.move) && step.move>=0 && step.move<working.length);
    assert.equal(working[step.move],0);
    if (actor!==winner && !forcedDouble) {
      assert.equal(certificateWins(working,size,actor).length,0,"defender has an immediate counter-win");
      const threats=certificateWins(working,size,winner);
      assert.ok(threats.length>0,"a free defender choice cannot be certified by one PV");
      if (threats.length===1) assert.equal(step.move,threats[0],"PV must include the unique mandatory block");
      else forcedDouble=true; // Every reply leaves another immediate completion.
    }
    working[step.move]=actor;actor=3-actor;
  }
  assert.equal(certificateWinner(working,size),winner);
}

test("independent forcing-leaf audit rejects cooperative replies and defender counter-wins", () => {
  const board=new Uint8Array(49);board[1]=board[2]=board[3]=1;
  assert.throws(()=>checkForcingContinuation(board,7,1,[
    {side:1,move:0},{side:2,move:7},{side:1,move:4},
  ]),/unique mandatory block/);
  const free=board.slice();free[3]=0;
  assert.throws(()=>checkForcingContinuation(free,7,1,[
    {side:1,move:3},{side:2,move:7},{side:1,move:0},{side:2,move:8},{side:1,move:4},
  ]),/free defender choice/);
  const valid=[{side:1,move:4},{side:2,move:0},{side:1,move:5}];
  checkForcingContinuation(board,7,1,valid);
  const counter=board.slice();for (let c=1;c<=4;c++) counter[7+c]=2;
  assert.throws(()=>checkForcingContinuation(counter,7,1,valid),/counter-win/);
});

function checkCompletedAudit(board, side, proof, records) {
  const reached=new Set();
  function check(parent,id) {
    assert.ok(Number.isInteger(id) && id>=0 && id<records.length);
    const record=records[id], after=parent.slice();
    assert.equal(record.side,side);
    assert.equal(after[record.move],0);
    after[record.move]=side;
    assert.deepEqual(Array.from(after),record.board);
    assert.equal(record.certifiedReplies,legal(after).length);
    assert.deepEqual(record.replies.map(r=>r.move).sort((a,b)=>a-b),legal(after));
    if (reached.has(id)) return;
    reached.add(id);
    for (const reply of record.replies) {
      const child=after.slice();assert.equal(child[reply.move],0);child[reply.move]=3-side;
      if (reply.auditId!==null) {
        assert.ok(reply.auditId<id, "cached certificate references an already completed DAG node");
        assert.equal(reply.line[0].move,records[reply.auditId].move);
        assert.equal(replay(child,16,side,reply.line),side);
        check(child,reply.auditId);
      } else {
        checkForcingContinuation(child,16,side,reply.line);
      }
    }
  }
  assert.equal(proof.line[0].move,records[proof.auditId].move);
  check(board,proof.auditId);
  return reached.size;
}

test("recursive/leaf positive reuse retains every completed defender branch and exact node accounting", async () => {
  const harness=await positiveCacheHarness(), board=v4Prefix11Successor(), before=board.slice();
  const options={milliseconds:10000,maxNodes:100000,width:16,quiet:2,total:64};
  const cold=withStoppedClock(()=>quietThreat(board,16,1,{...options,cacheProofs:false}));
  harness.completedAuditRecords.length=0;
  const warm=withStoppedClock(()=>harness.quietThreat(board,16,1,options));
  for (const proof of [cold,warm]) {
    assert.equal(proof.value,1);
    assert.equal(proof.certifiedReplies,243);
    assert.equal(replay(board,16,1,proof.line),1);
    assert.ok(proof.line.length<=64);
    assert.ok(proof.nodes<=options.maxNodes);
    assert.equal(proof.nodes,proof.iterations.reduce((sum,i)=>sum+i.nodes,0));
  }
  assert.equal(cold.cacheHits,0);
  assert.ok(warm.cacheHits>0);
  assert.equal(warm.cacheHits,warm.recursiveCacheHits+warm.forcingCacheHits);
  assert.equal(warm.cacheHitNodes,warm.cacheHits);
  assert.ok(warm.cacheHitNodes<=warm.nodes);
  assert.ok(checkCompletedAudit(board,1,warm,harness.completedAuditRecords)>0);
  assert.deepEqual(board,before);
  // The completed call's cache cannot bypass a new request's zero allowance.
  const exhausted=quietThreat(board,16,1,{...options,maxNodes:0});
  assert.equal(exhausted.value,null);
  assert.equal(exhausted.cacheHits,0);
  assert.equal(exhausted.nodes,0);
});
