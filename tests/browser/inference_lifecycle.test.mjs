import test from "node:test";
import assert from "node:assert/strict";
import { Tensor as OrtTensor } from "../../web/browser/vendor/ort.wasm.min.mjs";
import { harness } from "./inference_harness.mjs";

function position(rows = 16, cols = rows) {
  const board = new Uint8Array(rows * cols);
  board[Math.floor(rows / 2) * cols + Math.floor(cols / 2)] = 1;
  board[0] = 3;
  return board;
}
function allDisposed(h) {
  assert.ok(h.tensors.length > 0);
  for (const tensor of h.tensors) assert.equal(tensor.disposed, 1);
}
function checkPrediction(result, board) {
  assert.equal(result.value, 0.25);
  for (const key of ["opponent", "play", "global", "combined"]) {
    const data = result[key];
    assert.equal(data.length, board.length);
    assert.ok(Math.abs(data.reduce((a, b) => a + b, 0) - 1) < 1e-12);
    data.forEach((p, i) => assert.ok(board[i] ? p === 0 : p > 0));
  }
  board.forEach((v, i) => assert.ok(v ? result.coverage[i] === 0 : result.coverage[i] > 0));
}

test("successful repeated infer disposes all feeds/outputs and keeps independent probabilities", async () => {
  const h = harness();
  await h.ready;
  const board = position(), original = board.slice();
  const first = await h.infer(board, 16, 2);
  allDisposed(h);
  checkPrediction(first, board);
  const firstCopy = first.combined.slice(), runs = h.calls.length;
  const second = await h.infer(board, 16, 2);
  allDisposed(h);
  assert.deepEqual(first.combined, firstCopy);
  assert.deepEqual(second, first);
  assert.deepEqual(board, original);
  assert.deepEqual(h.created, ["opponent", "play", "global"]);
  assert.equal(h.calls.length, 2 * runs);
  assert.equal(h.sessions.reduce((n, s) => n + s.released, 0), 0);
  for (const call of h.calls) {
    if (call.role === "opponent") assert.ok(call.feeds.side.data.every(s => s === 1));
    if (call.role === "play") assert.ok(call.feeds.side.data.every(s => s === 2));
  }
});
for (const role of ["opponent", "play", "global"]) {
  test(`${role} run error disposes every created tensor`, async () => {
    const h = harness({ runFailure: role });
    await h.ready;
    await assert.rejects(h.infer(position(), 16, 2), new RegExp(`run failed ${role}`));
    allDisposed(h);
  });
  test(`${role} invalid logits still dispose feeds and all returned outputs`, async () => {
    const h = harness({ badOutput: role });
    await h.ready;
    await assert.rejects(h.infer(position(), 16, 2), /非有限/);
    allDisposed(h);
  });
}
for (const options of [
  { createFailure: "opponent" }, { createFailure: "play" }, { createFailure: "global" },
  { fetchFailure: true }, { nativeFailure: true },
  { nativeFailure: true, releaseFailure: "opponent" },
]) {
  test(`initialization rollback attempts every earlier session release: ${JSON.stringify(options)}`, async () => {
    const h = harness(options);
    await assert.rejects(h.ready, /create failed|fetch failed|instantiate failed/);
    assert.equal(h.models.size, 0);
    for (const session of h.sessions) assert.equal(session.released, 1);
    assert.ok(!h.events.some(e => e.type === "ready"));
  });
}
test("rectangle inference selects per-axis global tokens and caches only sessions", async () => {
  const h = harness();
  await h.ready;
  for (const [rows, cols] of [[5, 9], [9, 5], [6, 12], [12, 6], [7, 16], [16, 7], [16, 19]]) {
    const board = position(rows, cols), shape = { rows, cols };
    const result = await h.infer(board, shape, 2);
    checkPrediction(result, board);
    const call = h.calls.at(-1), tr = Math.min(rows, 8), tc = Math.min(cols, 8);
    assert.equal(call.role, tr === tc ? (tr < 8 ? `global${tr}` : "global") : `global${tr}x${tc}`);
    assert.deepEqual(call.feeds.inputs.dims, [1, 9, rows, cols]);
  }
  assert.deepEqual(h.created, ["opponent", "play", "global", "global5x8", "global8x5", "global6x8", "global8x6", "global7x8", "global8x7"]);
  allDisposed(h);
});
test("rectangle native bridge supplies real rows/cols and leaves input board unchanged", async () => {
  const h = harness();
  await h.ready;
  const board = position(6, 9), copy = board.slice();
  const result = h.nativeMove(board, { rows: 6, cols: 9 }, 2, new Float64Array(54), 1, 10);
  assert.equal(board[result.move], 0);
  assert.deepEqual(board, copy);
  assert.deepEqual(h.nativeCalls[0], { rows: 6, cols: 9, side: 2, milliseconds: 1, nodes: 10, depth: 9, width: 16 });
});
test("queued requests initialize once, serialize infer, and return rectangle dimensions", async () => {
  const h = harness(), sizes = [];
  h.stubSearch((board, size) => { sizes.push(size); return { move: board.indexOf(0) }; });
  for (const [id, n, cols] of [[1, 16, undefined], [2, 6, 9]]) {
    h.context.onmessage({ data: { type: "analyze", id, n, ...(cols ? { cols } : {}),
      board: position(n, cols), side: 2, seconds: 1 } });
  }
  await h.queue();
  const results = h.events.filter(e => e.type === "result");
  assert.deepEqual(results.map(e => [e.id, e.n, e.cols]), [[1, 16, 16], [2, 6, 9]]);
  assert.deepEqual(JSON.parse(JSON.stringify(sizes)), [16, { rows: 6, cols: 9 }]);
  assert.equal(h.maxActive(), 1);
  assert.deepEqual(h.created, ["opponent", "play", "global", "global6x8"]);
  allDisposed(h);
});
test("terminal rectangle skips model inference and reports dimensions", async () => {
  const h = harness();
  await h.ready;
  const board = new Uint8Array(54); board.fill(1, 9, 14);
  const result = await h.analyze({ id: 1, n: 6, cols: 9, side: 2, seconds: 1, board });
  assert.equal(result.terminal, true);
  assert.equal(result.winner, 1);
  assert.equal(result.n, 6);
  assert.equal(result.cols, 9);
  assert.equal(h.calls.length, 0);
});
test("malformed rectangle requests fail before any inference", async () => {
  const h = harness();
  await h.ready;
  for (const changes of [{ cols: 4 }, { cols: 33 }, { cols: 9.5 }, { n: 5 }, { cols: null }]) {
    await assert.rejects(h.analyze({ id: 1, n: 6, cols: 9, side: 2, seconds: 1, board: position(6, 9), ...changes }), /无效|尺寸|行数/);
  }
  assert.equal(h.calls.length, 0);
});

test("lazy rectangle variant load failure leaves base sessions reusable without registering a bad model", async () => {
  const options = { createFailure: "global5x8" }, h = harness(options);
  await h.ready;
  await assert.rejects(h.infer(position(5, 9), { rows: 5, cols: 9 }, 2), /create failed global5x8/);
  assert.equal(h.models.has("global5x8"), false);
  assert.equal(h.models.size, 3);
  allDisposed(h);
  options.createFailure = null;
  const prediction = await h.infer(position(5, 9), { rows: 5, cols: 9 }, 2);
  checkPrediction(prediction, position(5, 9));
  assert.equal(h.models.has("global5x8"), true);
  assert.equal(h.sessions.reduce((n, s) => n + s.released, 0), 0);
  allDisposed(h);
});

test("bundled ORT CPU Tensor disposal releases its reference without detaching reusable batch data", () => {
  const data = new Float32Array([1, 2, 3]), original = data.slice();
  const tensor = new OrtTensor("float32", data, [3]);
  tensor.dispose();
  assert.equal(tensor.location, "none");
  assert.throws(() => tensor.data, /disposed/);
  assert.deepEqual(data, original);
  data[0] = 4;
  const reused = new OrtTensor("float32", data, [3]);
  assert.equal(reused.data[0], 4);
  reused.dispose();
});

test("all sixteen token geometries retain per-axis model identity", async () => {
  const h = harness();
  await h.ready;
  for (const rows of [5, 6, 7, 8]) for (const cols of [5, 6, 7, 8]) {
    const prediction = await h.infer(position(rows, cols), { rows, cols }, 2);
    const expected = rows === cols ? (rows === 8 ? "global" : `global${rows}`) : `global${rows}x${cols}`;
    assert.equal(h.calls.at(-1).role, expected);
    checkPrediction(prediction, position(rows, cols));
  }
  assert.equal(h.models.size, 18); // Two local sessions and sixteen global variants.
  assert.equal(new Set(h.created).size, h.created.length);
  allDisposed(h);
});
test("a failed queued inference reports its id and allows the next request to run", async () => {
  const options = { runFailure: "play" }, h = harness(options);
  h.stubSearch(board => ({ move: board.indexOf(0) }));
  const request = { type: "analyze", n: 16, board: position(), side: 2, seconds: 1 };
  h.context.onmessage({ data: { ...request, id: 1 } });
  await h.queue();
  const error = h.events.find(e => e.type === "error");
  assert.equal(error.id, 1);
  assert.match(error.error, /run failed play/);
  allDisposed(h);
  options.runFailure = null;
  h.context.onmessage({ data: { ...request, id: 2 } });
  await h.queue();
  assert.equal(h.events.filter(e => e.type === "result").length, 1);
  assert.equal(h.events.at(-1).id, 2);
  assert.deepEqual(h.created, ["opponent", "play", "global"]);
  allDisposed(h);
});
