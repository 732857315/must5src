import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { facts, legal } from '../../web/browser/core.mjs';
import { harness } from './inference_harness.mjs';

function cross() {
  return Uint8Array.from(['333033333','333133333','333133333','011000033',
    '333033333','333033333','333033330'].join(''), Number);
}
async function wasm() {
  return (await WebAssembly.instantiate(await readFile(new URL('../../web/browser/search.wasm', import.meta.url)),
    {env: {now: () => performance.now()}})).instance.exports;
}
function solve(e, board, rows, cols, side, options = {}) {
  const request = {milliseconds: 2000, nodes: 20000, quiet: 1, total: 21, width: 16, ...options};
  new Uint8Array(e.memory.buffer, e.browser_board(), board.length).set(board);
  const minimum = request.minimumQuiet ?? (request.quiet === 0 ? 0 : 1);
  const args = [rows, cols, side, request.milliseconds, request.nodes,
    request.quiet, request.total, request.width];
  assert.equal(request.minimumQuiet === undefined ? e.browser_threat_solve(...args) :
    e.browser_threat_solve_range(...args, minimum), 0);
  const output = Array.from(new Int32Array(e.memory.buffer, e.browser_threat_output(), 12));
  const context = e.browser_threat_context();
  assert.deepEqual(Array.from(new Uint8Array(e.memory.buffer, e.browser_board(), board.length)), Array.from(board));
  assert.deepEqual(Array.from(new Uint8Array(e.memory.buffer, e.native_threat_board(context), board.length)), Array.from(board));
  assert.ok(output[2] <= request.nodes);
  assert.equal(e.native_threat_storage_version(), 2);
  const groups = e.native_threat_group_count(context), groupCapacity = e.native_threat_group_capacity(context);
  assert.ok(groups >= 0 && groups <= groupCapacity && groups <= output[6]);
  assert.ok(output[6] <= output[5] * board.length, 'logical edges use node/cell bounds, not physical capacity');
  const iterations = Array.from({length: output[11]}, (_, i) => {
    const pointer = e.native_threat_iteration(context, i);
    assert.ok(pointer);
    const values = Array.from(new Int32Array(e.memory.buffer, pointer, 5));
    assert.ok(values[1] >= minimum && values[1] <= request.quiet);
    return values;
  });
  if (iterations.length) assert.equal(iterations.reduce((n, it) => n + it[3], 0), output[2]);
  return {output, context, iterations, groups, groupCapacity, line: Array.from(new Int32Array(e.memory.buffer, e.browser_threat_pv(), output[7]))};
}
// Audit all actual defender replies. A leaf PV is valid only under mandatory
// defense or a checked pair of distinct winning cells, never cooperation.
function replay(board, shape, side, line, forcing = false) {
  const copy = board.slice(); let actor = side, fork = false;
  for (const move of line) {
    const f = facts(copy, shape, actor);
    assert.equal(f.winner, 0, 'PV must stop at the first terminal');
    assert.equal(copy[move], 0, 'PV step must be legal');
    if (forcing && actor !== side && !fork) {
      assert.equal(f.wins.length, 0, 'defender must not have an immediate counter-win');
      assert.ok(f.blocks.length, 'leaf cannot assume an unforced defender reply');
      if (f.blocks.length === 1) assert.equal(move, f.blocks[0]);
      else fork = true;
    }
    copy[move] = actor; actor = 3 - actor;
  }
  assert.equal(facts(copy, shape, actor).winner, side);
}
function verifyGraph(e, input, shape, side, result) {
  const {output: o, context, line} = result;
  assert.equal(o[0], 1); assert.ok(o[4] >= 0);
  replay(input, shape, side, line);
  const reached = new Set(), reachedEdges = new Set(), lineCount = e.native_threat_line_count(context);
  let expectedGroups = 0;
  function visit(id, before) {
    assert.ok(id >= 0 && id < o[5]);
    const pointer = e.native_threat_node(context, id); assert.ok(pointer);
    const [move, actor, start, count] = new Int32Array(e.memory.buffer, pointer, 4);
    assert.equal(actor, side); assert.equal(before[move], 0);
    const after = before.slice(); after[move] = side;
    assert.deepEqual(Array.from(new Uint8Array(e.memory.buffer, pointer + 16, before.length)), Array.from(after));
    assert.equal(facts(after, shape, 3-side).winner, 0);
    const expected = new Set(legal(after)), replies = new Set();
    assert.equal(count, expected.size); assert.ok(count > 0);
    assert.ok(start >= 0 && start + count <= o[6]);
    if (reached.has(id)) return;
    reached.add(id);
    const leafSequences = new Map();
    for (let i = 0; i < count; i++) {
      assert.ok(!reachedEdges.has(start+i), 'each logical edge belongs to exactly one parent');
      reachedEdges.add(start+i);
      const edgePointer = e.native_threat_edge(context, start + i); assert.ok(edgePointer);
      // Copy immediately: the next edge getter may overwrite this same scratch pointer.
      const [reply, child, offset, length] = Array.from(new Int32Array(e.memory.buffer, edgePointer, 4));
      assert.ok(expected.has(reply) && !replies.has(reply)); replies.add(reply);
      const next = after.slice(); next[reply] = 3 - side;
      assert.equal(facts(next, shape, side).winner, 0);
      assert.ok(offset >= 0 && length > 0 && offset + length <= lineCount);
      const pv = Array.from(new Int32Array(e.memory.buffer, e.native_threat_lines(context) + 4*offset, length));
      // Shared storage never skips this defender's legal/forcing replay.
      replay(next, shape, side, pv, child === -1);
      if (child >= 0) {
        expectedGroups++;
        assert.ok(child < id, 'postorder IDs prohibit cycles');
        assert.equal(new Int32Array(e.memory.buffer, e.native_threat_node(context, child), 4)[0], pv[0]);
        visit(child, next);
      } else {
        assert.equal(child, -1);
        const key = JSON.stringify(pv);
        if (leafSequences.has(key)) assert.equal(offset, leafSequences.get(key), 'identical parent PVs share one group');
        else leafSequences.set(key, offset);
      }
    }
    assert.deepEqual(replies, expected);
    expectedGroups += leafSequences.size;
  }
  visit(o[4], input);
  assert.equal(reached.size, o[5], 'all returned proof nodes must be reachable');
  assert.equal(reachedEdges.size, o[6], 'all logical defender edges must be reachable');
  assert.equal(expectedGroups, result.groups, 'only same-parent identical complete leaf PVs may be grouped');
  assert.equal(o[8], legal(input).length - 1);
  assert.equal(e.native_threat_node(context, -1), 0);
  assert.equal(e.native_threat_edge(context, -1), 0);
  assert.equal(e.native_threat_node(context, o[5]), 0);
  assert.equal(e.native_threat_edge(context, o[6]), 0);
  assert.equal(e.native_threat_iteration(context, o[11]), 0);
}

test('real WASM proves every defense on a gray rectangle, including a remote legal point', async () => {
  const e = await wasm(), board = cross();
  const result = solve(e, board, 7, 9, 1);
  verifyGraph(e, board, {rows: 7, cols: 9}, 1, result);
  const root = e.native_threat_node(result.context, result.output[4]);
  const [, , first, count] = new Int32Array(e.memory.buffer, root, 4);
  const replies = Array.from({length: count}, (_, i) =>
    new Int32Array(e.memory.buffer, e.native_threat_edge(result.context, first+i), 4)[0]);
  assert.ok(replies.includes(62));
  assert.equal(result.output[8], 9);
  assert.ok(result.groups < result.output[6], 'logical replies exceed the physical group count');
  const firstPointer = e.native_threat_edge(result.context, first);
  const saved = Array.from(new Int32Array(e.memory.buffer, firstPointer, 4));
  const nextPointer = e.native_threat_edge(result.context, first+1);
  assert.equal(nextPointer, firstPointer, 'storage-v2 edge getter exposes one reusable scratch record');
  assert.notEqual(new Int32Array(e.memory.buffer, nextPointer, 4)[0], saved[0]);
  assert.equal(saved[0], replies[0]);
});

test('real WASM handles a transposed rectangle and exchanged colors with complete proofs', async () => {
  const e = await wasm(), original = cross(), board = new Uint8Array(63);
  for (let r = 0; r < 7; r++) for (let c = 0; c < 9; c++) {
    const v = original[r*9+c]; board[c*7+r] = v === 1 || v === 2 ? 3-v : v;
  }
  verifyGraph(e, board, {rows: 9, cols: 7}, 2, solve(e, board, 9, 7, 2));
});

test('real WASM drops interrupted proofs and safely reuses the shared normal-search context', async () => {
  const e = await wasm(), board = cross();
  for (const options of [{milliseconds: 0}, {nodes: 0}, {nodes: 2}]) {
    const {output, line, groups, context} = solve(e, board, 7, 9, 1, options);
    assert.equal(output[0], 2); assert.equal(output[1], -1); assert.equal(output[4], -1);
    assert.equal(output[5], 0); assert.equal(output[6], 0); assert.equal(output[3], 1);
    assert.deepEqual(line, []);
    assert.equal(groups, 0);
    assert.equal(e.native_threat_line_count(context), 0);
    assert.equal(e.native_threat_edge(context, 0), 0);
  }
  verifyGraph(e, board, {rows: 7, cols: 9}, 1, solve(e, board, 7, 9, 1));
  const ordinary = new Uint8Array(80); ordinary.fill(1, 22, 26); ordinary[21] = 3;
  new Uint8Array(e.memory.buffer, e.browser_board(), 80).set(ordinary);
  new Float64Array(e.memory.buffer, e.browser_priors(), 80).fill(0);
  assert.equal(e.browser_select(8, 10, 1, 100, 5000, 9, 16), 0);
  const main = Array.from(new Int32Array(e.memory.buffer, e.browser_output(), 7));
  assert.equal(main[0], 26); assert.equal(main[4], 1);
  verifyGraph(e, board, {rows: 7, cols: 9}, 1, solve(e, board, 7, 9, 1));
});

test('real WASM permits a strict forcing win with zero quiet allowance', async () => {
  const e = await wasm(), board = new Uint8Array(63).fill(3);
  board.fill(0, 27, 34); board.fill(1, 29, 32); board[62] = 0;
  const result = solve(e, board, 7, 9, 1, {quiet: 0, total: 3, nodes: 1000});
  assert.equal(result.output[0], 1); assert.equal(result.output[4], -1);
  assert.equal(result.output[5], 0); assert.equal(result.line.length, 3);
  replay(board, {rows: 7, cols: 9}, 1, result.line, true);
});

test('real WASM later quiet stages do not repeat earlier iterations', async () => {
  const e = await wasm(), board = new Uint8Array(35).fill(3);
  board[0] = board[34] = 0;
  for (const [quiet, minimumQuiet] of [[3,3], [4,4], [4,3]]) {
    const result = solve(e, board, 5, 7, 1, {quiet, minimumQuiet, nodes: 1000, total: 9});
    assert.equal(result.output[0], 2);
    assert.ok(result.iterations.length);
    assert.ok(result.iterations.every(row => row[1] >= minimumQuiet && row[1] <= quiet));
    assert.equal(result.groups, 0);
  }
  for (const [quiet, minimum] of [[0,1], [1,0], [3,4], [4,-1]]) {
    new Uint8Array(e.memory.buffer, e.browser_board(), board.length).set(board);
    assert.equal(e.browser_threat_solve_range(5, 7, 1, 1000, 1000, quiet, 9, 16, minimum), -1);
  }
});

test('Worker threat bridge retains rectangle dimensions and copies PV before native context reuse', async () => {
  const options = {threatOutput: [1, 0, 1, 0, -1, 0, 0, 1, 0, 1, 1, 1], threatLine: [0]}, h = harness(options);
  await h.ready;
  const board = new Uint8Array(63), before = board.slice();
  const result = h.nativeThreat(board, {rows: 7, cols: 9}, 2, {milliseconds: 80, maxNodes: 100});
  options.threatLine = [1]; options.threatOutput[1] = 1;
  h.nativeThreat(board, {rows: 7, cols: 9}, 2, {milliseconds: 80, maxNodes: 100});
  assert.deepEqual(JSON.parse(JSON.stringify(result.line)), [{side: 2, move: 0}]);
  assert.deepEqual(board, before);
  assert.deepEqual(h.nativeThreatCalls[0], {rows: 7, cols: 9, side: 2, milliseconds: 80,
    nodes: 100, quiet: 2, total: 64, width: 16, minimumQuiet: 1});
});

test('Worker rejects illegal, partial, or inconsistent native proof messages', async () => {
  for (const change of [{2: 101}, {10: 9}, {4: 0}, {1: 3}, {7: 1}, {11: 2}, {2: 2}]) {
    const output = [2, -1, 1, 0, -1, 0, 0, 0, 0, 1, 0, 1];
    for (const [index, value] of Object.entries(change)) output[index] = value;
    const h = harness({threatOutput: output}); await h.ready;
    assert.throws(() => h.nativeThreat(new Uint8Array(63), {rows: 7, cols: 9}, 1,
      {milliseconds: 80, maxNodes: 100}), /无效|一致|缺少/);
  }
  const h = harness({threatCode: -2}); await h.ready;
  assert.throws(() => h.nativeThreat(new Uint8Array(63), {rows: 7, cols: 9}, 1,
    {milliseconds: 80, maxNodes: 100}), /搜索错误/);
});
