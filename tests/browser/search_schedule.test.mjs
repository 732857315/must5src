import test from "node:test";
import assert from "node:assert/strict";
import { harness } from "./inference_harness.mjs";
import { searchNodeCap, mainNodeCap, SEARCH_BUDGET_VERSION } from "../../web/browser/search-budget.mjs";

// Deterministic search/time stand-ins exercise the actual Worker orchestration.
// These checks prove scheduling and status invariants, not playing strength.
async function setup(options = {}) {
  const h = harness();
  await h.ready;
  let now = options.start ?? 20;
  const seconds = options.seconds ?? 1, deadline = seconds * 1000, cap = searchNodeCap(seconds), calls = [];
  let afterMeasuredReturn = 0;
  const board = new Uint8Array(25), original = board.slice();
  const result = (value = null, move = null, nodes = 1, exhausted = false) =>
    ({ value, move, nodes, exhausted, line: [], certifiedReplies: 0, iterations: [] });
  function record(kind, b, side, limits, fn) {
    const move = b.findIndex(v => v === 2);
    assert.ok(limits.milliseconds >= 0 && limits.milliseconds <= deadline - now + 1e-8);
    assert.ok(limits.maxNodes >= 0 && limits.maxNodes <= cap);
    const call = { kind, side, move, at: now, ...limits };
    calls.push(call);
    const spec = fn?.(call) ?? {};
    const elapsed = spec.elapsed ?? 0;
    if (!options.allowBackendOverrun) assert.ok(elapsed <= limits.milliseconds + 1e-8);
    now += elapsed;
    const r = { ...result(spec.value, spec.move, spec.nodes, spec.exhausted),
      status: spec.status ?? (spec.exhausted ? 2 : 0),
      completedQuiet: spec.completedQuiet ?? limits.quiet, iterations: spec.iterations ?? [] };
    afterMeasuredReturn = spec.afterMeasuredReturn ?? 0;
    assert.ok(r.nodes <= limits.maxNodes);
    return r;
  }
  h.context.performance = { now: () => {
    const measured = now; now += afterMeasuredReturn; afterMeasuredReturn = 0; return measured;
  } };
  h.context.ordered = b => Array.from(b.keys()).filter(p => !b[p]);
  h.context.forcing = (b, size, side, limits) => record("forcing", b, side, limits, options.forcing);
  h.stubThreat((b, size, side, limits) => record("quiet", b, side, limits, options.quiet));
  h.stubNative((b, size, side, prior, milliseconds, maxNodes) => ({
    ...record("native", b, side, { milliseconds, maxNodes }, options.native),
    move: options.nativeMove ?? 0, depth: 5,
  }));
  return { board, calls, now: () => now,
    run() {
      const r = h.select(board, 5, 2, new Float64Array(25), deadline, seconds);
      assert.deepEqual(board, original);
      assert.equal(r.nodes, r.stages.reduce((sum, s) => sum + s.nodes, 0));
      assert.ok(r.nodes <= cap);
      if (!options.allowBackendOverrun) assert.ok(now <= deadline + 1e-8);
      assert.equal(board[r.move], 0);
      return r;
    } };
}

test("candidate defense precedes optional own quiet attack and has most of the budget", async () => {
  const h = await setup({
    native: c => ({ elapsed: c.milliseconds, nodes: 4000 }),
    quiet: c => c.side === 1
      ? { elapsed: c.milliseconds, nodes: 8000, exhausted: true }
      : assert.fail("own quiet attack must not preempt candidate defense"),
  });
  const r = h.run(), defense = h.calls.find(c => c.kind === "quiet");
  assert.equal(defense.side, 1);
  assert.equal(defense.move, 0);
  assert.ok(defense.milliseconds > 600);
  assert.equal(r.selectedStatus, "defense_budget_exhausted_unknown");
  assert.equal(r.value, null);
  assert.match(r.reason, /未决/);
});

test("proved bad candidates are rejected; every following probe shares the original deadline", async () => {
  const h = await setup({
    native: c => ({ elapsed: c.milliseconds, nodes: 4000 }),
    quiet: c => c.move < 2
      ? { value: 1, move: 24, elapsed: 200, nodes: 3000 }
      : { elapsed: c.milliseconds, nodes: 1200, exhausted: true },
  });
  const r = h.run();
  assert.deepEqual(Array.from(r.rejected), [0, 1]);
  assert.equal(r.move, 2);
  assert.equal(r.selectedStatus, "defense_budget_exhausted_unknown");
  assert.equal(r.value, null);
  assert.equal(h.calls.filter(c => c.kind === "quiet").length, 3);
});

test("completed defense stays unknown while own attack is blocked and deeper defenses continue", async () => {
  const h = await setup({ quiet: c => c.side === 1 ? { elapsed: 5 }
    : { elapsed: c.milliseconds, exhausted: true } });
  const r = h.run();
  assert.equal(r.value, null);
  assert.equal(r.selectedStatus, "defense_unknown");
  assert.deepEqual(h.calls.filter(c => c.kind === "quiet").map(c => [c.side, c.quiet]),
    [[1, 2], [2, 2], [1, 3], [1, 4]]);
  assert.equal(r.ownAttackBlocked, true);
});

test("an optional completed own win can replace a previously probed unknown move", async () => {
  const h = await setup({ quiet: c => c.side === 1 ? { elapsed: 1 }
    : { value: 1, move: 7, elapsed: 1 } });
  const r = h.run();
  assert.equal(r.move, 7);
  assert.equal(r.value, 1);
  assert.equal(r.selectedStatus, "proved_win");
});

test("the root forcing win returns immediately without native or defense calls", async () => {
  const h = await setup({ forcing: () => ({ value: 1, move: 9 }) });
  const r = h.run();
  assert.equal(r.move, 9);
  assert.equal(r.value, 1);
  assert.equal(r.selectedStatus, "proved_win");
  assert.equal(h.calls.length, 1);
});

test("the native proved win still precedes defense checks", async () => {
  const h = await setup({ nativeMove: 8, native: () => ({ value: 1 }) });
  const r = h.run();
  assert.equal(r.move, 8);
  assert.equal(r.value, 1);
  assert.equal(r.selectedStatus, "proved_win");
  assert.deepEqual(h.calls.map(c => c.kind), ["forcing", "native"]);
});

test("all-legal loss requires a positive opponent proof after every legal move", async () => {
  const h = await setup({ forcing: c => c.side === 1 ? { value: 1, move: 24 } : {} });
  const r = h.run();
  assert.equal(r.value, -1);
  assert.equal(r.selectedStatus, "proved_loss");
  assert.equal(new Set(r.rejected).size, 25);
  assert.deepEqual(h.calls.filter(c => c.side === 1).map(c => c.move), Array.from({length:25}, (_,i)=>i));
});

test("time exhausted after a rejection chooses an unexamined legal move, never a loss proof", async () => {
  const h = await setup({ quiet: c => ({ value: 1, move: 24,
    elapsed: c.milliseconds - 0.5, afterMeasuredReturn: 0.5 }) });
  const r = h.run();
  assert.equal(r.move, 1);
  assert.equal(r.value, null);
  assert.equal(r.selectedStatus, "unexamined_budget_fallback");
  assert.deepEqual(Array.from(r.rejected), [0]);
});

test("node exhausted after a rejection does not allocate another search", async () => {
  const h = await setup({ quiet: c => ({ value: 1, move: 24, nodes: c.maxNodes }) });
  const r = h.run();
  assert.equal(r.move, 1);
  assert.equal(r.nodes, searchNodeCap(1));
  assert.equal(r.value, null);
  assert.equal(r.selectedStatus, "unexamined_budget_fallback");
  assert.equal(h.calls.filter(c => c.kind === "quiet").length, 1);
});

test("a thrown defense search restores the caller board", async () => {
  const h = await setup({ quiet: () => { throw Error("probe failed"); } });
  assert.throws(() => h.run(), /probe failed/);
  assert.ok(h.board.every(v => !v));
});


test("a native full-position loss remains proved rather than becoming an unknown fallback", async () => {
  const h = await setup({ native: () => ({ value: -1 }) });
  const r = h.run();
  assert.equal(r.value, -1);
  assert.equal(r.selectedStatus, "proved_loss");
  assert.equal(r.rejected.length, 0); // Root certificate, not individual guard certificates.
  assert.deepEqual(h.calls.map(c => c.kind), ["forcing", "native"]);
});


test("opponent proved loss after a candidate proves that move wins", async () => {
  const h = await setup({ forcing: c => c.side === 1 ? { value: -1 } : {} });
  const r = h.run();
  assert.equal(r.move, 0);
  assert.equal(r.value, 1);
  assert.equal(r.selectedStatus, "proved_win");
  assert.equal(h.calls.filter(c => c.kind === "quiet").length, 0);
});

test("a proved child draw guarantees nonloss but does not prove the root cannot win", async () => {
  const h = await setup({ forcing: c => c.side === 1 ? { value: 0 } : {},
    quiet: c => ({ elapsed: c.milliseconds, exhausted: true }) });
  const r = h.run();
  assert.equal(r.value, null);
  assert.equal(r.selectedMoveValue, 0);
  assert.equal(r.selectedStatus, "proved_nonloss");
  assert.match(r.reason, /保和/);
});

test("a native complete root draw remains a proved draw", async () => {
  const h = await setup({ native: () => ({ value: 0 }) });
  const r = h.run();
  assert.equal(r.value, 0);
  assert.equal(r.selectedStatus, "proved_draw");
  assert.equal(h.calls.length, 2);
});


test("the larger threat guard does not enlarge the main native node allocation", async () => {
  for (const seconds of [0.1, 1, 30]) {
    const h = await setup({ seconds, native: c => ({ nodes: 1 }),
      quiet: c => ({ elapsed: c.milliseconds, exhausted: true }) });
    h.run();
    const main = h.calls.find(c => c.kind === "native");
    assert.equal(main.maxNodes, mainNodeCap(seconds));
    assert.ok(main.milliseconds <= (seconds * 1000 - main.at) / 3 + 1e-8);
  }
});

test("remaining wall time can pass the old 150k ceiling without repeating native work", async () => {
  const h = await setup({ native: () => ({ elapsed: 300, nodes: 4000 }),
    quiet: c => {
      if (c.side === 2) return { elapsed: 10, nodes: 10 };
      if (c.move === 0 && c.quiet === 2) return { elapsed: 300, nodes: 145998 };
      if (c.move === 0 && c.quiet === 3) return { value: 1, move: 24, elapsed: 10, nodes: 1000 };
      return { elapsed: c.milliseconds, nodes: 20, exhausted: true };
    } });
  const r = h.run();
  assert.ok(r.nodes > 150000 && r.nodes <= searchNodeCap(1));
  assert.equal(r.move, 1);
  assert.deepEqual(Array.from(r.rejected), [0]);
  assert.equal(h.calls.filter(c => c.kind === "native").length, 1);
  const deep = h.calls.find(c => c.kind === "quiet" && c.side === 1 && c.quiet === 3);
  assert.equal(deep.minimumQuiet, 3);
  assert.ok(deep.at < 1000 && deep.maxNodes > 150000);
});

test("bounded q2 unknowns continue q3 and q4 without revisiting finished levels", async () => {
  const h = await setup({ native: () => ({ elapsed: 300, nodes: 4000 }),
    quiet: () => ({ elapsed: 5, nodes: 10 }) });
  const r = h.run(), quiet = h.calls.filter(c => c.kind === "quiet");
  assert.deepEqual(quiet.map(c => [c.side, c.quiet, c.minimumQuiet]),
    [[1, 2, 1], [2, 2, 1], [1, 3, 3], [2, 3, 3], [1, 4, 4], [2, 4, 4]]);
  assert.equal(h.calls.filter(c => c.kind === "native").length, 1);
  assert.equal(h.calls.filter(c => c.kind === "forcing" && c.side === 1).length, 1);
  assert.equal(r.move, 0);
  assert.equal(r.value, null);
  assert.equal(r.selectedStatus, "defense_unknown");
  assert.ok(h.now() < 1000); // Once all supported levels finish, never busy-wait.
});

test("an interrupted deeper defense preserves earlier rejections and the current unknown", async () => {
  const h = await setup({ quiet: c => {
    if (c.side === 1 && c.move === 0) return { value: 1, move: 24, elapsed: 10 };
    if (c.side === 1 && c.quiet === 3) return { elapsed: c.milliseconds, exhausted: true };
    return { elapsed: 5 };
  } });
  const r = h.run();
  assert.equal(r.move, 1);
  assert.deepEqual(Array.from(r.rejected), [0]);
  assert.equal(r.value, null);
  assert.equal(r.selectedStatus, "defense_budget_exhausted_unknown");
  assert.deepEqual(h.calls.filter(c => c.kind === "quiet" && c.side === 1)
    .map(c => [c.move, c.minimumQuiet, c.quiet]), [[0, 1, 2], [1, 1, 2], [1, 3, 3]]);
});

test("a deeper own win is considered only after that level's defense", async () => {
  const h = await setup({ quiet: c => c.side === 2 && c.quiet === 3
    ? { value: 1, move: 7, elapsed: 1 } : { elapsed: 1 } });
  const r = h.run();
  assert.equal(r.move, 7);
  assert.equal(r.value, 1);
  assert.equal(r.selectedStatus, "proved_win");
  assert.deepEqual(h.calls.filter(c => c.kind === "quiet").map(c => [c.side, c.quiet]),
    [[1, 2], [2, 2], [1, 3], [2, 3]]);
});

test("certificate capacity exhaustion stays unknown and is not retried at greater quiet", async () => {
  const h = await setup({ quiet: () => ({ elapsed: 10, nodes: 100, status: 3, exhausted: true }) });
  const r = h.run();
  assert.equal(r.value, null);
  assert.equal(r.selectedStatus, "defense_budget_exhausted_unknown");
  assert.equal(h.calls.filter(c => c.kind === "quiet").length, 1);
  assert.equal(r.stages.find(c => c.kind === "quiet").status, 3);
  assert.ok(h.now() < 1000);
});

test("a positive defense returned after the deadline cannot reject the incumbent", async () => {
  const h = await setup({ allowBackendOverrun: true,
    quiet: c => ({ value: 1, move: 24, elapsed: c.milliseconds + 1 }) });
  const r = h.run();
  assert.equal(r.move, 0);
  assert.equal(r.value, null);
  assert.deepEqual(Array.from(r.rejected), []);
  assert.equal(r.selectedStatus, "defense_budget_exhausted_unknown");
  const late = r.stages.find(c => c.deadlineExceeded);
  assert.equal(late.reportedValue, 1);
  assert.equal(late.value, null);
  assert.equal(h.calls.filter(c => c.kind === "quiet").length, 1);
});

test("a late optional own proof leaves the completed defense unknown", async () => {
  const h = await setup({ allowBackendOverrun: true, quiet: c => c.side === 1
    ? { elapsed: 1 } : { value: 1, move: 7, elapsed: 1001 - c.at } });
  const r = h.run();
  assert.equal(r.move, 0);
  assert.equal(r.value, null);
  assert.equal(r.selectedStatus, "defense_unknown");
  assert.ok(r.stages.some(c => c.purpose === "root_attack" && c.deadlineExceeded));
});

test("a late root proof is suppressed without starting a positive defense claim", async () => {
  const h = await setup({ allowBackendOverrun: true,
    native: c => ({ value: 1, elapsed: 1001 - c.at }) });
  const r = h.run();
  assert.equal(r.value, null);
  assert.equal(r.selectedStatus, "unexamined_budget_fallback");
  assert.equal(h.calls.filter(c => c.kind === "quiet").length, 0);
  assert.ok(r.stages.find(c => c.kind === "native").deadlineExceeded);
});

test("range ABI forwards only the requested quiet stage and storage-v2 identity", async () => {
  const h = harness(); await h.ready;
  const board = new Uint8Array(25);
  const result = h.nativeThreat(board, 5, 1, { milliseconds: 50, maxNodes: 1000,
    quiet: 3, minimumQuiet: 3 });
  assert.equal(h.nativeThreatCalls[0].quiet, 3);
  assert.equal(h.nativeThreatCalls[0].minimumQuiet, 3);
  assert.equal(result.minimumQuiet, 3);
  assert.equal(result.storageVersion, 2);
  assert.equal(result.certificateGroups, 0);
  assert.equal(result.iterations[0].quiet, 3);
  assert.throws(() => h.nativeThreat(board, 5, 1, { milliseconds: 50, maxNodes: 1000,
    quiet: 3, minimumQuiet: 0 }), /阶段范围/);
  assert.throws(() => h.nativeThreat(board, 5, 1, { milliseconds: 50, maxNodes: 1000,
    quiet: 0, minimumQuiet: 1 }), /阶段范围/);
  assert.equal(h.nativeThreatCalls.length, 1);
});

test("normal analyze receipts identify the configured node budget version", async () => {
  const h = harness(); await h.ready;
  h.stubSearch(() => ({ move: 0, nodes: 0, stages: [], value: null }));
  const r = await h.analyze({ id: 1, n: 5, side: 1, board: new Array(25).fill(0), seconds: 1 });
  assert.equal(r.search.nodeLimit, searchNodeCap(1));
  assert.equal(r.search.budgetVersion, SEARCH_BUDGET_VERSION);
  assert.equal(r.valueUsedInSearch, false);
});


test("an own q2 capacity stop does not discard the remaining defense time", async () => {
  const h = await setup({ quiet: c => c.side === 2
    ? { elapsed: 200, nodes: 100, status: 3, exhausted: true }
    : { elapsed: 5, nodes: 10 } });
  const r = h.run();
  assert.equal(r.move, 0);
  assert.equal(r.value, null);
  assert.equal(r.selectedStatus, "defense_unknown");
  assert.equal(r.ownAttackBlocked, true);
  assert.deepEqual(h.calls.filter(c => c.kind === "quiet" && c.side === 1)
    .map(c => [c.minimumQuiet, c.quiet]), [[1, 2], [3, 3], [4, 4]]);
  assert.equal(h.calls.filter(c => c.kind === "quiet" && c.side === 2).length, 1);
  assert.ok(h.now() < 1000);
});

test("completed own stages survive a later rejection without another root search", async () => {
  const h = await setup({ quiet: c => c.side === 1 && c.move === 0 && c.quiet === 3
    ? { value: 1, move: 24, elapsed: 5 } : { elapsed: 5 } });
  const r = h.run();
  assert.equal(r.move, 1);
  assert.equal(r.value, null);
  assert.deepEqual(Array.from(r.rejected), [0]);
  assert.deepEqual(h.calls.filter(c => c.kind === "quiet" && c.side === 2)
    .map(c => [c.minimumQuiet, c.quiet]), [[1, 2], [3, 3], [4, 4]]);
  assert.deepEqual(h.calls.filter(c => c.kind === "quiet" && c.side === 1)
    .map(c => [c.move, c.quiet]), [[0, 2], [0, 3], [1, 2], [1, 3], [1, 4]]);
  assert.equal(h.calls.filter(c => c.kind === "native").length, 1);
});
