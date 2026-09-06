import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { validate, validateSeconds, facts, legal } from "../../web/browser/core.mjs";
import { dimensions } from "../../web/browser/geometry.mjs";
import { starPoints } from "../../web/browser/star-points.mjs";

// Execute the actual application code. The DOM/Worker stand-ins control
// storage failures and delivery timing, never replace the commit functions.
const source = readFileSync(new URL("../../web/browser/app.mjs", import.meta.url), "utf8")
  .replace(/^import[^\n]+\n/gm, "");
const KEY = "must5.browser.v1";

class Element {
  constructor(id = "") {
    this.id = id;
    this.value = id === "overlay" || id === "edge" ? "none" : "";
    this.style = { setProperty() {} };
    this.classList = { toggle() {}, add() {} };
    this.dataset = {};
    this.attributes = {};
    this.children = [];
    this.textContent = "";
    this.disabled = false;
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  append(e) { this.children.push(e); }
  replaceChildren(...children) { this.children = children; }
}
function record(moves = [136, 137], changes = {}) {
  const size = dimensions({ rows: changes.n ?? 16, cols: changes.cols ?? changes.n ?? 16 });
  const board = Array(size.rows * size.cols).fill(0);
  moves.forEach((p, i) => {
    assert.ok(Number.isInteger(p) && p >= 0 && p < board.length, "fixture move must be in bounds");
    board[p] = 1 + i % 2;
  });
  // Deliberately omit cols unless supplied: original fixtures exercise old saves.
  return { version: 1, n: size.rows, board, history: moves.slice(),
    human: 1, seconds: 1, started: true, ...changes };
}
function storage(saved) {
  const values = new Map(saved ? [[KEY, JSON.stringify(saved)]] : []);
  return {
    values, attempts: [], fail: null,
    getItem(key) {
      if (this.fail === "readback") throw Error("readback denied");
      return values.get(key) ?? null;
    },
    setItem(key, text) {
      this.attempts.push({ key, text });
      if (this.fail === "before") throw Error("QuotaExceededError");
      values.set(key, text);
      if (this.fail === "after") throw Error("write acknowledgement failed");
    },
    saved() { return JSON.parse(values.get(KEY)); },
  };
}
function app(backend = storage(record())) {
  const elements = new Map(), presets = [];
  const get = (id) => {
    if (!elements.has(id)) elements.set(id, new Element(id));
    return elements.get(id);
  };
  for (const seconds of [".5", "1", "3", "5"]) {
    const button = new Element();
    button.dataset.time = seconds;
    presets.push(button);
  }
  class Worker {
    constructor() { Worker.instance = this; this.jobs = []; }
    postMessage(request) { this.jobs.push(JSON.parse(JSON.stringify(request))); }
  }
  const context = {
    validate, validateSeconds, facts, legal, dimensions, starPoints,
    document: { getElementById: get, createElement: () => new Element(),
      querySelectorAll: (selector) => selector === "[data-time]" ? presets : [] },
    localStorage: backend, Worker, navigator: {}, window: {},
    Uint8Array, JSON, Array, Math, Number, String, Error, setTimeout: () => 0,
  };
  vm.createContext(context);
  vm.runInContext(source, context, { filename: "web/browser/app.mjs" });
  const worker = Worker.instance;
  const snapshot = () => JSON.parse(JSON.stringify(context.window.__gomokuSnapshot()));
  const send = (move, id = worker.jobs.at(-1).id) => worker.onmessage({
    data: { type: "result", id, search: { move, depth: 1, nodes: 1, reason: "fixture" },
      value: 0, elapsedMs: 1, overrunMs: 0, windowCount: 16 },
  });
  return {
    backend, get, worker, snapshot, send, presets,
    ready() { worker.onmessage({ data: { type: "ready" } }); },
    click(p) { get("board").children[p].onclick(); },
    retry() { get("retry-save").onclick(); },
  };
}
function readyHuman(backend) {
  const h = app(backend);
  h.ready();
  h.send(119);
  assert.equal(h.snapshot().busy, false);
  return h;
}

test("human move failure stays invisible, blocks alternatives and retries the same action", () => {
  const h = readyHuman();
  const committed = h.snapshot();
  h.backend.fail = "before";
  h.click(119);
  const failed = h.snapshot();
  assert.deepEqual(failed.board, committed.board);
  assert.deepEqual(failed.history, committed.history);
  assert.equal(failed.revision, committed.revision);
  assert.deepEqual(h.backend.saved().history, [136, 137]);
  assert.equal(failed.storageBlocked, true);
  assert.equal(failed.pendingCommit.kind, "human_move");
  assert.equal(failed.pendingCommit.move, 119);
  assert.deepEqual(failed.pendingCommit.record.history, [136, 137, 119]);
  assert.match(failed.storageError, /保存失败/);
  assert.equal(h.get("retry-save").hidden, false);
  assert.equal(h.get("board").children[119].disabled, true);
  const pending = failed.pendingCommit;
  h.click(120);
  h.get("new").onclick();
  h.get("undo").onclick();
  h.presets[2].onclick();
  h.get("overlay").onchange();
  h.worker.onmessage({ data: { type: "ready" } });
  assert.deepEqual(h.snapshot().pendingCommit, pending);
  assert.equal(h.worker.jobs.length, 1);
  assert.equal(h.get("message").textContent, h.snapshot().storageError);
  h.retry();
  assert.deepEqual(h.snapshot().pendingCommit, pending);
  assert.equal(h.worker.jobs.length, 1);
  assert.equal(h.backend.attempts[0].text, h.backend.attempts[1].text);
  h.backend.fail = null;
  h.retry();
  assert.deepEqual(h.snapshot().history, [136, 137, 119]);
  assert.deepEqual(h.backend.saved().history, [136, 137, 119]);
  assert.equal(h.snapshot().revision, committed.revision + 1);
  assert.equal(h.snapshot().storageBlocked, false);
  assert.equal(h.snapshot().pendingCommit, null);
  assert.equal(h.snapshot().storageError, null);
  assert.equal(h.worker.jobs.length, 2);
  assert.equal(h.worker.jobs[1].side, 2);
  assert.equal(h.worker.jobs[1].board[119], 1);
  h.retry();
  assert.equal(h.worker.jobs.length, 2);
});

test("AI move failure keeps the chosen move and retry never searches that turn again", () => {
  const h = readyHuman();
  h.click(119);
  assert.equal(h.worker.jobs.at(-1).side, 2);
  const committed = h.snapshot();
  h.backend.fail = "before";
  h.send(135);
  const failed = h.snapshot();
  assert.deepEqual(failed.history, committed.history);
  assert.equal(failed.board[135], 0);
  assert.equal(failed.revision, committed.revision);
  assert.equal(failed.pendingCommit.kind, "ai_move");
  assert.equal(failed.pendingCommit.move, 135);
  assert.equal(failed.pendingCommit.side, 2);
  assert.deepEqual(h.backend.saved().history, [136, 137, 119]);
  assert.equal(h.worker.jobs.filter(j => j.side === 2).length, 1);
  h.backend.fail = null;
  h.retry();
  assert.deepEqual(h.snapshot().history, [136, 137, 119, 135]);
  assert.equal(h.snapshot().board[135], 2);
  assert.equal(h.worker.jobs.filter(j => j.side === 2).length, 1);
  assert.equal(h.worker.jobs.at(-1).side, 1);
  assert.equal(h.backend.attempts.at(-1).text, h.backend.attempts.at(-2).text);
});

test("a terminal human move is neither shown nor countable before storage succeeds", () => {
  const moves = [136, 0, 137, 2, 138, 4, 139, 6];
  const h = readyHuman(storage(record(moves)));
  h.backend.fail = "before";
  h.click(140);
  assert.equal(facts(h.snapshot().board, 16, 1).winner, 0);
  assert.equal(h.snapshot().history.length, 8);
  assert.equal(h.snapshot().storageBlocked, true);
  assert.equal(h.worker.jobs.length, 1);
  h.backend.fail = null;
  h.retry();
  assert.equal(facts(h.snapshot().board, 16, 1).winner, 1);
  assert.equal(h.snapshot().history.length, 9);
  assert.equal(h.backend.saved().history.length, 9);
  assert.equal(h.snapshot().storageBlocked, false);
  assert.equal(h.worker.jobs.length, 1);
});

test("a terminal AI move is published once after retry with no extra search", () => {
  const moves = [0, 136, 32, 137, 64, 138, 96, 139, 16];
  const h = app(storage(record(moves)));
  h.ready();
  h.backend.fail = "before";
  h.send(140);
  assert.equal(facts(h.snapshot().board, 16, 2).winner, 0);
  assert.equal(h.snapshot().history.length, 9);
  assert.equal(h.snapshot().pendingCommit.move, 140);
  h.backend.fail = null;
  h.retry();
  assert.equal(facts(h.snapshot().board, 16, 2).winner, 2);
  assert.equal(h.snapshot().history.length, 10);
  assert.equal(h.backend.saved().history.length, 10);
  assert.equal(h.worker.jobs.length, 1);
  h.retry();
  assert.equal(h.snapshot().history.length, 10);
  assert.equal(h.worker.jobs.length, 1);
});

test("an uncertain write acknowledgement retries identical data and publishes once", () => {
  const h = readyHuman();
  h.backend.fail = "after";
  h.click(119);
  assert.equal(h.snapshot().history.length, 2);
  assert.equal(h.backend.saved().history.length, 3);
  assert.equal(h.snapshot().storageBlocked, true);
  const text = h.backend.attempts.at(-1).text;
  h.backend.fail = null;
  h.retry();
  assert.equal(h.backend.attempts.at(-1).text, text);
  assert.equal(h.snapshot().history.length, 3);
  assert.equal(h.worker.jobs.filter(j => j.side === 2).length, 1);
});

test("a failed readback never publishes an unconfirmed human move", () => {
  const h = readyHuman();
  h.backend.fail = "readback";
  h.click(119);
  assert.equal(h.snapshot().history.length, 2);
  assert.equal(h.backend.saved().history.length, 3);
  assert.equal(h.snapshot().storageBlocked, true);
  h.backend.fail = null;
  h.retry();
  assert.equal(h.snapshot().history.length, 3);
  assert.equal(h.snapshot().board[119], 1);
});

test("a worker result arriving during a failed setting save is held until retry", () => {
  const h = app(storage(record([136, 137, 119])));
  h.ready();
  assert.equal(h.worker.jobs[0].side, 2);
  h.backend.fail = "before";
  h.get("seconds").value = "3";
  h.get("seconds").onchange();
  assert.equal(h.snapshot().pendingCommit.kind, "time_setting");
  assert.equal(h.snapshot().seconds, 1);
  h.send(135);
  assert.equal(h.snapshot().history.length, 3);
  assert.equal(h.snapshot().board[135], 0);
  assert.equal(h.worker.jobs.length, 1);
  h.backend.fail = null;
  h.retry();
  assert.equal(h.snapshot().seconds, 3);
  assert.equal(h.snapshot().history.length, 4);
  assert.equal(h.snapshot().board[135], 2);
  assert.equal(h.backend.saved().history.length, 4);
  assert.equal(h.worker.jobs.filter(j => j.side === 2).length, 1);
  assert.equal(h.worker.jobs.at(-1).seconds, 3);
});

test("a failed explicit new game keeps the old board; retry discards only stale analysis", () => {
  const h = app(storage(record([136, 137, 119])));
  h.ready();
  h.backend.fail = "before";
  h.get("new").onclick();
  assert.equal(h.snapshot().history.length, 3);
  assert.equal(h.snapshot().started, true);
  assert.equal(h.snapshot().pendingCommit.kind, "new_game");
  h.send(135);
  assert.equal(h.snapshot().history.length, 3);
  h.backend.fail = null;
  h.retry();
  assert.deepEqual(h.snapshot().history, []);
  assert.equal(h.snapshot().started, false);
  assert.equal(h.snapshot().busy, false);
  assert.equal(h.worker.jobs.length, 1);
  assert.deepEqual(h.backend.saved().history, []);
});

test("normal human/AI saves refresh to the exact committed board and do not replay AI", () => {
  const h = readyHuman();
  h.click(119);
  h.send(135);
  const committed = h.snapshot();
  assert.deepEqual(committed.history, [136, 137, 119, 135]);
  assert.equal(committed.storageBlocked, false);
  assert.equal(h.get("retry-save").hidden, true);
  const restored = app(h.backend);
  assert.deepEqual(restored.snapshot().history, committed.history);
  assert.deepEqual(restored.snapshot().board, committed.board);
  assert.equal(restored.snapshot().storageBlocked, false);
  restored.ready();
  assert.equal(restored.worker.jobs.length, 1);
  assert.equal(restored.worker.jobs[0].side, 1);
  assert.deepEqual(restored.worker.jobs[0].board, committed.board);
});

test("reload after an acknowledged-late AI write restores that exact move without replay", () => {
  const h = readyHuman();
  h.click(119);
  h.backend.fail = "after";
  h.send(135);
  assert.equal(h.snapshot().history.length, 3);
  assert.equal(h.snapshot().storageBlocked, true);
  assert.deepEqual(h.backend.saved().history, [136, 137, 119, 135]);
  h.backend.fail = null;
  const restored = app(h.backend);
  assert.deepEqual(restored.snapshot().history, [136, 137, 119, 135]);
  assert.equal(restored.snapshot().board[135], 2);
  restored.ready();
  assert.equal(restored.worker.jobs.length, 1);
  assert.equal(restored.worker.jobs[0].side, 1);
});


test("legacy square saves default cols to n and upgrade only on commit", () => {
  const old = record();
  assert.equal(Object.hasOwn(old, "cols"), false);
  const h = app(storage(old));
  assert.equal(h.snapshot().n, 16);
  assert.equal(h.snapshot().cols, 16);
  assert.equal(h.snapshot().board.length, 256);
  assert.equal(Object.hasOwn(h.backend.saved(), "cols"), false);
  h.ready();
  assert.equal(h.worker.jobs[0].n, 16);
  assert.equal(h.worker.jobs[0].cols, 16);
  h.send(119);
  h.click(119);
  assert.equal(h.backend.saved().cols, 16);
  assert.deepEqual(h.backend.saved().history, [136, 137, 119]);
});

test("8x12 saves preserve flat boundary indices, worker geometry and refresh", () => {
  const h = app(storage(record([52, 53], { n: 8, cols: 12 })));
  assert.equal(h.snapshot().n, 8);
  assert.equal(h.snapshot().cols, 12);
  assert.equal(h.get("board").children.length, 96);
  assert.equal(h.get("board").attributes["aria-rowcount"], "8");
  assert.equal(h.get("board").attributes["aria-colcount"], "12");
  assert.match(h.get("board").children[11].attributes["aria-label"], /^1 行 12 列/);
  assert.match(h.get("board").children[12].attributes["aria-label"], /^2 行 1 列/);
  h.ready(); h.send(11); h.click(11); h.send(12);
  const committed = h.snapshot();
  assert.deepEqual(committed.history, [52, 53, 11, 12]);
  assert.equal(committed.board[11], 1);
  assert.equal(committed.board[12], 2);
  assert.equal(facts(committed.board, { rows: 8, cols: 12 }, 1).winner, 0);
  assert.ok(h.worker.jobs.every(job => job.n === 8 && job.cols === 12 && job.board.length === 96));
  assert.equal(h.backend.saved().cols, 12);
  const restored = app(h.backend);
  assert.deepEqual(restored.snapshot().history, committed.history);
  assert.deepEqual(restored.snapshot().board, committed.board);
  assert.equal(restored.snapshot().n, 8);
  assert.equal(restored.snapshot().cols, 12);
  restored.ready();
  assert.equal(restored.worker.jobs.length, 1);
  assert.equal(restored.worker.jobs[0].cols, 12);
  assert.equal(restored.worker.jobs[0].board.length, 96);
});

test("failed rectangular new game retries the same dimensions and rejects stale AI", () => {
  const h = app(storage(record([136, 137, 119])));
  h.ready();
  const oldRevision = h.snapshot().revision;
  h.get("size").value = "8"; h.get("cols").value = "12";
  h.backend.fail = "before";
  h.get("new").onclick();
  const failed = h.snapshot();
  assert.equal(failed.n, 16);
  assert.equal(failed.cols, 16);
  assert.equal(failed.board.length, 256);
  assert.deepEqual(failed.history, [136, 137, 119]);
  assert.equal(failed.pendingCommit.record.n, 8);
  assert.equal(failed.pendingCommit.record.cols, 12);
  assert.equal(failed.pendingCommit.record.board.length, 96);
  assert.equal(h.backend.saved().n, 16);
  assert.equal(h.backend.saved().board.length, 256);
  h.send(135); // Old 16x16 analysis arrives while the new dimensions are pending.
  h.get("size").value = "10"; h.get("cols").value = "14";
  h.get("cols").onchange();
  assert.equal(h.snapshot().pendingCommit.record.cols, 12);
  h.backend.fail = null; h.retry();
  const committed = h.snapshot();
  assert.equal(committed.n, 8);
  assert.equal(committed.cols, 12);
  assert.equal(committed.board.length, 96);
  assert.ok(committed.board.every(cell => cell === 0));
  assert.deepEqual(committed.history, []);
  assert.equal(committed.revision, oldRevision + 1);
  assert.equal(committed.started, false);
  assert.equal(committed.busy, false);
  assert.equal(committed.pendingCommit, null);
  assert.equal(h.worker.jobs.length, 1);
  assert.equal(h.backend.saved().n, 8);
  assert.equal(h.backend.saved().cols, 12);
  assert.equal(h.backend.saved().board.length, 96);
  h.get("start").onclick();
  assert.equal(h.worker.jobs.at(-1).n, 8);
  assert.equal(h.worker.jobs.at(-1).cols, 12);
  assert.equal(h.worker.jobs.at(-1).board.length, 96);
});

test("star and tianyuan decoration does not alter cells, stones or saved history", () => {
  const h = app(storage(record([], { n: 9, cols: 13, started: false })));
  const snapshot = h.snapshot(), cells = h.get("board").children;
  assert.equal(cells.length, 117);
  assert.equal(cells.filter(cell => cell.dataset.star).length, 5);
  assert.equal(cells[4 * 13 + 6].dataset.star, "tianyuan");
  assert.match(cells[4 * 13 + 6].attributes["aria-label"], /天元/);
  assert.ok(cells.every(cell => cell.className === "cell"));
  assert.equal(cells.flatMap(cell => cell.children).filter(child => child.className.startsWith("stone ")).length, 0);
  assert.ok(snapshot.board.every(cell => cell === 0));
  assert.deepEqual(snapshot.history, []);
  assert.deepEqual(h.backend.saved().history, []);
});

test("rectangle saved buffers must agree with both dimensions", () => {
  const malformed = record([], { n: 8, cols: 12, started: false });
  malformed.board.pop();
  const h = app(storage(malformed));
  assert.match(h.get("message").textContent, /保存内容无法恢复/);
  assert.deepEqual(h.backend.saved(), malformed);
  assert.deepEqual(JSON.parse(h.backend.values.get(KEY + ".recovery")), malformed);
  assert.equal(h.backend.saved().board.length, 95);
  assert.equal(h.snapshot().n, 15);
  assert.equal(h.snapshot().cols, 15);
  assert.equal(h.snapshot().board.length, 225);
  assert.equal(h.get("size").value, 15);
  assert.equal(h.get("cols").value, 15);
});


test("rectangular edge preset survives cold loads and width and height changes", () => {
  const h = app(storage(record([], { started: false })));
  assert.equal(h.get("edge").value, "none");
  h.get("size").value = "8";
  h.get("cols").value = "12";
  h.get("edge").value = "top";
  h.get("edge").onchange();
  assert.equal(h.backend.saved().edge, "top");
  const assertTop = (instance, rows, cols) => {
    const snapshot = instance.snapshot();
    assert.equal(snapshot.n, rows);
    assert.equal(snapshot.cols, cols);
    assert.equal(snapshot.board.length, rows * cols);
    assert.equal(instance.get("edge").value, "top");
    assert.deepEqual(snapshot.board, Array.from({ length: rows * cols }, (_, p) => p < cols ? 3 : 0));
    assert.deepEqual(snapshot.history, []);
    assert.equal(snapshot.started, false);
    assert.equal(instance.backend.saved().edge, "top");
  };
  assertTop(h, 8, 12);
  const reload = app(h.backend);
  assertTop(reload, 8, 12);
  reload.get("cols").value = "14";
  reload.get("cols").onchange();
  assertTop(reload, 8, 14);
  const secondReload = app(h.backend);
  assertTop(secondReload, 8, 14);
  secondReload.get("size").value = "10";
  secondReload.get("size").onchange();
  assertTop(secondReload, 10, 14);
});
