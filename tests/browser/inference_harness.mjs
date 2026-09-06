import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import * as core from "../../web/browser/core.mjs";
import { dimensions } from "../../web/browser/geometry.mjs";
import { searchNodeCap, mainNodeCap, SEARCH_BUDGET_VERSION } from "../../web/browser/search-budget.mjs";

// Execute actual Worker infer/init/analyze code with deterministic ORT stand-ins.
// No ONNX model or search is run, and no speed claim is derived from this test.
export const workerSource = readFileSync(new URL("../../web/browser/engine-worker.mjs", import.meta.url), "utf8");
export function harness(options = {}, source = workerSource) {
  const tensors = [], sessions = [], calls = [], created = [], events = [];
  let active = 0, maxActive = 0;
  class Tensor {
    constructor(type, data, dims) {
      this.type = type; this.raw = data; this.dims = Array.from(dims);
      this.disposed = 0;
      tensors.push(this);
    }
    get data() {
      assert.equal(this.disposed, 0, "consumer read a disposed tensor");
      return this.raw;
    }
    dispose() {
      this.disposed++;
      assert.equal(this.disposed, 1, "tensor disposed twice");
      this.raw = undefined;
    }
  }
  const ort = {
    env: { wasm: {} }, Tensor,
    InferenceSession: { async create(url) {
      const role = new URL(url).pathname.split("/").at(-1).replace(".onnx", "");
      created.push(role);
      if (options.createFailure === role) throw Error(`create failed ${role}`);
      const session = {
        role, released: 0,
        async release() {
          this.released++;
          if (options.releaseFailure === role) throw Error(`release failed ${role}`);
        },
        async run(feeds) {
          active++;
          maxActive = Math.max(maxActive, active);
          try {
            // Fail on overlap even if future code accidentally parallelizes ORT.
            await Promise.resolve();
            const copied = Object.fromEntries(Object.entries(feeds).map(([k, v]) =>
              [k, { dims: v.dims.slice(), data: Array.from(v.data, x => typeof x === "bigint" ? Number(x) : x) }]));
            calls.push({ role, feeds: copied });
            if (options.runFailure === role) throw Error(`run failed ${role}`);
            const global = role.startsWith("global"),
              count = global ? feeds.inputs.dims[2] * feeds.inputs.dims[3] : feeds.rgb.dims[0] * 25;
            const logits = Float32Array.from({ length: count }, (_, i) =>
              Math.sin(i * 0.3) + (global ? 0.2 : role === "play" ? 0.4 : -0.3));
            if (options.badOutput === role) logits.fill(NaN);
            return {
              logits: new Tensor("float32", logits, global ? [1, 1, ...feeds.inputs.dims.slice(2)] : [feeds.rgb.dims[0], 25]),
              // Deliberately include an unused output to test complete cleanup.
              value: new Tensor("float32", new Float32Array([0.25]), [1]),
            };
          } finally { active--; }
        },
      };
      sessions.push(session);
      return session;
    } },
  };
  const memory = { buffer: new ArrayBuffer(65536) }, nativeCalls = [], nativeThreatCalls = [];
  const native = {
    memory,
    browser_board: () => 0, browser_priors: () => 2048, browser_output: () => 16384,
    browser_threat_context: () => 20480, browser_threat_output: () => 17408,
    browser_threat_pv: () => 18000,
    native_threat_storage_version: () => options.threatStorageVersion ?? 2,
    native_threat_group_count: () => options.threatGroups ?? 0,
    native_threat_group_capacity: () => options.threatGroupCapacity ?? 65536,
    native_threat_iteration: (context, index) => index === 0 ? 20000 : 0,
    browser_threat_solve(rows, cols, side, milliseconds, nodes, quiet, total, width) {
      return this.browser_threat_solve_range(rows, cols, side, milliseconds, nodes,
        quiet, total, width, quiet === 0 ? 0 : 1);
    },
    browser_threat_solve_range(rows, cols, side, milliseconds, nodes, quiet, total, width, minimumQuiet) {
      nativeThreatCalls.push({ rows, cols, side, milliseconds, nodes, quiet, total, width, minimumQuiet });
      new Int32Array(memory.buffer, 17408, 12).set(options.threatOutput ?? [2, -1, 1, 0, -1, 0, 0, 0, 0, minimumQuiet, 0, 1]);
      new Int32Array(memory.buffer, 20000, 5).set(options.threatIteration ?? [1, minimumQuiet, 5, 1, 1]);
      new Int32Array(memory.buffer, 18000, 64).set(options.threatLine ?? []);
      return options.threatCode ?? 0;
    },
    browser_select(rows, cols, side, milliseconds, nodes, depth, width) {
      nativeCalls.push({ rows, cols, side, milliseconds, nodes, depth, width });
      const board = new Uint8Array(memory.buffer, 0, rows * cols);
      new Int32Array(memory.buffer, 16384, 7).set([board.indexOf(0), 1, 1, 0, 2, 0, 0]);
      return 0;
    },
  };
  const context = {
    ...core, dimensions, searchNodeCap, mainNodeCap, SEARCH_BUDGET_VERSION,
    ort, URL, performance, Uint8Array, Uint32Array, Int32Array,
    Float32Array, Float64Array, BigInt64Array, Map, Set, Promise, console,
    postMessage: (message) => events.push(message), onmessage: null,
    fetch: async () => {
      if (options.fetchFailure) throw Error("fetch failed search wasm");
      return { arrayBuffer: async () => new ArrayBuffer(0) };
    },
    WebAssembly: { instantiate: async () => {
      if (options.nativeFailure) throw Error("instantiate failed search wasm");
      return { instance: { exports: native } };
    } },
  };
  vm.createContext(context);
  const executable = source
    .replace(/^import[\s\S]*?;\r?\n/gm, "")
    .replaceAll("import.meta.url", JSON.stringify("http://test.invalid/engine-worker.mjs"));
  vm.runInContext(executable + `
    globalThis.__test = { ready, infer, models, analyze, nativeMove, nativeThreat, select,
      stubThreat: (fn) => { nativeThreat = fn; },
      stubNative: (fn) => { nativeMove = fn; },
      queue: () => queue,
      stubSearch: (fn) => { select = fn; } };
  `, context, { filename: "web/browser/engine-worker.mjs" });
  return { ...context.__test, context, tensors, sessions, calls, created, events,
    nativeCalls, nativeThreatCalls, maxActive: () => maxActive };
}
