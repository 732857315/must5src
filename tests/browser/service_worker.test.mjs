import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const template = readFileSync(new URL("../../web/browser/sw.template.js", import.meta.url), "utf8");
const version = "a".repeat(20), files = ["./index.html", "./app.mjs"];
function worker(scope, storage = new Map(), failInstall = false) {
  const handlers = new Map(), network = [];
  const absolute = request => new URL(typeof request === "string" ? request : request.url, scope).href;
  const caches = {
    async keys() { return [...storage.keys()]; },
    async delete(key) { return storage.delete(key); },
    async open(key) {
      if (!storage.has(key)) storage.set(key, new Map());
      const entries = storage.get(key);
      return {
        async keys() { return [...entries.keys()].map(url => ({ url })); },
        async addAll(names) {
          for (const name of names) {
            entries.set(absolute(name), { body: name });
            if (failInstall) throw Error("download failed");
          }
        },
        async match(request, options = {}) {
          const url = new URL(absolute(request));
          if (options.ignoreSearch) url.search = "";
          return entries.get(url.href);
        },
      };
    },
  };
  const self = { registration: { scope }, location: new URL("sw.js", scope),
    clients: { async claim() {} }, addEventListener: (name, fn) => handlers.set(name, fn) };
  const source = template.replace("__VERSION__", JSON.stringify(version)).replace("__FILES__", JSON.stringify(files));
  vm.runInNewContext(source, { self, caches, URL,
    fetch: async request => { network.push(request.url); return { body: "network" }; } });
  return {
    storage, network,
    key: "must5-browser-" + encodeURIComponent(scope) + "-" + version,
    async dispatch(name, fields = {}) {
      let result;
      handlers.get(name)({ ...fields, waitUntil(promise) { result = promise; },
        respondWith(promise) { result = promise; } });
      return result;
    },
  };
}

test("activation removes this scope's obsolete caches and preserves other deployments", async () => {
  const scope = "https://example.test/must5/", other = "https://example.test/preview/";
  const old = "must5-browser-" + encodeURIComponent(scope) + "-old";
  const foreign = "must5-browser-" + encodeURIComponent(other) + "-old";
  const legacy = "must5-browser-" + "b".repeat(20);
  const shared = "must5-browser-" + "c".repeat(20);
  const storage = new Map([
    [old, new Map()], [foreign, new Map()],
    [legacy, new Map([[scope + "index.html", {}]])],
    [shared, new Map([[scope + "index.html", {}], [other + "index.html", {}]])],
    ["unrelated-site", new Map()],
  ]);
  const h = worker(scope, storage);
  await h.dispatch("install");
  await h.dispatch("activate");
  assert.deepEqual([...storage.keys()].sort(), [foreign, shared, "unrelated-site", h.key].sort());
});

test("failed installation deletes its partial cache without affecting active or foreign caches", async () => {
  const existing = new Map([["active-version", new Map()]]);
  const h = worker("https://example.test/must5/", existing, true);
  await assert.rejects(h.dispatch("install"), /download failed/);
  assert.deepEqual([...existing.keys()], ["active-version"]);
});

test("cached requests and offline navigations stay inside the worker scope", async () => {
  const h = worker("https://example.test/must5/");
  await h.dispatch("install");
  const request = (url, mode = "cors", method = "GET") => h.dispatch("fetch", { request: { url, mode, method } });
  assert.deepEqual(await request("https://example.test/must5/app.mjs?v=1"), { body: "./app.mjs" });
  assert.deepEqual(await request("https://example.test/must5/", "navigate"), { body: "./index.html" });
  assert.deepEqual(await request("https://example.test/must5/missing"), { body: "network" });
  for (const url of ["https://example.test/elsewhere/", "https://example.test/must5-other/", "https://other.test/must5/"])
    assert.equal(await request(url, "navigate"), undefined);
  assert.equal(await request("https://example.test/must5/app.mjs", "cors", "POST"), undefined);
  assert.equal(h.network.length, 1);
});

test("offline readiness requires every declared asset in this scope", async () => {
  const h = worker("https://example.test/must5/");
  await h.dispatch("install");
  let status;
  const check = () => h.dispatch("message", { data: { type: "cache-status" },
    ports: [{ postMessage(value) { status = value; } }] });
  await check();
  assert.equal(status.ready, true);
  assert.equal(status.version, version);
  h.storage.get(h.key).delete("https://example.test/must5/app.mjs");
  await check();
  assert.equal(status.ready, false);
});
