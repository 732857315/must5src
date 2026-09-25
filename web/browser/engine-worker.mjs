import * as ort from "./vendor/ort.wasm.min.mjs";
import {
  validate,
  legal,
  facts,
  centers,
  windowAt,
  encodeRGB,
  maskedSoftmax,
  normalizeFusion,
  globalInput,
  ordered,
  forcing,
  validateSeconds,
} from "./core.mjs";
import { dimensions } from "./geometry.mjs";
import { searchNodeCap, mainNodeCap, SEARCH_BUDGET_VERSION } from "./search-budget.mjs";
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
const models = new Map();
const MAX_GLOBAL_MODELS = 2;
let native;
const ready = (async () => {
  try {
    for (const role of ["opponent", "play", "global"])
      models.set(
        role,
        await ort.InferenceSession.create(
          new URL(`./models/${role}.onnx`, import.meta.url).href,
          { executionProviders: ["wasm"] },
        ),
      );
    const bytes = await (
      await fetch(new URL("./search.wasm", import.meta.url))
    ).arrayBuffer();
    native = (
      await WebAssembly.instantiate(bytes, {
        env: { now: () => performance.now() },
      })
    ).instance.exports;
    return true;
  } catch (error) {
    // A failed later model or WASM load must not retain earlier sessions.
    const sessions = [...models.values()];
    models.clear();
    await Promise.allSettled(sessions.map((session) => session.release()));
    throw error;
  }
})();
ready
  .then(() => postMessage({ type: "ready" }))
  .catch((error) => postMessage({ type: "error", error: String(error) }));
function nativeMove(board, n, side, prior, milliseconds, maxNodes) {
  const { rows, cols } = dimensions(n);
  new Uint8Array(
    native.memory.buffer,
    native.browser_board(),
    board.length,
  ).set(board);
  new Float64Array(
    native.memory.buffer,
    native.browser_priors(),
    board.length,
  ).set(prior);
  const code = native.browser_select(
    rows,
    cols,
    side,
    Math.max(0, milliseconds),
    maxNodes,
    9,
    16,
  );
  if (code) throw Error(`浏览器搜索错误 ${code}`);
  const [move, nodes, depth, score, proof, exhausted, status] = new Int32Array(
    native.memory.buffer,
    native.browser_output(),
    7,
  );
  if (move < 0 || board[move] !== 0) throw Error("搜索返回了非法落点");
  return {
    move,
    nodes,
    depth,
    score,
    value: proof === 2 ? null : proof,
    exhausted: !!exhausted,
    status,
  };
}
// Complete VCF/quiet proofs execute in WASM; the JS implementation remains an
// independent reference in core.mjs. Copy outputs before another call reuses C.
function nativeThreat(board, n, side, { milliseconds, maxNodes, quiet = 2, total = 64, width = 16,
  minimumQuiet = quiet === 0 ? 0 : 1 }) {
  if (!Number.isInteger(quiet) || quiet < 0 || quiet > 4 ||
      !Number.isInteger(minimumQuiet) || (quiet === 0 ? minimumQuiet !== 0 : minimumQuiet < 1 || minimumQuiet > quiet))
    throw Error("威胁搜索阶段范围无效");
  const { rows, cols } = dimensions(n);
  new Uint8Array(native.memory.buffer, native.browser_board(), board.length).set(board);
  const code = native.browser_threat_solve_range(rows, cols, side, Math.max(0, milliseconds),
    maxNodes, quiet, total, width, minimumQuiet);
  if (code) throw Error(`浏览器威胁搜索错误 ${code}`);
  const [value, move, nodes, exhausted, rootId, certificateNodes, certificateEdges,
    pvLength, certifiedReplies, completedQuiet, status, iterationCount] =
    new Int32Array(native.memory.buffer, native.browser_threat_output(), 12);
  if (![1, 2].includes(value) || nodes < 0 || nodes > maxNodes ||
      ![0, 1].includes(exhausted) || move < -1 || move >= board.length ||
      (move >= 0 && board[move] !== 0) || pvLength < 0 || pvLength > total ||
      certificateNodes < 0 || certificateNodes > 2048 || certificateEdges < 0 ||
      certificateEdges > 2048 * 1024 || certificateEdges > certificateNodes * board.length ||
      rootId < -1 || rootId >= certificateNodes ||
      iterationCount < 0 || iterationCount > 32 || certifiedReplies < 0 ||
      certifiedReplies > board.length || completedQuiet < 0 || completedQuiet > quiet ||
      (completedQuiet !== 0 && completedQuiet < minimumQuiet) || ![0, 1, 2, 3, 4].includes(status) ||
      (value === 1 && (exhausted || ![1, 4].includes(status))) ||
      (value === 2 && status === 1) || (!!exhausted !== [2, 3].includes(status)))
    throw Error("威胁搜索返回了无效结果");
  const moves = Array.from(new Int32Array(native.memory.buffer, native.browser_threat_pv(), pvLength));
  if (moves.some(p => p < 0 || p >= board.length) ||
      (value === 1 && move >= 0 && moves[0] !== move) ||
      (value === 2 && (move !== -1 || pvLength || rootId !== -1 ||
        certificateNodes || certificateEdges || certifiedReplies)) ||
      (rootId === -1 && (certificateNodes || certificateEdges || certifiedReplies)))
    throw Error("威胁搜索返回了无效证明线路");
  const replayed = board.slice();
  for (const point of moves) {
    if (replayed[point] !== 0) throw Error("威胁搜索返回了非法证明落点");
    replayed[point] = 1; // Occupancy check, independent of subsequent native reuse.
  }
  if (value === 1 && move === -1 && (status !== 4 || facts(board, n, side).winner !== side))
    throw Error("威胁搜索返回了无效终局证明");
  const context = native.browser_threat_context(), iterations = [];
  const storageVersion = native.native_threat_storage_version(),
    certificateGroups = native.native_threat_group_count(context),
    groupCapacity = native.native_threat_group_capacity(context);
  if (storageVersion !== 2 || !Number.isInteger(groupCapacity) || groupCapacity < 0 || groupCapacity > 65536 ||
      !Number.isInteger(certificateGroups) || certificateGroups < 0 || certificateGroups > groupCapacity ||
      certificateGroups > certificateEdges || (rootId === -1 && certificateGroups))
    throw Error("威胁证书存储信息无效");
  for (let i = 0; i < iterationCount; i++) {
    const pointer = native.native_threat_iteration(context, i);
    if (!pointer) throw Error("威胁搜索缺少迭代记录");
    const [ordering, q, leafMs, usedNodes, completed] = new Int32Array(native.memory.buffer, pointer, 5);
    if (![0, 1].includes(ordering) || q < minimumQuiet || q > quiet || ![5, 50].includes(leafMs) ||
        usedNodes < 0 || ![0, 1].includes(completed)) throw Error("威胁迭代记录无效");
    iterations.push({ ordering: ordering ? "natural" : "quiet_first", quiet: q,
      leafMs, nodes: usedNodes, completed: !!completed });
  }
  if (iterations.length && iterations.reduce((sum, p) => sum + p.nodes, 0) !== nodes)
    throw Error("威胁迭代节点与总数不一致");
  return { value: value === 1 ? 1 : null, move: move < 0 ? null : move, nodes,
    exhausted: !!exhausted, line: moves.map((point, i) => ({ side: i % 2 ? 3 - side : side, move: point })),
    certifiedReplies, iterations, completedQuiet, status, rootId, certificateNodes,
    certificateEdges, certificateGroups, groupCapacity, storageVersion,
    minimumQuiet, engine: "native_full_threat_v2" };
}
// Consumers must return independent arrays/scalars, never Tensor data views.
// WASM CPU runs already release native handles; dispose also drops JS references
// promptly and covers unsuccessful runs or probability decoding.
async function runModel(model, feeds, consume) {
  let outputs;
  try {
    outputs = await model.run(feeds);
    return consume(outputs);
  } finally {
    for (const tensor of new Set([
      ...Object.values(feeds),
      ...Object.values(outputs ?? {}),
    ])) tensor.dispose();
  }
}
async function globalSession(key) {
  // Retain only the two most recently used geometries. Rectangular boards have
  // sixteen variants, each owning a separate native inference session.
  let session = models.get(key);
  if (!session) {
    session = await ort.InferenceSession.create(
      new URL(`./models/${key}.onnx`, import.meta.url).href,
      { executionProviders: ["wasm"] },
    );
  }
  models.delete(key);
  models.set(key, session);
  const globals = [...models.keys()].filter(name => name.startsWith("global"));
  for (const name of globals.slice(0, -MAX_GLOBAL_MODELS)) {
    const previous = models.get(name);
    models.delete(name);
    await previous.release();
  }
  return session;
}
async function infer(board, n, side) {
  const { rows, cols } = dimensions(n);
  const all = centers(board, n),
    coverage = new Uint32Array(board.length),
    oppSum = new Float64Array(board.length),
    playSum = new Float64Array(board.length);
  let count = 0;
  for (let first = 0; first < all.length; first += 32) {
    const views = all
      .slice(first, first + 32)
      .map((p) => windowAt(board, n, p))
      .filter((v) => v.cells.includes(0));
    if (!views.length) continue;
    count += views.length;
    const rgb = new Float32Array(views.length * 75),
      sides = new BigInt64Array(views.length).fill(BigInt(3 - side));
    views.forEach((v, i) => rgb.set(encodeRGB(v.cells), i * 75));
    const reds = await runModel(
      models.get("opponent"),
      {
        rgb: new ort.Tensor("float32", rgb, [views.length, 3, 5, 5]),
        side: new ort.Tensor("int64", sides, [views.length]),
      },
      (output) => views.map((v, i) =>
        maskedSoftmax(output.logits.data.subarray(i * 25, i * 25 + 25), v.cells),
      ),
    );
    views.forEach((v, i) => rgb.set(encodeRGB(v.cells, reds[i]), i * 75));
    sides.fill(BigInt(side));
    await runModel(
      models.get("play"),
      {
        rgb: new ort.Tensor("float32", rgb, [views.length, 3, 5, 5]),
        side: new ort.Tensor("int64", sides, [views.length]),
      },
      (output) => views.forEach((v, i) => {
        const green = maskedSoftmax(
            output.logits.data.subarray(i * 25, i * 25 + 25),
            v.cells,
          ),
          moves = legal(v.cells);
        for (const p of moves) {
          const q = v.indices[p];
          if (q < 0) throw Error("边界进入合法落点");
          coverage[q]++;
          oppSum[q] += reds[i][p] * moves.length;
          playSum[q] += green[p] * moves.length;
        }
      }),
    );
  }
  const opponent = normalizeFusion(oppSum, coverage, board),
    play = normalizeFusion(playSum, coverage, board);
  const tokenRows = Math.min(rows, 8),
    tokenCols = Math.min(cols, 8),
    key = tokenRows === tokenCols
      ? (tokenRows < 8 ? `global${tokenRows}` : "global")
      : `global${tokenRows}x${tokenCols}`;
  const { global, value } = await runModel(
    await globalSession(key),
    {
      inputs: new ort.Tensor(
        "float32",
        globalInput(board, n, side, opponent, play),
        [1, 9, rows, cols],
      ),
    },
    (output) => ({
      global: maskedSoftmax(output.logits.data, board),
      value: output.value.data[0],
    }),
  );
  const combined = global.map((v, i) => 0.7 * v + 0.3 * play[i]);
  return {
    opponent,
    play,
    global,
    combined,
    value,
    coverage,
    windowCount: count,
  };
}
function select(board, n, side, prior, deadline, seconds) {
  const cap = searchNodeCap(seconds);
  let nodes = 0;
  const stages = [];
  const left = () => Math.max(0, deadline - performance.now());
  const available = () => left() > 0 && nodes < cap;
  // A backend result is not publishable proof after the request's deadline.
  // Retain its measured work and raw value for audit, but keep the move unknown.
  const timed = (result, ended) => ended >= deadline
    ? { ...result, value: null, exhausted: true, deadlineExceeded: true,
        reportedValue: result.value ?? null }
    : result;
  const timingFields = r => r.deadlineExceeded
    ? { deadlineExceeded: true, reportedValue: r.reportedValue } : {};
  const probe = (position, actor, ms, limit, purpose, candidate = null) => {
    const allocatedMs = Math.min(ms, left()), allocatedNodes = Math.max(0, Math.min(limit, cap - nodes));
    const started = performance.now();
    const raw = forcing(position, n, actor, {
      milliseconds: allocatedMs, maxNodes: allocatedNodes, depth: 32,
    });
    const ended = performance.now(), r = timed(raw, ended);
    nodes += r.nodes;
    stages.push({ kind: "forcing", purpose, actor, candidate, allocatedMs, allocatedNodes,
      elapsedMs: ended - started, nodes: r.nodes, value: r.value,
      exhausted: r.exhausted, move: r.move, cacheHits: r.cacheHits ?? 0,
      cacheHitNodes: r.cacheHitNodes ?? 0, ...timingFields(r) });
    return r;
  };
  const root = probe(board, side, Math.min(30, left() * 0.09), 5000, "root_attack");
  if (root.value === 1 && root.move !== null)
    return { ...root, nodes, stages, depth: 0, reason: "全盘连续冲四胜线",
      selectedStatus: "proved_win", rejected: [] };
  // Extra threat-search nodes must not enlarge the main alpha-beta allocation.
  const nativeMs = left() / 3,
    nativeNodes = Math.max(0, Math.min(cap - nodes, mainNodeCap(seconds))),
    nativeStarted = performance.now();
  const rawInitial = nativeMove(board, n, side, prior, nativeMs, nativeNodes),
    nativeEnded = performance.now(), initial = timed(rawInitial, nativeEnded);
  nodes += initial.nodes;
  stages.push({ kind: "native", purpose: "main_search", actor: side, candidate: null,
    allocatedMs: nativeMs, allocatedNodes: nativeNodes, elapsedMs: nativeEnded - nativeStarted,
    nodes: initial.nodes, value: initial.value, exhausted: initial.exhausted,
    move: initial.move, depth: initial.depth, ...timingFields(initial) });
  if (initial.value === 1)
    return { ...initial, nodes, stages, reason: "全盘搜索找到强制获胜着",
      selectedStatus: "proved_win", rejected: [] };
  if (initial.value === -1 || initial.value === 0)
    return { ...initial, nodes, stages,
      selectedStatus: initial.value === 0 ? "proved_draw" : "proved_loss",
      reason: initial.value === 0 ? "全盘搜索已证明当前局面为和棋" : "全盘搜索已证明当前局面必败",
      rejected: [] };
  const quiet = (position, actor, ms, purpose, candidate, level) => {
    const minimumQuiet = level === 2 ? 1 : level,
      allocatedMs = Math.min(ms, left()), allocatedNodes = Math.max(0, cap - nodes);
    const started = performance.now();
    const raw = nativeThreat(position, n, actor, {
      milliseconds: allocatedMs, maxNodes: allocatedNodes,
      quiet: level, minimumQuiet,
    });
    const ended = performance.now(), r = timed(raw, ended);
    nodes += r.nodes;
    stages.push({ kind: "quiet", purpose, actor, candidate, allocatedMs, allocatedNodes,
      elapsedMs: ended - started, nodes: r.nodes, value: r.value,
      exhausted: r.exhausted, status: r.status, move: r.move,
      minimumQuiet, quiet: level, completedQuiet: r.completedQuiet,
      certifiedReplies: r.certifiedReplies, iterations: r.iterations, engine: r.engine,
      storageVersion: r.storageVersion, certificateNodes: r.certificateNodes,
      certificateEdges: r.certificateEdges, certificateGroups: r.certificateGroups,
      groupCapacity: r.groupCapacity, cacheHits: r.cacheHits ?? 0,
      cacheHitNodes: r.cacheHitNodes ?? 0, recursiveCacheHits: r.recursiveCacheHits ?? 0,
      forcingCacheHits: r.forcingCacheHits ?? 0, ...timingFields(r) });
    return r;
  };
  const rejected = [], rejectedSet = new Set(), ownAttacks = new Map(),
    candidates = [...new Set([initial.move, ...ordered(board, n, side, prior)])];
  let ownAttackBlocked = false;
  for (const move of candidates) {
    if (rejectedSet.has(move)) continue;
    if (!available())
      return { move, nodes, stages, depth: move === initial.move ? initial.depth : 0,
        value: null, reason: "预算用尽，当前落点尚未完成防守检查",
        selectedStatus: "unexamined_budget_fallback", rejected };
    let reply, selectedStatus = "defense_unknown", candidateRejected = false;
    // Each new level starts at exactly that quiet depth. This is a new depth
    // stage, not restoration of an interrupted DFS or repetition of q1/q2.
    for (let level = 2; level <= 4; level++) {
      board[move] = side;
      try {
        if (level === 2) reply = probe(board, 3 - side, 50, 5000, "candidate_defense", move);
        if ((level !== 2 || reply.value === null) && available())
          reply = quiet(board, 3 - side, left(), "candidate_defense", move, level);
      } finally {
        board[move] = 0;
      }
      if (reply.value === 1) {
        rejectedSet.add(move); rejected.push(move); candidateRejected = true;
        break;
      }
      if (reply.value === -1)
        return { move, nodes, stages, depth: 0, value: 1,
          selectedStatus: "proved_win", reason: "此着之后对手已被证明必败", rejected };
      selectedStatus = reply.value === 0 ? "proved_nonloss" : reply.exhausted
        ? "defense_budget_exhausted_unknown" : "defense_unknown";
      if (!reply.exhausted && reply.status !== 3 && !ownAttackBlocked && available()) {
        // Our attack board is unchanged when a candidate is rejected. Reuse
        // already completed root stages instead of replaying them for its successor.
        let attack = ownAttacks.get(level);
        if (!attack) {
          attack = quiet(board, side, Math.min(200, left()), "root_attack", null, level);
          ownAttacks.set(level, attack);
        }
        if (attack.value === 1 && attack.move !== null) {
          if (rejectedSet.has(attack.move)) throw Error("进攻与防守证明互相矛盾");
          return { ...attack, nodes, stages, depth: 0, reason: "主动布局威胁胜线",
            selectedStatus: "proved_win", rejected };
        }
        if (attack.exhausted || attack.status === 3) ownAttackBlocked = true;
      }
      // A local own-attack timeout/capacity stop blocks further own stages,
      // not deeper defense work while the request still has time and nodes.
      // Never retry an unfinished stage or call its next level complete.
      if (reply.value !== null || reply.exhausted || reply.status === 3 || !available()) break;
    }
    if (candidateRejected) continue;
    return { move, nodes, stages, depth: move === initial.move ? initial.depth : 0,
      value: null, selectedStatus, selectedMoveValue: reply.value === 0 ? 0 : null,
      ownAttackBlocked,
      reason: reply.value === 0 ? "当前落点已证明至少可以保和" : reply.exhausted
        ? "防守检查达到预算，当前落点仍未决" : rejected.length
          ? "已避开证实会输的落点，当前落点仍未决" : "搜索完成，当前落点尚无确定胜负证明",
      rejected };
  }
  return { move: initial.move, nodes, stages, depth: 0, value: -1,
    reason: "全部合法落点都已证实失败", selectedStatus: "proved_loss", rejected };
}

async function analyze(request) {
  await ready;
  const { id, n, cols = n, side } = request,
    size = cols === n ? n : { rows: n, cols },
    seconds = validateSeconds(request.seconds);
  validate(request.board, size, side);
  const board = Uint8Array.from(request.board);
  const f = facts(board, size, side);
  if (f.winner || !board.includes(0))
    return { id, n, cols, type: "result", terminal: true, winner: f.winner };
  const started = performance.now(),
    deadline = started + seconds * 1000;
  const prediction = await infer(board, size, side),
    inferenceMs = performance.now() - started;
  const search = { ...select(board, size, side, prediction.combined, deadline, seconds),
    nodeLimit: searchNodeCap(seconds), budgetVersion: SEARCH_BUDGET_VERSION };
  const elapsedMs = performance.now() - started;
  return {
    id,
    n,
    cols,
    type: "result",
    ...prediction,
    search,
    elapsedMs,
    inferenceMs,
    seconds,
    overrunMs: Math.max(0, elapsedMs - seconds * 1000),
    valueUsedInSearch: false,
  };
}
// Serialize jobs so a reset or quick tap cannot reenter one WASM context.
let queue = Promise.resolve();
onmessage = ({ data }) => {
  if (data.type !== "analyze") return;
  queue = queue
    .then(() => analyze(data))
    .then((result) => postMessage(result))
    .catch((error) =>
      postMessage({ type: "error", id: data.id, error: String(error) }),
    );
};
