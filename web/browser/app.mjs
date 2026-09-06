import { validate, validateSeconds, facts, legal } from "./core.mjs";
import { starPoints } from "./star-points.mjs";
const $ = (id) => document.getElementById(id),
  KEY = "must5.browser.v1";
let board = new Uint8Array(225),
  n = 15,
  cols = 15,
  edgePreset = "none",
  history = [],
  human = 1,
  seconds = 1,
  started = false,
  ready = false,
  busy = false,
  revision = 0,
  analysis = null,
  zoom = false,
  storageError = null,
  pendingCommit = null,
  deferredResult = null;
function boardSize() {
  return { rows: n, cols };
}
function storageBlocked() {
  return pendingCommit !== null || storageError !== null;
}
function say(text) {
  $("message").textContent = storageError || text;
}
function savedRecord(changes = {}) {
  return {
    version: 1,
    n,
    cols,
    edge: edgePreset,
    board: Array.from(board),
    history: history.slice(),
    human,
    seconds,
    started,
    ...changes,
  };
}
function pendingSnapshot() {
  if (!pendingCommit) return null;
  return {
    kind: pendingCommit.kind,
    move: pendingCommit.move,
    side: pendingCommit.side,
    baseRevision: pendingCommit.baseRevision,
    targetRevision: pendingCommit.targetRevision,
    record: JSON.parse(pendingCommit.serialized),
  };
}
function retryStorage() {
  if (!pendingCommit) return false;
  const pending = pendingCommit;
  try {
    localStorage.setItem(KEY, pending.serialized);
    if (localStorage.getItem(KEY) !== pending.serialized)
      throw Error("保存后读取的内容不一致");
  } catch (error) {
    storageError = `保存失败，对局已暂停，棋盘仍显示上次已保存的状态。请保留本页，排除存储故障后点击“重试保存”；待提交动作不会重新搜索。原因：${error.message || String(error)}`;
    say(storageError);
    draw();
    return false;
  }
  // Publish only the exact record acknowledged by storage. Retrying an AI
  // action reuses its already chosen move, never another search result.
  const record = JSON.parse(pending.serialized);
  ({ n, human, seconds, started } = record);
  cols = record.cols ?? n;
  edgePreset = record.edge ?? "none";
  board = Uint8Array.from(record.board);
  history = record.history.slice();
  revision = pending.targetRevision;
  if (pending.clearAnalysis) analysis = null;
  pendingCommit = null;
  storageError = null;
  $("size").value = n;
  $("cols").value = cols;
  $("edge").value = edgePreset;
  $("human").value = human;
  $("seconds").value = seconds;
  say("保存成功。");
  draw();
  pending.after?.();
  if (deferredResult) {
    const data = deferredResult;
    deferredResult = null;
    handleWorkerMessage({ data });
  }
  return true;
}
function commitRecord(record, kind, after, options = {}) {
  if (storageBlocked()) {
    say(storageError);
    return false;
  }
  pendingCommit = {
    kind,
    move: options.move ?? null,
    side: options.side ?? null,
    serialized: JSON.stringify(record),
    baseRevision: revision,
    targetRevision: revision + (options.bumpRevision === false ? 0 : 1),
    clearAnalysis: options.clearAnalysis !== false,
    after,
  };
  return retryStorage();
}
function afterMove() {
  if (!terminal()) think();
  else say($("turn").textContent + "。可以悔棋或开始新局。");
}
function commitMove(p, kind) {
  const side = turn(),
    nextBoard = board.slice();
  nextBoard[p] = side;
  return commitRecord(
    savedRecord({ board: Array.from(nextBoard), history: [...history, p] }),
    kind,
    afterMove,
    { move: p, side },
  );
}
function load() {
  let raw;
  try {
    raw = localStorage.getItem(KEY);
    if (!raw) return;
    const s = JSON.parse(raw);
    const savedSize = { rows: s.n, cols: s.cols ?? s.n };
    validate(s.board, savedSize, s.human);
    validateSeconds(s.seconds);
    if (
      s.version !== 1 ||
      !["none", "top", "bottom", "left", "right", "all"].includes(s.edge ?? "none") ||
      typeof s.started !== "boolean" ||
      !Array.isArray(s.history)
    )
      throw Error("保存格式错误");
    const replay = Uint8Array.from(s.board, (v) => (v === 3 ? 3 : 0));
    for (let i = 0; i < s.history.length; i++) {
      const p = s.history[i];
      if (
        !Number.isInteger(p) ||
        p < 0 ||
        p >= replay.length ||
        replay[p] !== 0 ||
        facts(replay, savedSize, 1).winner
      )
        throw Error("保存棋谱不合法");
      replay[p] = 1 + (i % 2);
    }
    if (
      (!s.started && s.history.length) ||
      replay.some((v, i) => v !== s.board[i])
    )
      throw Error("棋盘与棋谱不一致");
    ({ n, human, seconds, started } = s);
    cols = savedSize.cols;
    edgePreset = s.edge ?? "none";
    board = replay;
    history = s.history.slice();
  } catch (error) {
    if (raw)
      try {
        localStorage.setItem(KEY + ".recovery", raw);
      } catch {}
    say(`保存内容无法恢复：${error.message}`);
  }
}
load();
$("size").value = n;
$("cols").value = cols;
$("edge").value = edgePreset;
$("human").value = human;
$("seconds").value = seconds;
function turn() {
  return 1 + (history.length % 2);
}
function terminal() {
  return facts(board, boardSize(), turn()).winner || (!board.includes(0) ? 3 : 0);
}
function draw() {
  const outcome = terminal(),
    blocked = storageBlocked();
  $("turn").textContent = blocked
    ? "保存失败 · 对局已暂停"
    : outcome === 3
      ? "和棋"
      : outcome
        ? `${outcome === 1 ? "黑" : "白"}方获胜`
        : !started
          ? "设置棋盘，然后开始"
          : `${turn() === 1 ? "黑" : "白"}方 · ${turn() === human ? "轮到你" : "AI 思考中"}`;
  for (const id of ["board", "board-scroll"]) {
    $(id).style.setProperty("--rows", n);
    $(id).style.setProperty("--cols", cols);
  }
  $("board-scroll").classList.toggle("zoomed", zoom);
  $("board").setAttribute("aria-rowcount", n);
  $("board").setAttribute("aria-colcount", cols);
  $("board").setAttribute("aria-label", `${n} 行 ${cols} 列五子棋棋盘`);
  const stars = new Map(starPoints(boardSize()).map(mark => [mark.point, mark.kind]));
  $("board").classList.toggle("zoomed", zoom);
  const mode = $("overlay").value,
    policy = analysis?.[mode],
    max = policy ? Math.max(...policy) : 0,
    red = mode === "opponent";
  $("legend-policy").textContent =
    mode === "none" ? "不显示预测" : red ? "红：对手下一步" : "绿：推荐落子";
  $("scale").replaceChildren(
    ...Array.from({ length: 25 }, (_, i) => {
      const el = document.createElement("i");
      el.style.background = color((i + 1) / 25, red);
      return el;
    }),
  );
  const nodes = [];
  for (let p = 0; p < board.length; p++) {
    const b = document.createElement("button");
    b.className = "cell";
    b.type = "button";
    b.dataset.point = p;
    b.setAttribute("role", "gridcell");
    const name = ["空位", "黑棋", "白棋", "禁下"][board[p]],
      pos = `${Math.floor(p / cols) + 1} 行 ${(p % cols) + 1} 列`;
    const star = board[p] !== 3 ? stars.get(p) : null;
    const markName = star === "tianyuan" ? "天元" : star ? "星位" : "";
    b.setAttribute("aria-label", `${pos}，${name}${markName ? "，" + markName : ""}`);
    if (star) {
      const mark = document.createElement("span");
      mark.className = `star-point ${star}`;
      mark.setAttribute("aria-hidden", "true");
      b.dataset.star = star;
      b.append(mark);
    }
    b.title = `${pos} · ${name}${policy && board[p] === 0 ? " · 偏好 " + (policy[p] * 100).toFixed(2) + "%" : ""}`;
    if (board[p] === 1 || board[p] === 2) {
      const stone = document.createElement("span");
      stone.className = `stone ${board[p] === 1 ? "black" : "white"}${history.at(-1) === p ? " last" : ""}`;
      b.append(stone);
    } else if (board[p] === 3) b.classList.add("forbidden");
    else if (max > 0)
      b.style.setProperty(
        "--heat",
        color(Math.ceil((policy[p] / max) * 25) / 25, red),
      );
    if (board[p] === 0 && analysis?.search?.move === p && mode !== "none")
      b.classList.add("recommended");
    b.disabled = blocked;
    b.onclick = () => clickCell(p);
    nodes.push(b);
  }
  $("board").replaceChildren(...nodes);
  $("start").hidden = started;
  $("start").disabled = !ready || busy || !!outcome || blocked;
  $("undo").disabled = !history.length || busy || blocked;
  $("new").disabled = blocked;
  $("seconds").disabled = blocked;
  $("retry-save").hidden = !blocked;
  $("retry-save").disabled = pendingCommit === null;
  for (const button of document.querySelectorAll("[data-time]"))
    button.disabled = blocked;
  for (const id of ["human", "size", "cols", "edge", "edit"])
    $(id).disabled = started || busy || blocked;
  if (analysis) {
    $("elapsed").textContent =
      `${(analysis.elapsedMs / 1000).toFixed(2)} 秒${analysis.overrunMs > 100 ? "（超预算）" : ""}`;
    $("windows").textContent = String(analysis.windowCount);
    $("search").textContent =
      `${analysis.search.depth || 0} 层 / ${analysis.search.nodes} 节点`;
    $("value").textContent = analysis.value.toFixed(3);
  } else
    for (const id of ["elapsed", "windows", "search", "value"])
      $(id).textContent = "—";
}
function color(alpha, red) {
  const base = [255, 244, 220],
    target = red ? [230, 35, 45] : [20, 180, 80];
  return `rgb(${base.map((v, i) => Math.round(v * (1 - alpha) + target[i] * alpha)).join(",")})`;
}
const worker = new Worker("./engine-worker.mjs", { type: "module" });
function think() {
  if (!started || !ready || busy || storageBlocked() || terminal()) return;
  busy = true;
  analysis = null;
  draw();
  say(
    turn() === human
      ? "正在计算本地推荐，请稍等…"
      : `AI 正在本机推算，预算 ${seconds} 秒…`,
  );
  worker.postMessage({
    type: "analyze",
    id: revision,
    n,
    cols,
    side: turn(),
    board: Array.from(board),
    seconds,
  });
}
function handleWorkerMessage({ data }) {
  if (data.type === "ready") {
    ready = true;
    draw();
    if (started) think();
    else say("模型已加载。设置棋盘后开始对局，或先点击空格设置禁下。");
    return;
  }
  if (data.type === "error") {
    busy = false;
    say(`本地计算失败：${data.error}`);
    draw();
    return;
  }
  if (data.type !== "result") return;
  if (storageBlocked()) {
    deferredResult = data;
    busy = false;
    draw();
    return;
  }
  busy = false;
  if (data.id !== revision) {
    draw();
    think();
    return;
  }
  if (data.terminal) {
    draw();
    return;
  }
  analysis = data;
  if (turn() !== human && !terminal()) {
    const p = data.search.move;
    if (!Number.isInteger(p) || board[p] !== 0) {
      say("AI 返回非法落点，已停止。");
      draw();
      return;
    }
    commitMove(p, "ai_move");
  } else {
    draw();
    say(`${data.search.reason}。点击空位落子。`);
  }
}
worker.onmessage = handleWorkerMessage;
worker.onerror = (event) => {
  busy = false;
  say(`无法启动本地计算：${event.message}`);
  draw();
};
function clickCell(p) {
  if (storageBlocked()) {
    say(storageError);
    return;
  }
  if (!started) {
    if (!$("edit").checked) {
      say("先点击“开始对局”，或开启禁下编辑。");
      return;
    }
    if (board[p] === 0 || board[p] === 3) {
      const nextBoard = board.slice();
      nextBoard[p] = board[p] === 3 ? 0 : 3;
      commitRecord(savedRecord({ board: Array.from(nextBoard) }), "forbidden_edit");
    }
    return;
  }
  if (board[p] === 3) {
    say("此处为灰色禁下格，不能落子。");
    return;
  }
  if (board[p] !== 0) {
    say("此处已有棋子。");
    return;
  }
  if (terminal()) {
    say("棋局已结束。");
    return;
  }
  if (busy || turn() !== human) {
    say("AI 正在本地推算，请稍等。");
    return;
  }
  commitMove(p, "human_move");
}
function setup(reset = false) {
  if (storageBlocked() || (started && !reset)) return;
  const size = Number($("size").value), nextCols = Number($("cols").value);
  if (![size, nextCols].every(value => Number.isInteger(value) && value >= 5 && value <= 32)) {
    say("棋盘横纵尺寸均须为 5 至 32 的整数。");
    $("size").value = n;
    $("cols").value = cols;
    return;
  }
  const nextBoard = new Uint8Array(size * nextCols),
    edge = $("edge").value;
  for (let p = 0; p < nextBoard.length; p++) {
    const r = Math.floor(p / nextCols),
      c = p % nextCols;
    if (
      ((edge === "top" || edge === "all") && r === 0) ||
      ((edge === "bottom" || edge === "all") && r === size - 1) ||
      ((edge === "left" || edge === "all") && c === 0) ||
      ((edge === "right" || edge === "all") && c === nextCols - 1)
    )
      nextBoard[p] = 3;
  }
  commitRecord(
    savedRecord({ n: size, cols: nextCols, edge, human: Number($("human").value),
      board: Array.from(nextBoard), history: [], started: false }),
    reset ? "new_game" : "setup",
    () => say("新棋局已准备好，可设置尺寸和禁下。"),
  );
}
$("size").onchange = () => setup();
$("cols").onchange = () => setup();
$("edge").onchange = () => setup();
$("human").onchange = () => {
  if (storageBlocked() || started || busy) {
    $("human").value = human;
    return;
  }
  commitRecord(savedRecord({ human: Number($("human").value) }), "human_setting");
};
$("seconds").onchange = () => {
  if (storageBlocked()) {
    $("seconds").value = seconds;
    return;
  }
  try {
    const nextSeconds = validateSeconds($("seconds").value);
    commitRecord(
      savedRecord({ seconds: nextSeconds }), "time_setting",
      () => say(`推算预算设为 ${seconds} 秒，从下一次计算生效。`),
      { bumpRevision: false, clearAnalysis: false },
    );
  } catch (e) {
    $("seconds").value = seconds;
    say(e.message);
  }
};
for (const b of document.querySelectorAll("[data-time]"))
  b.onclick = () => {
    if (storageBlocked()) return;
    $("seconds").value = b.dataset.time;
    $("seconds").onchange();
  };
$("overlay").onchange = draw;
$("zoom").onclick = () => {
  zoom = !zoom;
  $("zoom").textContent = zoom ? "适应屏幕" : "放大棋盘";
  draw();
};
$("start").onclick = () => {
  if (!ready || busy || started || storageBlocked() || terminal()) return;
  commitRecord(savedRecord({ started: true }), "start", () => {
    $("edit").checked = false;
    think();
  });
};
$("new").onclick = () => {
  if (storageBlocked()) return;
  setup(true);
};
$("undo").onclick = () => {
  if (busy || !history.length || storageBlocked()) return;
  const nextBoard = board.slice(),
    nextHistory = history.slice();
  do {
    const p = nextHistory.pop();
    nextBoard[p] = 0;
  } while (nextHistory.length && 1 + (nextHistory.length % 2) !== human);
  commitRecord(
    savedRecord({ board: Array.from(nextBoard), history: nextHistory }),
    "undo", () => think(),
  );
};
$("retry-save").onclick = () => retryStorage();
$("export").onclick = () => {
  const blob = new Blob(
    [
      JSON.stringify(
        {
          format: "must5-browser-game-v1",
          n,
          cols,
          board: Array.from(board),
          history: history.map((p, i) => ({
            row: Math.floor(p / cols),
            col: p % cols,
            side: 1 + (i % 2),
          })),
          human,
          seconds,
          winner: terminal(),
          storageError,
          pendingCommit: pendingSnapshot(),
        },
        null,
        2,
      ),
    ],
    { type: "application/json" },
  );
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `gomoku-${Date.now()}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};
// Read-only audit surface, also useful for local support and saved-game checks.
window.__gomokuSnapshot = () => ({
  n,
  cols,
  board: Array.from(board),
  history: history.slice(),
  human,
  seconds,
  started,
  ready,
  busy,
  revision,
  analysis,
  storageError,
  storageBlocked: storageBlocked(),
  pendingCommit: pendingSnapshot(),
});
async function offline() {
  if (!("serviceWorker" in navigator) || !isSecureContext) {
    $("offline").textContent = "本机计算 · 未启用离线";
    $("cache-note").textContent =
      "离线缓存需要 HTTPS 或 localhost。当前仍在浏览器内计算。";
    return;
  }
  try {
    await navigator.serviceWorker.register("./sw.js", { type: "module" });
    const reg = await navigator.serviceWorker.ready;
    const channel = new MessageChannel();
    const status = new Promise((resolve, reject) => {
      channel.port1.onmessage = (e) => resolve(e.data);
      setTimeout(() => reject(Error("缓存检查超时")), 15000);
    });
    (navigator.serviceWorker.controller || reg.active).postMessage(
      { type: "cache-status" },
      [channel.port2],
    );
    const result = await status;
    if (!result.ready) throw Error("离线资源未完整保存");
    $("offline").textContent = "本机计算 · 离线已就绪";
    $("cache-note").textContent =
      "全部模型和运行资源已保存，可以断网使用。棋局只保存在当前设备。";
  } catch (error) {
    $("offline").textContent = "本机计算 · 离线未就绪";
    $("cache-note").textContent =
      `离线准备未完成：${error.message}。保持联网后刷新重试。`;
  }
}
draw();
offline();
