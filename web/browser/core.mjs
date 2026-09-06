/** Shared browser rules, window fusion, and bounded tactical certificates. */
import { dimensions } from "./geometry.mjs";
export const PALETTE = [
  [255, 244, 220],
  [24, 27, 33],
  [255, 255, 255],
  [128, 128, 128],
];
export const RED = [230, 35, 45];
export function validate(board, n, side = 1) {
  const { rows, cols } = dimensions(n);
  if (
    !board ||
    board.length !== rows * cols ||
    ![1, 2].includes(side) ||
    Array.from(board).some((v) => !Number.isInteger(v) || v < 0 || v > 3)
  )
    throw Error("无效棋盘或行棋方");
}
const linesCache = new Map();
export function segments(n) {
  const { rows, cols } = dimensions(n), key = `${rows}x${cols}`;
  if (linesCache.has(key)) return linesCache.get(key);
  const lines = [];
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      for (const [dr, dc] of [
        [0, 1],
        [1, 0],
        [1, 1],
        [1, -1],
      ])
        if (r + 4 * dr < rows && c + 4 * dc >= 0 && c + 4 * dc < cols)
          lines.push(
            Array.from({ length: 5 }, (_, k) => (r + k * dr) * cols + c + k * dc),
          );
  linesCache.set(key, lines);
  return lines;
}
export function facts(board, n, side) {
  const wins = new Set(),
    blocks = new Set(),
    forcing = new Map();
  let winner = 0;
  for (const line of segments(n)) {
    let own = 0,
      enemy = 0,
      gray = 0;
    const empty = [];
    for (const p of line) {
      const v = board[p];
      if (v === side) own++;
      else if (v === 3 - side) enemy++;
      else if (v === 3) gray++;
      else empty.push(p);
    }
    if (own === 5) winner = side;
    if (enemy === 5) winner = 3 - side;
    if (gray) continue;
    if (own === 4 && empty.length === 1) wins.add(empty[0]);
    if (enemy === 4 && empty.length === 1) blocks.add(empty[0]);
    if (own === 3 && empty.length === 2)
      for (const p of empty) forcing.set(p, (forcing.get(p) || 0) + 1);
  }
  return {
    winner,
    wins: [...wins].sort((a, b) => a - b),
    blocks: [...blocks].sort((a, b) => a - b),
    forcing: [...forcing.keys()].sort(
      (a, b) => forcing.get(b) - forcing.get(a) || a - b,
    ),
  };
}
export function legal(board) {
  return Array.from(board, (_, i) => i).filter((i) => board[i] === 0);
}
export function ordered(board, n, side, priors = null) {
  const { rows, cols } = dimensions(n);
  const score = new Float64Array(board.length),
    weight = [0, 1, 8, 64, 1024, 1e6];
  for (const line of segments(n)) {
    let a = 0,
      b = 0,
      gray = false;
    for (const p of line) {
      a += board[p] === side;
      b += board[p] === 3 - side;
      gray ||= board[p] === 3;
    }
    if (gray) continue;
    const s =
      (b === 0 ? 10 * (weight[Math.min(a + 1, 5)] - weight[a]) : 0) +
      (a === 0 ? 11 * (weight[Math.min(b + 1, 5)] - weight[b]) : 0);
    for (const p of line) score[p] += s;
  }
  const max = priors ? Math.max(...priors) : 0,
    centerRow = (rows - 1) / 2,
    centerCol = (cols - 1) / 2;
  if (max > 0)
    for (let i = 0; i < score.length; i++) score[i] += (240 * priors[i]) / max;
  const distance = (p) =>
    (Math.floor(p / cols) - centerRow) ** 2 + ((p % cols) - centerCol) ** 2;
  return legal(board).sort(
    (a, b) => score[b] - score[a] || distance(a) - distance(b) || a - b,
  );
}
export function centers(board, n) {
  const { rows, cols } = dimensions(n);
  const stones = Array.from(board, (_, i) => i).filter(
    (p) => board[p] === 1 || board[p] === 2,
  );
  const result = stones.length
    ? [...stones]
    : [Math.floor(rows / 2) * cols + Math.floor(cols / 2)];
  const unseen = new Set(legal(board));
  const covered = (p) => {
    const r = Math.floor(p / cols),
      c = p % cols,
      points = [];
    for (let dr = -2; dr <= 2; dr++)
      for (let dc = -2; dc <= 2; dc++)
        if (r + dr >= 0 && r + dr < rows && c + dc >= 0 && c + dc < cols)
          points.push((r + dr) * cols + c + dc);
    return points;
  };
  for (const p of result) for (const q of covered(p)) unseen.delete(q);
  const empties = legal(board);
  while (unseen.size) {
    let best = -1,
      gain = 0;
    for (const p of empties) {
      const count = covered(p).filter((q) => unseen.has(q)).length;
      if (count > gain) {
        best = p;
        gain = count;
      }
    }
    if (best < 0) throw Error("棋盘覆盖不完整");
    result.push(best);
    for (const q of covered(best)) unseen.delete(q);
  }
  return result;
}
export function windowAt(board, n, p) {
  const { rows, cols } = dimensions(n);
  const row = Math.floor(p / cols),
    col = p % cols,
    cells = new Uint8Array(25),
    indices = new Int32Array(25).fill(-1);
  for (let i = 0; i < 25; i++) {
    const r = row + Math.floor(i / 5) - 2,
      c = col + (i % 5) - 2;
    cells[i] = 3;
    if (r >= 0 && r < rows && c >= 0 && c < cols) {
      indices[i] = r * cols + c;
      cells[i] = board[r * cols + c];
    }
  }
  return { cells, indices };
}
export function encodeRGB(cells, red = null) {
  const x = new Float32Array(cells.length * 3);
  for (let i = 0; i < cells.length; i++)
    for (let ch = 0; ch < 3; ch++) {
      const alpha = cells[i] === 0 && red ? red[i] : 0;
      x[ch * cells.length + i] =
        (PALETTE[cells[i]][ch] / 255) * (1 - alpha) + (RED[ch] / 255) * alpha;
    }
  return x;
}
export function maskedSoftmax(logits, board) {
  const result = new Float64Array(board.length),
    empty = legal(board);
  if (!empty.length) return result;
  let max = -Infinity;
  for (const p of empty) {
    if (!Number.isFinite(logits[p])) throw Error("模型输出非有限数值");
    max = Math.max(max, logits[p]);
  }
  let sum = 0;
  for (const p of empty) {
    result[p] = Math.exp(logits[p] - max);
    sum += result[p];
  }
  for (const p of empty) result[p] /= sum;
  return result;
}
export function normalizeFusion(sum, coverage, board) {
  const out = new Float64Array(board.length);
  let mass = 0;
  for (const p of legal(board)) {
    if (!coverage[p]) throw Error("存在未覆盖空格");
    out[p] = sum[p] / coverage[p];
    mass += out[p];
  }
  if (!(mass > 0)) throw Error("无效概率总量");
  for (let i = 0; i < out.length; i++) out[i] /= mass;
  return out;
}
export function globalInput(board, n, side, opponent, play) {
  const { rows, cols } = dimensions(n);
  const size = board.length,
    x = new Float32Array(size * 9);
  for (let i = 0; i < size; i++) {
    const v = side === 2 && [1, 2].includes(board[i]) ? 3 - board[i] : board[i];
    for (let ch = 0; ch < 3; ch++) x[ch * size + i] = PALETTE[v][ch] / 255;
    x[3 * size + i] = opponent[i];
    x[4 * size + i] = play[i];
    x[5 * size + i] = board[i] === 0;
    x[6 * size + i] = -1;
    x[7 * size + i] = (2 * Math.floor(i / cols)) / (rows - 1) - 1;
    x[8 * size + i] = (2 * (i % cols)) / (cols - 1) - 1;
  }
  return x;
}
const BUDGET = Symbol("budget");
const MAX_POSITIVE_PROOFS = 4096;
// Exact bytes include forbidden cells. Shape/actor/horizon prevent collisions
// between rectangular boards or a proof obtained with a larger allowance.
function proofKey(board, n, side, ...allowances) {
  return `${n.rows}x${n.cols}:${side}:${allowances.join(":")}:${board.join("")}`;
}
function retainPositive(cache, key, proof) {
  if (cache && proof?.value === 1 && cache.size < MAX_POSITIVE_PROOFS)
    // Preserve every proof field (including optional auditId), with no mutable
    // references to the search's working result or its representative line.
    cache.set(key, structuredClone(proof));
}
export function forcing(
  board,
  n,
  side,
  { milliseconds = 50, maxNodes = 5000, depth = 32, cacheProofs = true } = {},
) {
  if (typeof cacheProofs !== "boolean") throw Error("cacheProofs must be boolean");
  return forcingWithCache(board, n, side, { milliseconds, maxNodes, depth },
    cacheProofs ? new Map() : null);
}
// Shared only by forcing leaves belonging to one quietThreat call. Keeping
// this entry private prevents callers from supplying unverified cached proofs.
function forcingWithCache(
  board, n, side, { milliseconds, maxNodes, depth }, positiveCache,
) {
  n = dimensions(n);
  const start = performance.now(),
    deadline = start + Math.max(0, milliseconds);
  let nodes = 0, cacheHits = 0;
  const check = () => {
    if (nodes >= maxNodes || performance.now() >= deadline) throw BUDGET;
  };
  function visit(actor, left) {
    check();
    nodes++; // A hit is an actual visited node and obeys the current leaf cap.
    const key = positiveCache ? proofKey(board, n, actor, left) : null;
    const cached = positiveCache?.get(key);
    if (cached) {
      cacheHits++;
      return structuredClone(cached);
    }
    const proof = explore(actor, left);
    retainPositive(positiveCache, key, proof);
    return proof;
  }
  function explore(actor, left) {
    const f = facts(board, n, actor);
    if (f.winner) return { value: f.winner === actor ? 1 : -1, line: [] };
    if (f.wins.length)
      return { value: 1, line: [{ side: actor, move: f.wins[0] }] };
    if (f.blocks.length > 1)
      return {
        value: -1,
        line: [
          { side: actor, move: f.blocks[0] },
          { side: 3 - actor, move: f.blocks[1] },
        ],
      };
    const empty = legal(board);
    if (!empty.length) return { value: 0, line: [] };
    if (left <= 0) return null;
    const moves = f.blocks.length ? f.blocks : f.forcing,
      complete = f.blocks.length > 0 || moves.length === empty.length;
    let best = null,
      known = true;
    for (const move of moves) {
      check();
      board[move] = actor;
      let child;
      try {
        child = visit(3 - actor, left - 1);
      } finally {
        board[move] = 0;
      }
      if (!child) {
        known = false;
        continue;
      }
      const proof = {
        value: -child.value,
        line: [{ side: actor, move }, ...child.line],
      };
      if (proof.value === 1) return proof;
      if (
        !best ||
        proof.value > best.value ||
        (proof.value === best.value && proof.line.length > best.line.length)
      )
        best = proof;
    }
    return complete && known ? best : null;
  }
  let proof = null,
    exhausted = false;
  try {
    proof = visit(side, depth);
  } catch (e) {
    if (e !== BUDGET) throw e;
    exhausted = true;
  }
  return {
    value: proof?.value ?? null,
    line: proof?.line ?? [],
    move: proof?.line[0]?.move ?? null,
    nodes,
    elapsed: performance.now() - start,
    exhausted,
    cacheHits,
    cacheHitNodes: cacheHits,
    cacheEntries: positiveCache?.size ?? 0,
  };
}
export function quietThreat(
  board,
  n,
  side,
  {
    milliseconds = 200,
    maxNodes = 20000,
    width = 16,
    quiet = 2,
    total = 64,
    cacheProofs = true,
  } = {},
) {
  n = dimensions(n);
  if (typeof cacheProofs !== "boolean") throw Error("cacheProofs must be boolean");
  // Only completed positive facts survive ordering/leaf-budget iterations.
  // These maps never escape this request or contain unknown/partial attempts.
  const positiveCache = cacheProofs ? new Map() : null,
    forcingCache = cacheProofs ? new Map() : null;
  let recursiveCacheHits = 0, forcingCacheHits = 0;
  const start = performance.now(),
    deadline = start + Math.max(0, milliseconds);
  let nodes = 0,
    leafMs = 5,
    leafNodes = 500,
    activeDeadline = deadline,
    activeMaxNodes = maxNodes,
    ordering = "quiet_first";
  const check = () => {
    if (nodes >= activeMaxNodes || performance.now() >= activeDeadline) throw BUDGET;
  };
  function visit(left, ply) {
    check();
    nodes++; // Hits obey both phase and whole-call node/time allowances.
    // Capture the parent board before explore temporarily applies an attack.
    const key = positiveCache ? proofKey(board, n, side, left, total - ply) : null;
    const cached = positiveCache?.get(key);
    if (cached) {
      recursiveCacheHits++;
      return structuredClone(cached);
    }
    const proof = explore(left, ply);
    // explore returns a setup proof only after its entire defender loop passes.
    // A budget exception unwinds without storing the unfinished parent.
    retainPositive(positiveCache, key, proof);
    return proof;
  }
  function explore(left, ply) {
    const f = facts(board, n, side);
    if (f.winner)
      return f.winner === side ? { value: 1, line: [], move: null } : null;
    if (ply >= total) return null;
    if (f.wins.length)
      return { value: 1, move: f.wins[0], line: [{ side, move: f.wins[0] }] };
    if (
      f.blocks.length > 1 ||
      !legal(board).length ||
      (!f.blocks.length && left <= 0) ||
      ply + 2 >= total
    )
      return null;
    let count = 0;
    const mandatory = f.blocks.length === 1, forcingMoves = new Set(f.forcing);
    const ranked = mandatory ? f.blocks : ordered(board, n, side),
      candidates = mandatory || ordering === "natural" ? ranked : [
        ...ranked.filter(move => !forcingMoves.has(move)),
        ...ranked.filter(move => forcingMoves.has(move)),
      ];
    for (const move of candidates) {
      // Each ordering pass bounds all attacker candidates, including active fours.
      // Defender replies below always retain every legal point.
      if (!mandatory && count++ >= width) break;
      check();
      nodes++;
      board[move] = side;
      try {
        const cf = facts(board, n, side);
        const quietCost = mandatory || cf.wins.length ? 0 : 1,
          nextQuiet = left - quietCost;
        if (cf.blocks.length) continue;
        const replies = ordered(board, n, 3 - side);
        if (!replies.length) continue;
        let complete = true,
          longest = [],
          used = 0;
        for (const reply of replies) {
          check();
          nodes++;
          board[reply] = 3 - side;
          try {
            const cap = total - ply - 2;
            let proof;
            const immediate = cf.wins.find(point => point !== reply);
            // cf.blocks was empty before this reply, so it cannot win now.
            // A five-cell line containing four of our stones has only one
            // empty point: a reply elsewhere cannot interrupt that same five.
            // Still visit and count every defender reply and proof leaf.
            if (cap >= 1 && immediate !== undefined && board[immediate] === 0) {
              check();
              nodes++;
              proof = { value: 1, line: [{ side, move: immediate }] };
            } else {
              proof = forcingWithCache(board, n, side, {
                milliseconds: Math.min(leafMs, activeDeadline - performance.now()),
                maxNodes: Math.min(leafNodes, activeMaxNodes - nodes),
                depth: Math.min(32, Math.max(0, cap - 2)),
              }, forcingCache);
              nodes += proof.nodes;
              forcingCacheHits += proof.cacheHits;
            }
            // With zero quiet allowance, visit still permits mandatory defense.
            // This preserves counter-four replies after the last quiet setup.
            if (proof.value === null && (nextQuiet > 0 ||
                (nextQuiet === 0 && facts(board, n, side).blocks.length === 1)))
              proof = visit(nextQuiet, ply + 2);
            if (proof?.value !== 1 || proof.line.length > cap) {
              complete = false;
              break;
            }
            used++;
            if (proof.line.length + 1 > longest.length)
              longest = [{ side: 3 - side, move: reply }, ...proof.line];
          } finally {
            board[reply] = 0;
          }
        }
        if (complete && used === replies.length)
          return {
            value: 1,
            move,
            line: [{ side, move }, ...longest],
            certifiedReplies: used,
          };
      } finally {
        board[move] = 0;
      }
    }
    return null;
  }
  let proof = null;
  const iterations = [];
  // Try quiet setups first, then the natural tactical ordering. Both passes
  // share the original deadline and node allowance; unknown excludes nothing.
  // Their candidate union can exceed width, but each pass is bounded by width.
  for (const order of ["quiet_first", "natural"]) {
    ordering = order;
    activeDeadline = order === "quiet_first"
      ? Math.min(deadline, start + Math.max(0, milliseconds) * 0.45) : deadline;
    activeMaxNodes = order === "quiet_first"
      ? Math.min(maxNodes, 3500, Math.floor(maxNodes * 0.7)) : maxNodes;
    try {
      for (const ms of [5, 50]) {
        leafMs = ms;
        leafNodes = ms === 5 ? 500 : 5000;
        for (let q = quiet === 0 ? 0 : 1; q <= quiet; q++) {
          const before = nodes;
          let completed = false;
          try {
            proof = visit(q, 0);
            completed = true;
          } finally {
            iterations.push({ ordering, quiet: q, leafMs: ms, nodes: nodes - before, completed });
          }
          if (proof) break;
        }
        if (proof) break;
      }
    } catch (e) {
      if (e !== BUDGET) throw e;
    }
    if (proof || nodes >= maxNodes || performance.now() >= deadline) break;
  }
  const elapsed = performance.now() - start;
  return {
    value: proof?.value ?? null,
    move: proof?.move ?? null,
    line: proof?.line ?? [],
    certifiedReplies: proof?.certifiedReplies ?? 0,
    nodes,
    elapsed,
    exhausted: nodes >= maxNodes || elapsed >= Math.max(0, milliseconds),
    iterations,
    cacheHits: recursiveCacheHits + forcingCacheHits,
    cacheHitNodes: recursiveCacheHits + forcingCacheHits,
    recursiveCacheHits,
    forcingCacheHits,
    recursiveCacheEntries: positiveCache?.size ?? 0,
    forcingCacheEntries: forcingCache?.size ?? 0,
  };
}
export function validateSeconds(value) {
  if (typeof value !== "number" && typeof value !== "string")
    throw Error("推算时间须为数字");
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds < 0.1 || seconds > 30)
    throw Error("推算时间须为 0.1 至 30 秒");
  return seconds;
}
