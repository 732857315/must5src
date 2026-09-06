"""Local browser game with server-owned state and checked JSON requests."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import hashlib
import os
import uuid
import json
import logging
import math
from numbers import Integral
from pathlib import Path
import threading
from urllib.parse import urlsplit

import numpy as np

from board_rules import normalize_board, board_winner, apply_board_move

BOARD_DEFAULT = 16
BODY_LIMIT = 262144


class APIError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise APIError(400, f"{name}必须是 {low} 到 {high} 的整数")
    return value


def strict_json(body):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("JSON不能包含重复字段")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("请求配置必须使用有限整数")

    try:
        result = json.loads(body.decode("utf-8"), object_pairs_hook=pairs,
                            parse_constant=reject, parse_float=reject)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise APIError(400, "请求必须是合法 UTF-8 JSON；不允许重复字段或非有限数字") from exc
    if not isinstance(result, dict):
        raise APIError(400, "JSON顶层必须是对象")
    return result


def build_analyzer(global_model=None):
    """Compose the local windows and optional independently trained global net."""
    from unet_board import analyze_board
    if global_model is None:
        return analyze_board
    from global_inference import analyze_global

    def analyze(board, side, opponent, play, *, tactical=True):
        result = analyze_board(board, side, opponent, play, tactical=tactical)
        result.update(analyze_global(board, side, result, global_model))
        if not result["terminal"] and not result["tactical_applied"]:
            result["play_policy"] = result["combined_policy"].copy()
            result["move"] = tuple(map(int, np.unravel_index(
                np.argmax(result["combined_policy"]), result["combined_policy"].shape)))
            result["reason"] = "大局与局部策略融合"
        return result
    return analyze


def describe_ai_move(board, side, move, selection, *, has_global=False, threats=None):
    """Explain observed changes once per chosen move, never as a motif proof."""
    base = str(selection.get("reason", ""))
    if move is None:
        return base
    before = normalize_board(board)
    after = apply_board_move(before, move, side)
    if selection.get("proven_value") is not None or board_winner(after) == side:
        return base or "这手已形成五连"
    from board_threats import analyze_threats
    reports = threats or {}
    own_before = reports.get("self") or analyze_threats(before, side)
    enemy_before = reports.get("opponent") or analyze_threats(before, 3 - side)
    point = tuple(map(int, move))
    winning = [tuple(cell) for cell in enemy_before.get("winning_cells", [])]
    if len(winning) == 1 and point == winning[0]:
        return base or "封堵对手唯一立即成五点"
    own_after = analyze_threats(after, side)
    enemy_after = analyze_threats(after, 3 - side)
    explanations = []
    shared = [tuple(cell) for cell in enemy_before.get("shared_completion_cells", [])]
    three_defenses = [tuple(cell) for cell in enemy_before.get("three_defense_cells", [])]
    def path_count(report):
        return sum(len(core.get("upgrade_paths", [])) for core in report.get("threes", []))
    if point in shared:
        explanations.append("占住对手多条四形的共享成五点")
    elif point in three_defenses and enemy_before.get("three_count", 0):
        explanations.append("占住多条活三的共同防点" if enemy_before["three_count"] > 1 else "封住对手活三的升四路径")
    elif path_count(enemy_after) < path_count(enemy_before):
        explanations.append("压制对手活三的升四路径")
    if own_after.get("open_four") and not own_before.get("open_four"):
        explanations.append("形成两端可成五的活四")
    elif own_after.get("four_count", 0) > own_before.get("four_count", 0):
        explanations.append("形成新的四形压力")
    elif own_after.get("three_count", 0) > own_before.get("three_count", 0):
        explanations.append("增加活三，保留后续升四空间")
    if explanations:
        return "；".join(explanations[:2])
    return "根据全盘搜索与大局估计选择此点" if has_global else "根据全盘搜索选择此点"


def build_move_selector(global_model=None, *, search_seconds=1.0, search_nodes=2000,
                        search_depth=5, search_width=12, value_weight=40.0,
                        engine="python", forcing_seconds=None, forcing_nodes=None, forcing_depth=None,
                        guard_native_fraction=.45, guard_probe_nodes=5000, guard_probe_seconds=.05,
                        guard_threat_seconds=.2, guard_threat_nodes=20000, guard_threat_width=16,
                        guard_threat_quiet_plies=2, guard_threat_total_plies=64,
                        guard_attack_seconds=.15, guard_attack_nodes=20000):
    """Bind search budgets and the optional cheap, cached neural leaf evaluator."""
    if engine not in ("python", "native", "guarded"):
        raise ValueError("engine must be python, native, or guarded")
    if engine == "guarded" and (forcing_seconds is not None or forcing_nodes is not None):
        raise ValueError("guarded uses shared budgets and guard_probe settings, not forcing_seconds/nodes")
    forcing_depth = (32 if engine == "guarded" else 24) if forcing_depth is None else forcing_depth
    forcing_seconds = .1 if forcing_seconds is None else forcing_seconds
    forcing_nodes = 10000 if forcing_nodes is None else forcing_nodes
    from board_search import select_move
    if engine == "guarded":
        from search_guard import select_guarded_move
    if engine == "native":
        from native_search import select_native_move
    if global_model is not None and engine == "python":
        from global_inference import evaluate_global_value

    def select(board, side, analysis):
        priors = analysis.get("combined_policy", analysis["raw_play_policy"])
        options = {}
        if global_model is not None and engine == "python":
            options.update(value_evaluator=lambda position, actor: evaluate_global_value(position, actor, global_model),
                           value_weight=value_weight)
        if engine == "guarded":
            result = select_guarded_move(board, side, np.asarray(priors, dtype=np.float64),
                                        max_nodes=search_nodes, time_limit=search_seconds,
                                        depth=search_depth, candidate_width=search_width,
                                        native_fraction=guard_native_fraction,
                                        probe_max_nodes=guard_probe_nodes, probe_time_limit=guard_probe_seconds,
                                        forcing_depth=forcing_depth, threat_time_limit=guard_threat_seconds,
                                        threat_max_nodes=guard_threat_nodes, threat_width=guard_threat_width,
                                        threat_quiet_plies=guard_threat_quiet_plies,
                                        threat_total_plies=guard_threat_total_plies,
                                        attack_time_limit=guard_attack_seconds, attack_max_nodes=guard_attack_nodes)
        elif engine == "native":
            result = select_native_move(board, side, np.asarray(priors, dtype=np.float64),
                                        max_nodes=search_nodes, time_limit=search_seconds,
                                        depth=search_depth, candidate_width=search_width,
                                        forcing_seconds=forcing_seconds, forcing_nodes=forcing_nodes,
                                        forcing_depth=forcing_depth)
        else:
            result = select_move(board, side, np.asarray(priors, dtype=np.float64),
                                 max_nodes=search_nodes, time_limit=search_seconds,
                                 depth=search_depth, candidate_width=search_width, **options)

        explanation = describe_ai_move(board, side, result["move"], result,
                                       has_global=global_model is not None, threats=analysis.get("threats"))
        if engine == "guarded" and result.get("proven_value") is None:
            result["reason"] += ("；"+explanation) if explanation != result["reason"] else ""
        else:
            result["reason"] = explanation
        if engine in ("native", "guarded"):
            result["reason"] = result["reason"].replace("大局估计", "大局策略")
        return result
    return select


class GameSession:
    """One local game; model calls and mutations are serialized by a lock.

    analyzer follows unet_board.analyze_board. An optional move_selector receives
    (board_copy, side, analysis) and returns a dict with move/reason.
    """

    def __init__(self, opponent_model, play_model, *, analyzer=None, move_selector=None,
                 max_size=64, model_source="", trained=False, auto_start=True,
                 strategy_source="", strategy_trained=False, model_status=None,
                 state_file=None, persistence_identity=None, engine="python", search_configuration=None):
        if analyzer is None:
            from unet_board import analyze_board
            analyzer = analyze_board
        self.opponent_model = opponent_model
        self.play_model = play_model
        self.analyzer = analyzer
        self.move_selector = move_selector
        self.max_size = integer(max_size, "最大棋盘尺寸", 6, 128)
        if engine not in ("python", "native", "guarded"):
            raise ValueError("unsupported search engine")
        if engine in ("native", "guarded") and self.max_size > 64:
            raise ValueError("native search supports maximum board size 64")
        self.engine = engine
        self.search_configuration = json.loads(json.dumps(search_configuration or {}, allow_nan=False))
        self.model_source = str(model_source)
        self.trained = bool(trained)
        self.model_status = model_status or {
            "opponent": {"loaded": True, "trained": self.trained, "source": self.model_source},
            "play": {"loaded": True, "trained": self.trained, "source": self.model_source},
            "global": {"loaded": bool(strategy_source), "trained": bool(strategy_trained),
                       "source": str(strategy_source)},
        }
        self.lock = threading.RLock()
        self.revision = 0
        self.active = False
        self.board = np.zeros((min(BOARD_DEFAULT, max_size),) * 2, dtype=np.uint8)
        self.human_side, self.ai_side = 2, 1
        self.history = []
        self.analysis = None
        self.last_ai_reason = ""
        self.last_ai_search = None
        self.pending_move = None
        self.state_file = Path(state_file).resolve() if state_file else None
        self.persistence_identity = json.loads(json.dumps(persistence_identity or {}, allow_nan=False))
        self._state_lock = None
        if self.state_file:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_lock = self.state_file.with_suffix(self.state_file.suffix + ".lock").open("a+b")
            self._state_lock.seek(0, 2)
            if self._state_lock.tell() == 0:
                self._state_lock.write(b"0")
                self._state_lock.flush()
            self._state_lock.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self._state_lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._state_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if self.state_file.exists():
                    self._restore_state()
                    return
            except Exception:
                self.close()
                raise
        if auto_start:
            try:
                self.new_game({})
            except BaseException:
                self.close()
                raise

    def close(self):
        if self._state_lock is not None:
            self._state_lock.close()
            self._state_lock = None

    def _state_record(self, *, board=None, history=None, human_side=None, ai_side=None,
                      revision=None, ai_reason=None, ai_search=None, pending=None):
        return {"format": "gomoku_local_game_v1", "identity": self.persistence_identity,
                "board": (self.board if board is None else board).tolist(),
                "history": self.history if history is None else history,
                "human_side": self.human_side if human_side is None else human_side,
                "ai_side": self.ai_side if ai_side is None else ai_side,
                "revision": self.revision if revision is None else revision,
                "last_ai_reason": self.last_ai_reason if ai_reason is None else ai_reason,
                "last_ai_search": ai_search, "pending": pending,
                "pending_status": "pending_move" if pending is not None else "committed"}

    def _write_state(self, record):
        if not self.state_file:
            return
        temporary = self.state_file.with_name(self.state_file.name + ".tmp-" + uuid.uuid4().hex)
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            # Windows scanners/readers can briefly deny the atomic rename.
            # Retry only that rename: the same serialized file is already fsynced.
            import time
            deadline = time.monotonic() + 0.5
            delays = (0.025, 0.05, 0.075, 0.1, 0.15)
            for attempt in range(len(delays) + 1):
                try:
                    os.replace(temporary, self.state_file)
                    break
                except OSError as exc:
                    if (getattr(exc, "winerror", None) not in (5, 32, 33)
                            or attempt == len(delays)
                            or time.monotonic() + delays[attempt] >= deadline):
                        raise
                    time.sleep(delays[attempt])
                    if time.monotonic() >= deadline:
                        raise
        finally:
            if temporary.exists():
                temporary.unlink()

    def _commit(self, board, history, human_side, ai_side, analysis, ai_reason, ai_search):
        revision = self.revision + 1
        record = self._state_record(board=board, history=history, human_side=human_side, ai_side=ai_side,
                                    revision=revision, ai_reason=ai_reason, ai_search=ai_search)
        self._write_state(record)
        self.board, self.history, self.human_side, self.ai_side = board, history, human_side, ai_side
        self.analysis, self.last_ai_reason, self.last_ai_search = analysis, ai_reason, ai_search
        self.revision, self.pending_move, self.active = revision, None, True
        return self._snapshot()

    def _restore_state(self):
        record = json.loads(self.state_file.read_text(encoding="utf-8"))
        json.dumps(record, allow_nan=False)
        if record.get("format") != "gomoku_local_game_v1" or record.get("identity") != self.persistence_identity:
            raise ValueError("持久化棋局的模型、搜索配置、源码或 run_id 不匹配，不能重新抽局")
        board = self._checked_board(record["board"])
        human = integer(record["human_side"], "存储人方", 1, 2)
        ai = integer(record["ai_side"], "存储AI方", 1, 2)
        if human == ai:
            raise ValueError("存储执子身份冲突")
        history = record["history"]
        if not isinstance(history, list):
            raise ValueError("存储棋谱无效")
        rebuilt = np.where(board == 3, 3, 0).astype(np.uint8)
        expected = 1
        for entry in history:
            if not isinstance(entry, dict) or integer(entry.get("side"), "存储行棋方", 1, 2) != expected:
                raise ValueError("存储棋谱行棋顺序无效")
            row = integer(entry.get("row"), "存储落子行", 0, board.shape[0]-1)
            col = integer(entry.get("col"), "存储落子列", 0, board.shape[1]-1)
            if entry.get("by") != ("ai" if expected == ai else "human"):
                raise ValueError("存储棋谱身份无效")
            rebuilt = apply_board_move(rebuilt, (row, col), expected)
            expected = 3 - expected
        if not np.array_equal(rebuilt, board):
            raise ValueError("存储棋盘与完整棋谱不一致")
        if not self._finished(board) and expected != human:
            raise ValueError("持久化棋局不是完整人机回合")
        if ai == 1 and (not history or (history[0]["row"], history[0]["col"]) !=
                        (board.shape[0] // 2, board.shape[1] // 2)):
            raise ValueError("AI黑棋存储开局不是中心")
        revision = integer(record["revision"], "存储棋局版本", 1, 2**53-1)
        pending = record.get("pending")
        if pending is not None:
            if not isinstance(pending, dict) or set(pending) != {"row", "col", "revision"} or integer(pending["revision"], "待提交版本", 1, 2**53-1) != revision:
                raise ValueError("待提交落子记录无效")
            row = integer(pending["row"], "待提交行", 0, board.shape[0]-1)
            col = integer(pending["col"], "待提交列", 0, board.shape[1]-1)
            apply_board_move(board, (row, col), human)
        if record.get("pending_status") != ("pending_move" if pending is not None else "committed"):
            raise ValueError("持久化提交状态无效")
        analysis = self._analyze(board, human)
        self.board, self.history, self.human_side, self.ai_side = board, history, human, ai
        self.revision, self.analysis, self.active = revision, analysis, True
        self.last_ai_reason = str(record.get("last_ai_reason", ""))
        self.last_ai_search = record.get("last_ai_search")
        self.pending_move = pending

    def _checked_board(self, board):
        board = normalize_board(board)
        if any(not 6 <= size <= self.max_size for size in board.shape):
            raise APIError(400, f"棋盘行列数必须为 6 到 {self.max_size}")
        return board

    @staticmethod
    def _finished(board):
        return board_winner(board) != 0 or not np.any(board == 0)

    def _analyze(self, board, side):
        try:
            result = self.analyzer(board.copy(), side, self.opponent_model,
                                   self.play_model, tactical=True)
            output = {"side": side, "move": None, "reason": str(result.get("reason", "")),
                      "window_count": int(result.get("window_count", 0))}
            legal = board == 0
            policy_names = ["opponent_policy", "play_policy", "raw_play_policy"]
            if "global_policy" in result or "combined_policy" in result:
                policy_names += ["global_policy", "combined_policy"]
            for name in policy_names:
                values = np.asarray(result[name], dtype=np.float64)
                if (values.shape != board.shape or not np.isfinite(values).all()
                        or np.any(values < -1e-10) or np.any(values > 1 + 1e-10)
                        or np.any(values[~legal] != 0)):
                    raise ValueError(f"invalid {name}")
                output[name] = np.clip(values, 0, 1).tolist()
            coverage = np.asarray(result.get("coverage", np.zeros_like(board)))
            if coverage.shape != board.shape or not np.isfinite(coverage).all() or np.any(coverage < 0):
                raise ValueError("invalid coverage")
            output["coverage"] = coverage.astype(int).tolist()
            move = result.get("move")
            if move is not None:
                if (len(move) != 2 or any(isinstance(value, bool) or not isinstance(value, Integral) for value in move)):
                    raise ValueError("analysis move must contain two integers")
                row, column = int(move[0]), int(move[1])
                apply_board_move(board, (row, column), side)
                output["move"] = [row, column]
            if self._finished(board):
                if move is not None or any(np.any(output[name]) for name in policy_names):
                    raise ValueError("terminal board has an action")
            elif move is None:
                raise ValueError("live board has no selected move")
            if "global_policy" in result:
                value = result.get("value")
                if value is not None:
                    if isinstance(value, bool) or not math.isfinite(float(value)) or not -1 <= float(value) <= 1:
                        raise ValueError("invalid global value estimate")
                    value = float(value)
                output.update(value=value, value_source=str(result.get("value_source", "network_estimate")),
                              global_weight=float(result.get("global_weight", 0.7)))
            output["threats"] = json.loads(json.dumps(result.get("threats", {}), allow_nan=False))
            return output
        except APIError:
            raise
        except Exception as exc:
            logging.exception("Board analysis failed")
            raise APIError(503, "模型计算失败，本次操作尚未保存；请查看服务终端") from exc

    def _check_revision(self, payload, *, required):
        if "revision" not in payload:
            if required:
                raise APIError(400, "缺少棋局版本，请刷新页面")
            return
        revision = integer(payload["revision"], "棋局版本", 0, 2**53 - 1)
        if revision != self.revision:
            raise APIError(409, "棋局已变化，请刷新后重试")

    def new_game(self, payload):
        if set(payload) - {"rows", "cols", "human_first", "forbidden", "revision"}:
            raise APIError(400, "新局请求含未支持字段")
        with self.lock:
            if self.pending_move is not None:
                raise APIError(409, "有待提交落子，必须恢复同一手，不能换局绕过")
            self._check_revision(payload, required=False)
            default = min(BOARD_DEFAULT, self.max_size)
            rows = integer(payload.get("rows", default), "行数", 6, self.max_size)
            cols = integer(payload.get("cols", default), "列数", 6, self.max_size)
            human_first = payload.get("human_first", False)
            if not isinstance(human_first, bool):
                raise APIError(400, "human_first必须是布尔值")
            forbidden = payload.get("forbidden", [])
            if not isinstance(forbidden, list) or len(forbidden) > rows * cols:
                raise APIError(400, "禁下格必须是棋盘范围内的坐标数组")
            board = np.zeros((rows, cols), dtype=np.uint8)
            seen = set()
            for pair in forbidden:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise APIError(400, "每个禁下格必须是[row,col]")
                point = (integer(pair[0], "禁下行", 0, rows - 1),
                         integer(pair[1], "禁下列", 0, cols - 1))
                if point in seen:
                    raise APIError(400, "禁下格坐标不能重复")
                seen.add(point)
                board[point] = 3
            if not np.any(board == 0):
                raise APIError(400, "棋盘至少要留一个可落空格")
            board = self._checked_board(board)
            human_side, ai_side = (1, 2) if human_first else (2, 1)
            history, ai_reason = [], ""
            if not human_first:
                center = rows // 2, cols // 2
                if board[center] != 0:
                    raise APIError(400, f"AI先手需要中心{center}可落，请取消该禁下格或选择人先手")
                board = apply_board_move(board, center, ai_side)
                history.append({"side": ai_side, "row": center[0], "col": center[1], "by": "ai"})
                ai_reason = "黑棋按设定先手落在中心"
            analysis = self._analyze(board, human_side)
            return self._commit(board, history, human_side, ai_side, analysis, ai_reason, None)

    def move(self, payload):
        if set(payload) != {"row", "col", "revision"}:
            raise APIError(400, "落子请求仅接受row、col和revision；棋盘由服务器保存")
        with self.lock:
            self._check_revision(payload, required=True)
            if self.pending_move is not None and payload != self.pending_move:
                raise APIError(409, "必须重试持久化记录中的同一手，不能替换落点")
            if not self.active or self._finished(self.board):
                raise APIError(409, "本局已结束，请开始新局")
            if self.analysis is None or self.analysis["side"] != self.human_side:
                raise APIError(409, "当前不是人的回合")
            row = integer(payload["row"], "落子行", 0, self.board.shape[0] - 1)
            column = integer(payload["col"], "落子列", 0, self.board.shape[1] - 1)
            try:
                working = apply_board_move(self.board, (row, column), self.human_side)
            except ValueError as exc:
                raise APIError(400, str(exc)) from exc
            if self.state_file and self.pending_move is None:
                self._write_state(self._state_record(ai_search=self.last_ai_search, pending=dict(payload)))
                self.pending_move = dict(payload)
            history = self.history + [{"side": self.human_side, "row": row, "col": column, "by": "human"}]
            ai_reason, ai_search = self.last_ai_reason, self.last_ai_search
            if not self._finished(working):
                decision = self._analyze(working, self.ai_side)
                if self.move_selector is not None:
                    try:
                        selected = self.move_selector(working.copy(), self.ai_side, decision)
                        decision = {**decision, **selected}
                        ai_search = {key: selected[key] for key in
                                     ("nodes", "completed_depth", "proven_value", "elapsed_seconds", "budget_exhausted", "value_evaluations", "forcing", "engine", "candidate_width", "heuristic_score",
                                      "status", "deadline_overrun_seconds", "initial_selection", "initial_move",
                                      "guard_changed", "guard_probes", "rejected_moves", "selected_reply",
                                      "legal_count", "unexamined_count", "principal_variation",
                                      "interpretation", "time_limit", "max_nodes", "threat_time_limit", "threat_max_nodes", "threat_width", "threat_quiet_plies",
                                      "threat_total_plies", "attack_time_limit", "attack_max_nodes",
                                      "attack_result")
                                     if key in selected}
                        json.dumps(ai_search, allow_nan=False)
                    except Exception as exc:
                        logging.exception("AI move selection failed")
                        raise APIError(503, "AI搜索失败，本次操作尚未保存") from exc
                try:
                    choice = decision["move"]
                    if not isinstance(choice, (tuple, list)) or len(choice) != 2:
                        raise ValueError("AI没有返回合法落点")
                    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in choice):
                        raise ValueError("AI坐标必须为整数")
                    ai_row, ai_col = int(choice[0]), int(choice[1])
                    working = apply_board_move(working, (ai_row, ai_col), self.ai_side)
                except (TypeError, ValueError, IndexError) as exc:
                    raise APIError(503, "AI返回了非法落点，本次操作尚未保存") from exc
                history.append({"side": self.ai_side, "row": ai_row, "col": ai_col, "by": "ai"})
                ai_reason = str(decision.get("reason", "模型策略"))
            # Fresh human-turn analysis AFTER the actual AI placement.
            current = self._analyze(working, self.human_side)
            return self._commit(working, history, self.human_side, self.ai_side, current, ai_reason, ai_search)

    def _snapshot(self):
        winner = int(board_winner(self.board))
        finished = self._finished(self.board)
        return {"board": self.board.tolist(), "rows": self.board.shape[0], "cols": self.board.shape[1],
                "max_size": self.max_size, "engine": self.engine, "search_configuration": self.search_configuration, "revision": self.revision, "active": self.active,
                "human_side": self.human_side, "ai_side": self.ai_side,
                "turn": 0 if finished else self.human_side, "winner": winner, "finished": finished,
                "move_count": len(self.history), "history": list(self.history),
                "last_move": self.history[-1] if self.history else None,
                "last_ai_reason": self.last_ai_reason,
                "last_ai_search": self.last_ai_search,
                "analysis": {**self.analysis, "revision": self.revision} if self.analysis else None,
                "model_source": self.model_source, "trained": self.trained,
                "models": self.model_status,
                "session": {"run_id": self.persistence_identity.get("run_id", ""),
                            "configuration_sha256": hashlib.sha256(json.dumps(self.persistence_identity, sort_keys=True,
                            separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest(),
                            "persisted": self.state_file is not None, "pending_move": self.pending_move}}

    def snapshot(self):
        with self.lock:
            return self._snapshot()


class GameHandler(BaseHTTPRequestHandler):
    server_version = "LocalGomoku/1.0"

    def log_message(self, format, *args):
        logging.info("%s %s", self.client_address[0], format % args)

    def _check_origin(self, *, mutation=False):
        hosts = self.headers.get_all("Host", [])
        if hosts != [self.server.expected_host]:
            raise APIError(403, "仅接受本地服务地址的请求")
        origins = self.headers.get_all("Origin", [])
        if (origins and origins != [self.server.expected_origin]) or (mutation and not origins):
            raise APIError(403, "拒绝跨来源请求；请在本地游戏页面操作")
        fetch_site = self.headers.get("Sec-Fetch-Site")
        if fetch_site and fetch_site not in ("same-origin", "none"):
            raise APIError(403, "拒绝跨来源请求")

    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                         "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            self._check_origin()
            path = urlsplit(self.path)
            if path.scheme or path.netloc or path.query:
                raise APIError(404, "页面不存在")
            if path.path in ("/", "/index.html"):
                self._send(200, self.server.page_bytes, "text/html; charset=utf-8")
            elif path.path == "/api/state":
                self._send(200, self.server.game.snapshot())
            elif path.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                raise APIError(404, "页面不存在")
        except APIError as exc:
            self._send(exc.status, {"error": str(exc)})

    def do_POST(self):
        try:
            self._check_origin(mutation=True)
            path = urlsplit(self.path)
            if path.scheme or path.netloc or path.query or path.path not in ("/api/new", "/api/move"):
                raise APIError(404, "接口不存在")
            if self.headers.get_all("Transfer-Encoding"):
                raise APIError(400, "不支持分块请求")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isdigit():
                raise APIError(411, "需要合法Content-Length")
            length = int(lengths[0])
            if not 0 < length <= BODY_LIMIT:
                raise APIError(413, "请求体为空或过大")
            if (len(self.headers.get_all("Content-Type", [])) != 1
                    or self.headers.get_content_type() != "application/json"
                    or self.headers.get_content_charset() not in (None, "utf-8")):
                raise APIError(415, "仅接受application/json UTF-8请求")
            body = self.rfile.read(length)
            if len(body) != length:
                raise APIError(400, "请求体不完整")
            payload = strict_json(body)
            result = self.server.game.new_game(payload) if path.path == "/api/new" else self.server.game.move(payload)
            self._send(200, result)
        except APIError as exc:
            self._send(exc.status, {"error": str(exc)})
        except Exception:
            logging.exception("Unexpected local game error")
            self._send(500, {"error": "服务内部错误，请查看终端"})

    def do_OPTIONS(self):
        self._send(405, {"error": "不支持跨来源预检请求"})


class LocalGameServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, game, page_bytes):
        super().__init__(address, GameHandler)
        self.game = game
        self.page_bytes = page_bytes
        self.expected_host = f"127.0.0.1:{self.server_address[1]}"
        self.expected_origin = f"http://{self.expected_host}"


def run_self_test():
    """Exercise state and HTTP handlers in memory: no port, model, or server process."""
    from types import SimpleNamespace

    calls = []
    def fake_analyzer(board, side, opponent, play, tactical=True):
        calls.append((board.copy(), side))
        legal = board == 0
        terminal = board_winner(board) != 0 or not legal.any()
        probabilities = legal.astype(float)
        if terminal:
            probabilities.fill(0)
        else:
            probabilities /= probabilities.sum()
        move = None if terminal else tuple(map(int, np.argwhere(legal)[0]))
        return dict(opponent_policy=probabilities, play_policy=probabilities,
                    raw_play_policy=probabilities, move=move, reason="test",
                    coverage=legal.astype(int), window_count=1)

    selected_calls = []
    def fake_selector(board, side, analysis):
        selected_calls.append((board.copy(), side))
        assert analysis["side"] == side and np.asarray(analysis["raw_play_policy"]).shape == board.shape
        return dict(move=tuple(map(int, np.argwhere(board == 0)[-1])), reason="test search",
                    nodes=7, completed_depth=2, proven_value=None, elapsed_seconds=0.01, budget_exhausted=False)

    game = GameSession(None, None, analyzer=fake_analyzer, move_selector=fake_selector)
    snapshot = game.snapshot()
    assert snapshot["board"][8][8] == 1 and snapshot["human_side"] == 2
    before = snapshot["revision"]
    current = game.move({"row": 1, "col": 1, "revision": before})
    assert current["move_count"] == 3 and current["analysis"]["revision"] == current["revision"]
    assert np.array_equal(calls[-1][0], np.array(current["board"])) and calls[-1][1] == 2
    assert current["board"][15][15] == 1 and current["last_ai_search"]["nodes"] == 7
    assert selected_calls[-1][0][1, 1] == 2 and selected_calls[-1][1] == 1
    # Invalid search output cannot partially save the human placement.
    game.move_selector = lambda *args: dict(move=(2.5, 3), reason="invalid")
    saved = json.dumps(game.snapshot(), sort_keys=True)
    try:
        game.move({"row": 2, "col": 2, "revision": game.revision})
        raise AssertionError("fractional AI move accepted")
    except APIError as exc:
        assert exc.status == 503 and json.dumps(game.snapshot(), sort_keys=True) == saved
    game.move_selector = fake_selector
    try:
        game.move({"row": 2, "col": 2, "revision": before})
        raise AssertionError("stale move accepted")
    except APIError as exc:
        assert exc.status == 409
    server = SimpleNamespace(game=game, page_bytes=b"test page", expected_host="127.0.0.1:8765",
                             expected_origin="http://127.0.0.1:8765")
    class Socket:
        def __init__(self, body):
            self.body, self.output = body, bytearray()
        def makefile(self, *args):
            return io.BytesIO(self.body)
        def sendall(self, data):
            self.output.extend(data)
    def request(path, payload=b"{}", headers=None, method="POST"):
        base = {"Host": server.expected_host, "Origin": server.expected_origin,
                "Content-Type": "application/json", "Content-Length": str(len(payload))}
        base.update(headers or {})
        raw = (method + " " + path + " HTTP/1.0\r\n" +
               "".join(f"{k}: {v}\r\n" for k, v in base.items()) + "\r\n").encode() + payload
        sock = Socket(raw)
        GameHandler(sock, ("127.0.0.1", 12345), server)
        response = bytes(sock.output)
        return int(response.split(b" ", 2)[1]), response.split(b"\r\n\r\n", 1)[1]
    assert request("/", method="GET") == (200, b"test page")
    assert request("/api/state", method="GET")[0] == 200
    assert request("/missing", method="GET")[0] == 404
    assert request("/api/new", headers={"Origin": ""})[0] == 403
    assert request("/api/new", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert request("/api/new", headers={"Origin": "https://example.com"})[0] == 403
    assert request("/api/new", headers={"Host": "other.invalid"})[0] == 403
    assert request("/api/new", b'{"rows":16,"rows":17}')[0] == 400
    assert request("/api/new", b'{"rows":NaN}')[0] == 400
    assert request("/api/new", b'{"rows":6.0}')[0] == 400
    assert request("/api/new", headers={"Content-Type": "text/plain"})[0] == 415
    assert request("/api/new", b'{"rows":5,"cols":16}')[0] == 400
    assert request("/api/new", b'{"rows":7,"cols":9,"human_first":true,"forbidden":[[0,0]]}')[0] == 200
    assert game.snapshot()["rows"] == 7 and game.snapshot()["cols"] == 9
    payload = json.dumps({"row": 0, "col": 0, "revision": game.revision}).encode()
    assert request("/api/move", payload)[0] == 400
    assert request("/api/move", b'{"board":[]}')[0] == 400
    # A nearly forbidden board ends in a draw after its only cell is played.
    forbidden = [[row, col] for row in range(6) for col in range(6) if (row, col) != (0, 0)]
    lone = game.new_game({"rows": 6, "cols": 6, "human_first": True, "forbidden": forbidden})
    end = game.move({"row": 0, "col": 0, "revision": lone["revision"]})
    assert end["finished"] and end["winner"] == 0 and end["move_count"] == 1
    assert end["analysis"]["move"] is None and np.asarray(end["analysis"]["play_policy"]).sum() == 0
    assert end["last_ai_search"] is None
    try:
        game.move({"row": 1, "col": 1, "revision": end["revision"]})
        raise AssertionError("terminal move accepted")
    except APIError as exc:
        assert exc.status == 409
    # The real selector is usable independently of neural weights or a listener.
    from board_search import select_move
    small = np.zeros((6, 7), dtype=np.uint8)
    small[2, 0:4] = 1
    selected = select_move(small, 2, (small == 0).astype(float), max_nodes=10, time_limit=0.01, depth=1)
    assert selected["move"] == (2, 4)
    print("PASS: in-memory game/HTTP checks; no network listener or models were started")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", help="directory containing trained opponent.pt and play.pt")
    parser.add_argument("--strategy", help="optional independently trained gomoku_global_v1 checkpoint")
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-size", type=int, default=64, help="maximum rows/columns, 6..128")
    parser.add_argument("--search-seconds", type=float, default=1.0, help="tree-search budget per AI move (model inference and mandatory tactic checks excluded)")
    parser.add_argument("--search-nodes", type=int, default=2000)
    parser.add_argument("--search-depth", type=int, default=5)
    parser.add_argument("--search-width", type=int, default=12)
    parser.add_argument("--engine", choices=("python", "native", "guarded"), default="python")
    parser.add_argument("--forcing-seconds", type=float, default=None, help="native only; independent root VCF budget")
    parser.add_argument("--guard-native-fraction", type=float, default=None, help="guarded only; fraction of shared budget for initial search")
    parser.add_argument("--guard-probe-nodes", type=int, default=None, help="guarded only; per-child VCF node cap within shared budget")
    parser.add_argument("--guard-probe-seconds", type=float, default=None, help="guarded only; per-child VCF time cap within shared budget")
    parser.add_argument("--guard-threat-seconds", type=float, default=None, help="guarded only; quiet-threat proof time cap within shared budget, default 0.2")
    parser.add_argument("--guard-threat-nodes", type=int, default=None, help="guarded only; quiet-threat node cap within shared budget, default 20000")
    parser.add_argument("--guard-threat-width", type=int, default=None, help="guarded only; quiet-threat candidate width, default 16")
    parser.add_argument("--guard-threat-quiet-plies", type=int, default=None, help="guarded only; quiet attacker moves in a proof, default 2")
    parser.add_argument("--guard-threat-total-plies", type=int, default=None, help="guarded only; total proof depth, default 64")
    parser.add_argument("--guard-attack-seconds", type=float, default=None, help="guarded only; active attack proof time cap within shared budget, default 0.15")
    parser.add_argument("--guard-attack-nodes", type=int, default=None, help="guarded only; active attack proof node cap within shared budget, default 20000")
    parser.add_argument("--forcing-nodes", type=int, default=None)
    parser.add_argument("--forcing-depth", type=int, default=None, help="default 24 native, 32 guarded")
    parser.add_argument("--value-weight", type=float, default=40.0, help="global neural value weight in leaf ranking")
    parser.add_argument("--state-file", help="optional atomically persisted game, restored on restart")
    parser.add_argument("--run-id", default="", help="immutable identity for a persisted acceptance session")
    parser.add_argument("--self-test", action="store_true", help="run model-free, in-memory handler checks")
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    guard_fields = ("guard_native_fraction", "guard_probe_nodes", "guard_probe_seconds",
                    "guard_threat_seconds", "guard_threat_nodes", "guard_threat_width",
                    "guard_threat_quiet_plies", "guard_threat_total_plies", "guard_attack_seconds", "guard_attack_nodes")
    if args.engine == "guarded":
        if args.forcing_seconds is not None or args.forcing_nodes is not None:
            parser.error("--forcing-seconds/--forcing-nodes do not apply to guarded; use --guard-probe-* within the shared budget")
        for name, default in zip(guard_fields, (.45, 5000, .05, .2, 20000, 16, 2, 64, .15, 20000)):
            if getattr(args, name) is None:
                setattr(args, name, default)
    else:
        if any(getattr(args, name) is not None for name in guard_fields):
            parser.error("--guard-* options require --engine guarded")
        args.forcing_seconds = .1 if args.forcing_seconds is None else args.forcing_seconds
        args.forcing_nodes = 10000 if args.forcing_nodes is None else args.forcing_nodes
    args.forcing_depth = (32 if args.engine == "guarded" else 24) if args.forcing_depth is None else args.forcing_depth
    if args.engine in ("native", "guarded") and args.max_size > 64:
        parser.error("native and guarded engines require --max-size <= 64")
    if args.engine == "guarded" and (not math.isfinite(args.guard_native_fraction)
            or not 0 <= args.guard_native_fraction <= 1 or args.guard_probe_nodes < 1
            or not math.isfinite(args.guard_probe_seconds) or args.guard_probe_seconds <= 0
            or not math.isfinite(args.guard_threat_seconds) or args.guard_threat_seconds < 0
            or args.guard_threat_nodes < 1 or args.guard_threat_width < 1
            or not 0 <= args.guard_threat_quiet_plies <= 32 or not 1 <= args.guard_threat_total_plies <= 256
            or not math.isfinite(args.guard_attack_seconds) or args.guard_attack_seconds < 0
            or args.guard_attack_nodes < 1):
        parser.error("invalid guarded search allocation")
    if not args.models:
        parser.error("--models is required to start the game")
    if not 0 <= args.port <= 65535 or args.threads < 1 or not 6 <= args.max_size <= 128:
        parser.error("invalid port, thread count, or max-size")
    if (not math.isfinite(args.search_seconds) or args.search_seconds <= 0 or args.search_nodes < 1
            or args.search_depth < 1 or args.search_width < 1
            or not math.isfinite(args.value_weight) or args.value_weight < 0
            or args.forcing_seconds is not None and (not math.isfinite(args.forcing_seconds) or args.forcing_seconds < 0)
            or args.forcing_nodes is not None and args.forcing_nodes < 1 or args.forcing_depth < 1):
        parser.error("search budgets, depth and width must be positive and finite")
    import torch
    from unet_pipeline import load_model
    torch.set_num_threads(args.threads)
    directory = Path(args.models).resolve()
    opponent, opponent_meta = load_model(directory / "opponent.pt", "opponent")
    play, play_meta = load_model(directory / "play.pt", "play")
    global_model, global_meta, strategy_path = None, {}, None
    if args.strategy:
        from global_inference import load_global_model
        strategy_path = Path(args.strategy).resolve()
        global_model, global_meta = load_global_model(strategy_path)
    analyzer = build_analyzer(global_model)

    search = build_move_selector(global_model, search_seconds=args.search_seconds,
                                 search_nodes=args.search_nodes, search_depth=args.search_depth,
                                 search_width=args.search_width, value_weight=args.value_weight,
                                 engine=args.engine, forcing_seconds=args.forcing_seconds,
                                 forcing_nodes=args.forcing_nodes, forcing_depth=args.forcing_depth,
                                 guard_native_fraction=args.guard_native_fraction if args.guard_native_fraction is not None else .45,
                                 guard_probe_nodes=args.guard_probe_nodes if args.guard_probe_nodes is not None else 5000,
                                 guard_probe_seconds=args.guard_probe_seconds if args.guard_probe_seconds is not None else .05,
                                 guard_threat_seconds=args.guard_threat_seconds if args.guard_threat_seconds is not None else .2,
                                 guard_threat_nodes=args.guard_threat_nodes if args.guard_threat_nodes is not None else 20000,
                                 guard_threat_width=args.guard_threat_width if args.guard_threat_width is not None else 16,
                                 guard_threat_quiet_plies=args.guard_threat_quiet_plies if args.guard_threat_quiet_plies is not None else 2,
                                 guard_threat_total_plies=args.guard_threat_total_plies if args.guard_threat_total_plies is not None else 64,
                                 guard_attack_seconds=args.guard_attack_seconds if args.guard_attack_seconds is not None else .15,
                                 guard_attack_nodes=args.guard_attack_nodes if args.guard_attack_nodes is not None else 20000)

    model_status = {
        "opponent": {"loaded": True, "trained": bool(opponent_meta.get("trained")), "source": str(directory / "opponent.pt")},
        "play": {"loaded": True, "trained": bool(play_meta.get("trained")), "source": str(directory / "play.pt")},
        "global": {"loaded": global_model is not None, "trained": bool(global_meta.get("trained")),
                   "value_used_in_search": args.engine == "python" and global_model is not None,
                   "source": str(strategy_path) if strategy_path else ""},
    }
    native_identity = None
    if args.engine in ("native", "guarded"):
        from native_search import native_fingerprint
        native_identity = native_fingerprint()
    persistence_identity = {}
    if args.state_file:
        sources = Path(__file__).resolve().parent
        persistence_identity = {
            "run_id": args.run_id,
            "native": native_identity,
            "configuration": {key: value for key, value in vars(args).items()
                              if key not in ("port", "host", "state_file", "self_test")},
            "weights": {role: hashlib.sha256(Path(info["source"]).read_bytes()).hexdigest()
                        for role, info in model_status.items() if info["loaded"]},
            "sources": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                        for file in sorted(sources.iterdir())
                        if file.is_file() and file.suffix.lower() in (".py", ".c", ".dll", ".mjs")},
        }
    game = GameSession(opponent, play, analyzer=analyzer, move_selector=search, max_size=args.max_size,
                       model_source=directory, model_status=model_status,
                       state_file=args.state_file, persistence_identity=persistence_identity, engine=args.engine,
                       search_configuration={key: value for key, value in vars(args).items()
                           if value is not None and (key.startswith("search_") or key.startswith("guard_") or key.startswith("forcing_"))},
                       trained=bool(opponent_meta.get("trained") and play_meta.get("trained")))
    page = (Path(__file__).resolve().parent / "web" / "unet_game.html").read_bytes()
    with LocalGameServer((args.host, args.port), game, page) as server:
        print(f"本地棋局：{server.expected_origin}/", flush=True)
        default = min(BOARD_DEFAULT, args.max_size)
        print(f"默认{default}×{default}，允许6..{args.max_size}行列；AI黑棋先手中心。Ctrl+C结束。", flush=True)
        print(f"AI搜索预算{args.search_seconds:g}秒、{args.search_nodes}节点、最多{args.search_depth}层；模型推理与必要战术检查另计。", flush=True)
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass
    game.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
