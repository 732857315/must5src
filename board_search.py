"""Budgeted whole-board search, keeping heuristic rankings separate from proofs.

Neural priors order plausible moves. Complete-board immediate wins and forcing
defenses always precede those priors. Candidate pruning cannot prove a loss or
draw; only a fully covered subtree or an explicit forcing tactic can do that.
"""
from functools import lru_cache
import math
from numbers import Integral
import time

import numpy as np

from board_rules import normalize_board, board_winner, tactical_candidates

MATE = 1_000_000.0
_WEIGHTS = np.array([0.0, 1.0, 6.0, 40.0, 400.0, 100_000.0])


@lru_cache(maxsize=32)
def _segments(shape, length=5):
    rows, columns = shape
    result = []
    for row in range(rows):
        for column in range(columns):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if 0 <= row + (length - 1) * dr < rows and 0 <= column + (length - 1) * dc < columns:
                    result.append([(row + step * dr) * columns + column + step * dc for step in range(length)])
    return np.array(result, dtype=np.int64).reshape(-1, length)


def _upgrade_moves(board, side):
    segments = _segments(board.shape, 6)
    cells = board.reshape(-1)[segments]
    middle = cells[:, 1:5]
    selected = ((cells[:, 0] == 0) & (cells[:, 5] == 0)
                & ((middle == side).sum(1) == 3) & ((middle == 0).sum(1) == 1))
    gaps = segments[selected, 1:5][middle[selected] == 0]
    return [tuple(map(int, np.unravel_index(index, board.shape))) for index in np.unique(gaps)]


def _features(board, side):
    segments = _segments(board.shape)
    cells = board.reshape(-1)[segments]
    own = (cells == side).sum(1)
    enemy = (cells == 3 - side).sum(1)
    blocked = (cells == 3).any(1)
    return segments, own, enemy, blocked


def _heuristic(board, side):
    _, own, enemy, blocked = _features(board, side)
    value = (_WEIGHTS[own] * ((enemy == 0) & ~blocked)).sum()
    value -= (_WEIGHTS[enemy] * ((own == 0) & ~blocked)).sum()
    # Never allow a static pattern score to masquerade as a solved result.
    return float(np.clip(value, -MATE / 4, MATE / 4))


def _frontier(board):
    occupied = (board == 1) | (board == 2)
    nearby = np.zeros(board.shape, dtype=bool)
    rows, columns = board.shape
    if not occupied.any():
        nearby[rows // 2, columns // 2] = True
        if board[rows // 2, columns // 2] != 0:
            nearby = board == 0
        return nearby & (board == 0)
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            source_r = slice(max(0, -dr), min(rows, rows - dr))
            source_c = slice(max(0, -dc), min(columns, columns - dc))
            target_r = slice(max(0, dr), min(rows, rows + dr))
            target_c = slice(max(0, dc), min(columns, columns + dc))
            nearby[target_r, target_c] |= occupied[source_r, source_c]
    return nearby & (board == 0)


def _ranked(board, side, priors, width, preferred=None):
    segments, own, enemy, blocked = _features(board, side)
    attack = (_WEIGHTS[np.minimum(own + 1, 5)] - _WEIGHTS[own]) * ((enemy == 0) & ~blocked)
    defense = (_WEIGHTS[np.minimum(enemy + 1, 5)] - _WEIGHTS[enemy]) * ((own == 0) & ~blocked)
    score = np.zeros(board.size, dtype=np.float64)
    np.add.at(score, segments.reshape(-1), np.repeat(attack + 1.1 * defense, 5))
    maximum = float(priors.max())
    if maximum > 0:
        score += 12.0 * priors.reshape(-1) / maximum
    candidates = [tuple(map(int, move)) for move in np.argwhere(_frontier(board))]
    if not candidates:
        candidates = [tuple(map(int, move)) for move in np.argwhere(board == 0)]
    center = (np.array(board.shape) - 1) / 2
    candidates.sort(key=lambda move: (move != preferred, -score[np.ravel_multi_index(move, board.shape)],
                                     float(np.square(np.array(move) - center).sum()), move))
    return candidates[:width], len(candidates) == int((board == 0).sum()) and len(candidates) <= width


class _BudgetExceeded(Exception):
    pass


class _Search:
    def __init__(self, board, side, priors, max_nodes, deadline, width, value_evaluator=None, value_weight=40.0):
        self.board = board
        self.root_side = side
        self.priors = priors
        self.max_nodes = max_nodes
        self.deadline = deadline
        self.width = width
        self.nodes = 0
        self.cache = {}
        self.value_evaluator = value_evaluator
        self.value_weight = value_weight
        self.value_cache = {}

    def evaluate(self, board, side):
        score = _heuristic(board, side)
        if self.value_evaluator is not None:
            self.check()
            key = (board.tobytes(), side)
            if key not in self.value_cache:
                supplied = self.value_evaluator(board.copy(), side)
                if isinstance(supplied, (bool, np.bool_, str, bytes)):
                    raise ValueError("global value must be a numeric scalar within [-1,1]")
                try:
                    value = float(supplied)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("global value must be a numeric scalar within [-1,1]") from exc
                if not math.isfinite(value) or not -1 <= value <= 1:
                    raise ValueError("global value must be finite within [-1,1]")
                self.value_cache[key] = value
                # Inference is atomic, but an overrun must not make a partial
                # iteration look completed. Count the performed call above.
                self.check()
            score += self.value_weight * self.value_cache[key]
        return float(np.clip(score, -MATE / 3, MATE / 3))

    def forcing_move(self, board, side, blocks):
        # A six-cell pattern only proposes a move; complete-board checks prove it.
        for move in _upgrade_moves(board, side):
            if blocks and move not in blocks:
                continue
            self.check()
            self.nodes += 1
            board[move] = side
            try:
                enemy_wins, own_wins, _ = self.tactics(board, 3 - side)
                proved = not enemy_wins and len(own_wins) >= 2
            finally:
                board[move] = 0
            if proved:
                return move
        return None

    def check(self):
        if self.nodes >= self.max_nodes or time.monotonic() >= self.deadline:
            raise _BudgetExceeded

    def tactics(self, board, side):
        key = (board.tobytes(), side)
        if key not in self.cache:
            self.cache[key] = tactical_candidates(board, side)
        return self.cache[key]

    def search(self, board, side, depth, alpha, beta, quiescence=4):
        self.check()
        self.nodes += 1
        result = board_winner(board)
        if result:
            value = 1 if result == side else -1
            return value * MATE, value
        wins, blocks, rest = self.tactics(board, side)
        if wins:
            return MATE, 1
        if len(blocks) > 1:
            return -MATE, -1
        if not blocks and not rest:
            return 0.0, 0
        if self.forcing_move(board, side, blocks) is not None:
            return MATE, 1
        if depth <= 0 and (not blocks or quiescence <= 0):
            return self.evaluate(board, side), None
        if blocks:
            candidates, complete = blocks, True
        else:
            # Priors belong to the root actor; deeper opposing turns use only
            # the full-board line ordering, not the root's neural distribution.
            priors = self.priors if side == self.root_side else np.zeros(board.shape)
            candidates, complete = _ranked(board, side, priors, self.width)
        best = -math.inf
        proofs = []
        explored_all = True
        for move in candidates:
            self.check()
            board[move] = side
            try:
                score, proof = self.search(board, 3 - side, max(0, depth - 1), -beta, -alpha,
                                           quiescence - int(depth <= 0))
            finally:
                board[move] = 0
            score = -score
            own_proof = None if proof is None else -proof
            proofs.append(own_proof)
            best = max(best, score)
            # One move whose child is a proved loss proves the parent win.
            if own_proof == 1:
                return MATE, 1
            alpha = max(alpha, score)
            if alpha >= beta:
                explored_all = False
                break
        proof = None
        if complete and explored_all and all(value is not None for value in proofs):
            proof = max(proofs)
        return best, proof


def select_move(grid, side, priors, *, max_nodes=2000, time_limit=1.0, depth=5, candidate_width=12,
                value_evaluator=None, value_weight=40.0):
    """Return a legal move, with an optional certified result and bounded search.

    proven_value belongs to side and is None for an unproved heuristic result.
    The time budget covers tree search; input validation and the mandatory
    complete-board immediate-tactic checks run first.
    """
    board = normalize_board(grid)
    if isinstance(side, bool) or not isinstance(side, Integral) or side not in (1, 2):
        raise ValueError("side must be BLACK=1 or WHITE=2")
    side = int(side)
    for value, name, minimum in ((max_nodes, "max_nodes", 0), (depth, "depth", 1),
                                  (candidate_width, "candidate_width", 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(time_limit, bool) or not isinstance(time_limit, (int, float)) or not math.isfinite(time_limit) or time_limit < 0:
        raise ValueError("time_limit must be finite and nonnegative")
    if value_evaluator is not None and not callable(value_evaluator):
        raise ValueError("value_evaluator must be callable")
    if isinstance(value_weight, bool) or not isinstance(value_weight, (int, float)) or not math.isfinite(value_weight) or value_weight < 0:
        raise ValueError("value_weight must be finite and nonnegative")
    priors = np.asarray(priors, dtype=np.float64)
    if priors.shape != board.shape or not np.isfinite(priors).all() or (priors < 0).any():
        raise ValueError("priors must be finite nonnegative values with the board shape")
    priors = np.where(board == 0, priors, 0.0)
    started = time.monotonic()
    result = board_winner(board)
    if result or not (board == 0).any():
        return dict(move=None, reason="棋局已结束", nodes=0, completed_depth=0,
                    proven_value=(1 if result == side else -1) if result else 0,
                    elapsed_seconds=time.monotonic() - started, budget_exhausted=False)
    wins, blocks, rest = tactical_candidates(board, side)
    if wins or len(blocks) > 1:
        candidates = wins or blocks
        move = max(candidates, key=lambda point: (priors[point], -point[0], -point[1]))
        return dict(move=move, reason="全盘立即成五" if wins else "对手有多个独立成五点，当前局面已无法一手全堵",
                    nodes=0, completed_depth=0, proven_value=1 if wins else -1,
                    elapsed_seconds=time.monotonic() - started, budget_exhausted=False)
    if blocks:
        candidates, complete = blocks, True
    else:
        candidates, complete = _ranked(board, side, priors, candidate_width)
    choice = candidates[0]
    reason = "封堵对手唯一立即成五点" if blocks else "全盘线型与 U-Net 偏好排序"
    engine = _Search(board, side, priors, max_nodes, time.monotonic() + time_limit, candidate_width,
                     value_evaluator=value_evaluator, value_weight=value_weight)
    completed_depth, certificate, exhausted = 0, None, False
    best_score = _heuristic(board, side)
    try:
        forced = engine.forcing_move(board, side, blocks)
    except _BudgetExceeded:
        forced, exhausted = None, True
    if forced is not None:
        return dict(move=forced, reason="形成活四并验证两个独立成五点，对手无法一手全堵",
                    nodes=engine.nodes, completed_depth=0, proven_value=1,
                    elapsed_seconds=time.monotonic() - started, budget_exhausted=False,
                    value_evaluations=len(engine.value_cache))
    for current_depth in range(1, depth + 1):
        if not blocks:
            candidates, complete = _ranked(board, side, priors, candidate_width, preferred=choice)
        local_best, local_choice, proofs = -math.inf, choice, []
        try:
            for move in candidates:
                engine.check()
                board[move] = side
                try:
                    score, proof = engine.search(board, 3 - side, current_depth - 1, -math.inf, -local_best)
                finally:
                    board[move] = 0
                score = -score
                proofs.append(None if proof is None else -proof)
                if score > local_best:
                    local_best, local_choice = score, move
                if proof == -1:
                    choice, certificate = move, 1
                    reason = "全盘搜索已证明此着可强制获胜"
                    completed_depth = current_depth
                    break
            else:
                choice, best_score = local_choice, local_best
                completed_depth = current_depth
                if complete and all(proof is not None for proof in proofs):
                    certificate = max(proofs)
                if not blocks:
                    reason = f"全盘有界搜索完成 {current_depth} 层"
                if certificate is not None:
                    break
                continue
            break
        except _BudgetExceeded:
            exhausted = True
            # A partially explored iteration is never a completed search proof.
            break
    return dict(move=choice, reason=reason, nodes=engine.nodes, completed_depth=completed_depth,
                proven_value=certificate, heuristic_score=best_score,
                elapsed_seconds=time.monotonic() - started, budget_exhausted=exhausted,
                candidate_width=candidate_width, value_evaluations=len(engine.value_cache))
