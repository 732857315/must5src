"""Bounded whole-board continuous-four search with explicit proof boundaries.

All five-cell segments are examined, including board edges and both diagonals.
At a quiet node only moves creating an immediate winning threat are searched.
Omitting other moves can establish a win, but never a loss or draw. A loss is
certified only by independent immediate enemy wins, a forced-defense chain,
or complete coverage of every legal move. Neural values and beam pruning are
not used. Unknown means no certificate was found within these restrictions.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Integral, Real
import time

import numpy as np

from board_rules import normalize_board


@lru_cache(maxsize=32)
def _segments(shape):
    rows, cols = shape
    lines = []
    for row in range(rows):
        for col in range(cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if 0 <= row + 4 * dr < rows and 0 <= col + 4 * dc < cols:
                    lines.append([(row + i * dr) * cols + col + i * dc for i in range(5)])
    result = np.asarray(lines, dtype=np.intp).reshape(-1, 5)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class _Proof:
    value: int
    line: tuple
    reason: str
    evidence: tuple = ()


class _BudgetExceeded(Exception):
    pass


class _ForcingSearch:
    def __init__(self, board, max_nodes, deadline):
        self.board = board
        self.flat = board.reshape(-1)
        self.lines = _segments(board.shape)
        self.max_nodes = max_nodes
        self.deadline = deadline
        self.nodes = 0
        self.cache_hits = 0
        self.candidate_moves = 0
        self.max_ply = 0
        self.depth_limited = False
        self.cache = {}

    def check(self):
        if self.nodes >= self.max_nodes or time.monotonic() >= self.deadline:
            raise _BudgetExceeded

    def _facts(self, side):
        cells = self.flat[self.lines]
        empty = cells == 0
        own = (cells == side).sum(1)
        enemy = (cells == 3 - side).sum(1)
        empties = empty.sum(1)
        own_four = (own == 4) & (empties == 1)
        enemy_four = (enemy == 4) & (empties == 1)
        wins = np.unique(self.lines[own_four][empty[own_four]]).tolist()
        blocks = np.unique(self.lines[enemy_four][empty[enemy_four]]).tolist()
        proposals = (own == 3) & (empties == 2)
        forcing = np.unique(self.lines[proposals][empty[proposals]]).tolist()
        # Count evidence only; it changes ordering, never candidate inclusion.
        counts = np.bincount(self.lines[proposals][empty[proposals]], minlength=self.flat.size)
        forcing.sort(key=lambda move: (-int(counts[move]), move))
        terminal = 1 if np.any(own == 5) else -1 if np.any(enemy == 5) else None
        if time.monotonic() >= self.deadline:
            raise _BudgetExceeded
        return terminal, wins, blocks, forcing

    def search(self, side, depth, ply=0):
        self.check()
        self.nodes += 1
        self.max_ply = max(self.max_ply, ply)
        key = (self.flat.tobytes(), side, depth)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        terminal, wins, blocks, forcing = self._facts(side)
        if terminal is not None:
            answer = _Proof(terminal, (), 'terminal_five')
        elif wins:
            answer = _Proof(1, ((side, wins[0], 'immediate_win'),), 'immediate_win', tuple(wins))
        elif len(blocks) >= 2:
            # With no own immediate win, defending one distinct winning cell
            # leaves the other intact. Other legal moves lose even sooner.
            answer = _Proof(-1, ((side, blocks[0], 'representative_defense'),
                                (3 - side, blocks[1], 'immediate_win')),
                            'two_independent_enemy_wins', tuple(blocks))
        elif not np.any(self.flat == 0):
            answer = _Proof(0, (), 'terminal_draw')
        elif depth <= 0:
            self.depth_limited = True
            return None
        else:
            candidates = blocks if blocks else forcing
            # A unique immediate enemy win leaves exactly one non-losing
            # response. Otherwise all quiet moves omitted here remain unknown.
            complete = bool(blocks) or len(candidates) == int(np.count_nonzero(self.flat == 0))
            children = []
            for move in candidates:
                self.check()
                self.candidate_moves += 1
                self.flat[move] = side
                try:
                    child = self.search(3 - side, depth - 1, ply + 1)
                finally:
                    self.flat[move] = 0
                if child is None:
                    children.append(None)
                    continue
                label = 'mandatory_defense' if blocks else 'forcing_four'
                proof = _Proof(-child.value, ((side, move, label),) + child.line,
                               'forced_defense_chain' if blocks else 'continuous_four', child.evidence)
                if proof.value == 1:
                    self.cache[key] = proof
                    return proof
                children.append(proof)
            if complete and children and all(child is not None for child in children):
                # For a proved loss retain a longest representative resistance.
                answer = max(children, key=lambda child: (child.value, len(child.line)))
            else:
                return None
        self.cache[key] = answer
        return answer


def solve_forcing(grid, side, *, max_nodes=20000, time_limit=0.25, max_depth=16):
    """Find a continuous-four certificate for the actual side to move.

    Returns status win/loss/draw/unknown, actor-side proven_value, legal move,
    principal_variation with actual colors, terminal evidence, and usage. A
    line illustrates forced resistance; alternatives to an immediate required
    defense lose sooner. max_depth bounds expanded plies; immediate tactical
    certificates at a leaf may append one or two further plies to the line.
    All budgets include proof-cache hits. Unknown has no recommended move.
    """
    if isinstance(side, (bool, np.bool_)) or not isinstance(side, Integral) or side not in (1, 2):
        raise ValueError('side must be explicit BLACK=1 or WHITE=2')
    for name, value in (('max_nodes', max_nodes), ('max_depth', max_depth)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if isinstance(time_limit, (bool, np.bool_)) or not isinstance(time_limit, Real) or not math.isfinite(time_limit) or time_limit < 0:
        raise ValueError('time_limit must be finite and nonnegative')
    board = normalize_board(grid)
    side = int(side)
    started = time.monotonic()
    search = _ForcingSearch(board, int(max_nodes), started + float(time_limit))
    exhausted = False
    try:
        proof = search.search(side, int(max_depth))
    except _BudgetExceeded:
        proof, exhausted = None, True
    line = []
    if proof is not None:
        for actor, index, kind in proof.line:
            row, col = divmod(index, board.shape[1])
            line.append(dict(side=actor, move=(row, col), kind=kind))
    value = None if proof is None else proof.value
    status = 'unknown' if proof is None else {1: 'win', -1: 'loss', 0: 'draw'}[value]
    reason = ('budget_exhausted' if exhausted else 'depth_limit_or_no_continuous_four') if proof is None else proof.reason
    evidence = [] if proof is None else [tuple(map(int, divmod(index, board.shape[1]))) for index in proof.evidence]
    return dict(status=status, proven_value=value, winner=(side if value == 1 else 3 - side if value == -1 else 0 if value == 0 else None),
                move=line[0]['move'] if line else None,
                principal_variation=line, proof_plies=len(line), reason=reason,
                final_tactical_cells=evidence,
                nodes=search.nodes, candidate_moves=search.candidate_moves, cache_hits=search.cache_hits,
                max_ply=search.max_ply, max_depth=int(max_depth), max_nodes=int(max_nodes),
                time_limit=float(time_limit), elapsed_seconds=time.monotonic() - started,
                budget_exhausted=exhausted, depth_limited=search.depth_limited,
                proof_scope='current position and actual side; unknown is not a draw or a loss')
