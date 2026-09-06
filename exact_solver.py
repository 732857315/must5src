"""Shared, budgeted exact search for the 5x5 game.

Only fully determined values enter the transposition table. Exceeding a budget
raises SearchLimitExceeded; it never turns an unfinished search into a draw.
"""

import math
import time

from game import CENTER_INDEX, EMPTY, legal_moves, ordered_moves
from game import player_at, validate_state, winner


DEFAULT_MAX_NODES = 100_000
DEFAULT_TIME_LIMIT = 10.0


class SearchLimitExceeded(RuntimeError):
    """The requested exact result could not be completed within its budget."""


class SearchBudget:
    """A shared budget for a search or a complete collection/labeling run.

Nodes count uncached search positions and, when collecting the whole game
tree, collected positions. Pass None explicitly to disable either limit.
"""

    def __init__(self, max_nodes=DEFAULT_MAX_NODES, time_limit=DEFAULT_TIME_LIMIT):
        if max_nodes is not None and (
            isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 0
        ):
            raise ValueError("max_nodes must be a nonnegative integer or None")
        if time_limit is not None and (
            not math.isfinite(time_limit) or time_limit < 0
        ):
            raise ValueError("time_limit must be finite and nonnegative, or None")
        self.max_nodes = max_nodes
        self.nodes = 0
        self.deadline = None if time_limit is None else time.monotonic() + time_limit

    def check(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise SearchLimitExceeded("exact search exceeded its time limit")

    def consume_node(self):
        self.check()
        if self.max_nodes is not None and self.nodes >= self.max_nodes:
            raise SearchLimitExceeded(
                f"exact search exceeded its node limit ({self.max_nodes})"
            )
        self.nodes += 1


def search_moves(state):
    """Return empty cells; only a completely empty full board forces center.

    A window containing boundary/forbidden cells has its own legal mask, even
    before any stones have been placed, and must not use the center rule.
    """
    if winner(state) != EMPTY:
        return []
    return [CENTER_INDEX] if state == 0 else legal_moves(state)


class ExactSolver:
    def __init__(self, table=None):
        self.table = {} if table is None else table
        self.nodes = 0

    def solve(self, state, *, budget=None):
        validate_state(state)
        if budget is None:
            budget = SearchBudget()
        return self._search(state, budget)

    def analyze(self, state, *, budget=None):
        """Return the exact value and every optimal move in board coordinates."""
        validate_state(state)
        if budget is None:
            budget = SearchBudget()
        value = self._search(state, budget)
        best = []
        me = player_at(state)
        for move in search_moves(state):
            child = state | (me << (2 * move))
            # A winning search can stop early, so this child may not be cached.
            if -self._search(child, budget) == value:
                best.append(move)
        return value, best

    def _search(self, state, budget):
        budget.check()
        hit = self.table.get(state)
        if hit is not None:
            return hit
        budget.consume_node()
        self.nodes += 1
        if winner(state) != EMPTY:
            self.table[state] = -1
            return -1

        me = player_at(state)
        if state == 0:
            moves = [CENTER_INDEX]
        else:
            wins, blocks, rest = ordered_moves(state, me)
            moves = wins + blocks + rest
        if not moves:
            self.table[state] = 0
            return 0

        best = -1
        for move in moves:
            # The candidate list contains only empty cells of a live position.
            value = -self._search(state | (me << (2 * move)), budget)
            best = max(best, value)
            if best == 1:
                break
        # This point is reached only after proving the value. Interrupted
        # parents stay absent, while previously proved children remain useful.
        self.table[state] = best
        return best
