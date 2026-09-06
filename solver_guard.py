"""Compatibility wrapper around the shared, budgeted exact solver."""

import time

from exact_solver import DEFAULT_MAX_NODES, DEFAULT_TIME_LIMIT, ExactSolver
from exact_solver import SearchBudget, SearchLimitExceeded
from game import initial_state, legal_moves, winner
from game import ordered_moves as _ordered_move_groups


TT = {}
NODES = 0
_winner_of = winner


def ordered_moves(state, me):
    """Retain the original flat move-list API."""
    wins, blocks, rest = _ordered_move_groups(state, me)
    return wins + blocks + rest


def negamax(state, *, max_nodes=DEFAULT_MAX_NODES, time_limit=DEFAULT_TIME_LIMIT):
    global NODES
    engine = ExactSolver(TT)
    try:
        return engine.solve(state, budget=SearchBudget(max_nodes, time_limit))
    finally:
        NODES += engine.nodes


def main():
    started = time.monotonic()
    try:
        value = negamax(initial_state())
    except SearchLimitExceeded as exc:
        raise SystemExit(f"Search stopped: {exc}.") from exc
    print(
        f"value(white view)={value} nodes={NODES} tt={len(TT)} "
        f"time={time.monotonic() - started:.1f}s"
    )


if __name__ == "__main__":
    main()