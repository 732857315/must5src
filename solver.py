"""Compatibility entry point for exact search and complete state collection."""

import pickle
import time

from exact_solver import DEFAULT_MAX_NODES, DEFAULT_TIME_LIMIT, ExactSolver
from exact_solver import SearchBudget, SearchLimitExceeded, search_moves
from game import Board, apply_move, player_at, validate_state


TT = {}


def negamax(b, alpha=-1, beta=1, *, max_nodes=DEFAULT_MAX_NODES,
            time_limit=DEFAULT_TIME_LIMIT):
    """Return the exact value: 1 = win, -1 = loss, 0 = draw.

    alpha and beta remain accepted for compatibility with the original API.
    The shared solver always returns an exact value or raises on exhaustion.
    """
    budget = SearchBudget(max_nodes, time_limit)
    return ExactSolver(TT).solve(b.pack(), budget=budget)


def solve_and_collect(store, *, max_nodes=DEFAULT_MAX_NODES,
                      time_limit=DEFAULT_TIME_LIMIT, start_state=0):
    """Collect reachable states under one budget, with a forced center opening.

    store maps packed states to (exact value, all optimal moves). On budget
    exhaustion, SearchLimitExceeded propagates and any entries already added
    remain exact. start_state permits bounded endgame collection.
    """
    global TT
    validate_state(start_state)
    budget = SearchBudget(max_nodes, time_limit)
    TT = {}
    engine = ExactSolver(TT)
    started = time.monotonic()
    visited = set()
    pending = [start_state]
    while pending:
        state = pending.pop()
        if state in visited:
            continue
        # Collection itself consumes budget even when all values are cached.
        budget.consume_node()
        value, best = engine.analyze(state, budget=budget)
        store[state] = (value, best)
        visited.add(state)
        me = player_at(state)
        pending.extend(
            apply_move(state, move, me) for move in reversed(search_moves(state))
        )
    return len(visited), time.monotonic() - started


def main():
    store = {}
    try:
        nodes, secs = solve_and_collect(store)
    except SearchLimitExceeded as exc:
        raise SystemExit(f"Collection stopped: {exc}; solve_data.pkl was not written.") from exc
    print(f"states={len(store)} nodes={nodes} time={secs:.1f}s")
    root_val, root_best = store[Board().pack()]
    print(f"root value(black view)={root_val}, first moves={root_best}")
    with open("solve_data.pkl", "wb") as output:
        pickle.dump(store, output)


if __name__ == "__main__":
    main()