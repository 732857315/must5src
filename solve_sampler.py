"""Collect endgame samples and label them with budgeted exact search."""

import os
import pickle
import random
import time

from exact_solver import DEFAULT_MAX_NODES, DEFAULT_TIME_LIMIT, ExactSolver
from exact_solver import SearchBudget, SearchLimitExceeded
from game import CELL_COUNT, EMPTY, NEIGHBORS, PERMS, apply_move, initial_state
from game import empty_count, legal_moves, player_at, winner


TT = {}
NODES = 0
DEFAULT_MAX_EMPTY = 6
PERM_SHIFT = [[(permutation[i] * 2, i * 2) for i in range(CELL_COUNT)]
              for permutation in PERMS]


def canon(state):
    best = state
    for pairs in PERM_SHIFT:
        transformed = 0
        for src, dst in pairs:
            transformed |= ((state >> src) & 3) << dst
        best = min(best, transformed)
    return best


def is_win(state):
    return winner(state) != EMPTY


def solve(state, *, max_nodes=DEFAULT_MAX_NODES, time_limit=DEFAULT_TIME_LIMIT):
    """Return an exact value or raise SearchLimitExceeded, without estimation."""
    global NODES
    engine = ExactSolver(TT)
    try:
        return engine.solve(state, budget=SearchBudget(max_nodes, time_limit))
    finally:
        NODES += engine.nodes


def play_random_game(rng):
    state = initial_state()
    sequence = [state]
    while not is_win(state):
        moves = legal_moves(state)
        if not moves:
            break
        state = apply_move(state, rng.choice(moves), player_at(state))
        sequence.append(state)
    return sequence


def collect_samples(rng, games, max_samples, max_empty=DEFAULT_MAX_EMPTY):
    """Sample only late positions by default; terminal positions are allowed."""
    if games < 0 or max_samples < 0:
        raise ValueError("games and max_samples must be nonnegative")
    if not 0 <= max_empty < CELL_COUNT:
        raise ValueError(f"max_empty must be between 0 and {CELL_COUNT - 1}")
    if max_samples == 0:
        return []
    seen = set()
    samples = []
    for _ in range(games):
        for state in play_random_game(rng):
            if empty_count(state) > max_empty:
                continue
            canonical = canon(state)
            if canonical in seen:
                continue
            seen.add(canonical)
            samples.append(state)
            if len(samples) >= max_samples:
                return samples
    return samples


def label_samples(samples, *, max_nodes=DEFAULT_MAX_NODES,
                  time_limit=DEFAULT_TIME_LIMIT):
    """Label a batch under one shared budget; return only after full success."""
    global NODES
    engine = ExactSolver(TT)
    budget = SearchBudget(max_nodes, time_limit)
    data = []
    try:
        for state in samples:
            value, best = engine.analyze(state, budget=budget)
            data.append((state, value, best))
    finally:
        NODES += engine.nodes
    return data


def main():
    global NODES
    rng = random.Random(12345)
    games = int(os.environ.get("GAMES", "1000"))
    max_samples = int(os.environ.get("SAMPLES", "1000"))
    max_empty = int(os.environ.get("MAX_EMPTY", str(DEFAULT_MAX_EMPTY)))
    max_nodes = int(os.environ.get("MAX_NODES", str(DEFAULT_MAX_NODES)))
    time_limit = float(os.environ.get("TIME_LIMIT", str(DEFAULT_TIME_LIMIT)))

    started = time.monotonic()
    samples = collect_samples(rng, games, max_samples, max_empty)
    print(f"collected {len(samples)} unique endgames in {time.monotonic()-started:.1f}s",
          flush=True)
    if not samples:
        raise SystemExit("No eligible samples; train_data.pkl was not written.")

    TT.clear()
    NODES = 0
    started = time.monotonic()
    try:
        data = label_samples(samples, max_nodes=max_nodes, time_limit=time_limit)
    except SearchLimitExceeded as exc:
        raise SystemExit(
            f"Labeling stopped: {exc}; train_data.pkl was not written. "
            "Use fewer samples, a smaller MAX_EMPTY, or an explicit larger budget."
        ) from exc
    print(f"labels done: tt={len(TT)} {time.monotonic()-started:.1f}s", flush=True)

    with open("train_data.pkl", "wb") as output:
        pickle.dump(data, output)
    values = [row[1] for row in data]
    print(
        f"samples={len(data)} tt={len(TT)} dist: +1={values.count(1)} "
        f"0={values.count(0)} -1={values.count(-1)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
