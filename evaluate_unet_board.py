"""Whole-board acceptance probes and paired, budgeted U-Net versus uniform-prior search.

Both players use the same >=5 Gomoku rules and the same bounded search. The AI
adds RGB U-Net window priors. A search certificate describes its current
position; it does not certify a general first-player winning strategy.
"""

import argparse
from collections import Counter
import json
from numbers import Integral
from pathlib import Path
import random
import time

import numpy as np

from board_rules import normalize_board, board_winner, legal_cells, winning_cells, apply_board_move
from board_search import select_move
from game import BLACK, WHITE, FORBIDDEN
from unet_board import analyze_board
from unet_codec import encode_rgb, FORBIDDEN_RGB
from unet_pipeline import load_model
from windows import extract_window


SEARCH_DEPTH = 3
CANDIDATE_WIDTH = 12
OPENING_PLIES = 4


def canonical_opening(board):
    """D4 on squares, only shape-preserving rotations/reflections on rectangles."""
    board = normalize_board(board)
    views = []
    for rotation in range(4):
        rotated = np.rot90(board, rotation)
        if rotated.shape == board.shape:
            views.extend((rotated.tobytes(), np.fliplr(rotated).tobytes()))
    return min(views)


def _start_board(rows, cols, prefix):
    board = np.zeros((rows, cols), dtype=np.uint8)
    center = (rows // 2, cols // 2)
    board = apply_board_move(board, center, BLACK)
    transcript = [{"side": BLACK, "move": center, "opening": True}]
    side = WHITE
    for move in prefix:
        board = apply_board_move(board, move, side)
        transcript.append({"side": side, "move": tuple(move), "opening": True})
        side = 3 - side
    return board, side, transcript


def generate_paired_openings(rows, cols, count, seed):
    if min(rows, cols) < 6 or count < 0:
        raise ValueError("boards must be at least 6x6 and pair count nonnegative")
    rng = random.Random(seed)
    seen, openings = set(), []
    center = (rows // 2, cols // 2)
    for _ in range(max(1000, count * 500)):
        if len(openings) == count:
            return openings
        board, side, _ = _start_board(rows, cols, ())
        prefix = []
        for _ in range(OPENING_PLIES):
            stones = np.argwhere((board == BLACK) | (board == WHITE))
            candidates = [
                (int(row), int(col)) for row, col in np.argwhere(board == 0)
                if max(abs(row - center[0]), abs(col - center[1])) <= 3
                and np.any(np.max(np.abs(stones - np.array((row, col))), axis=1) <= 2)
            ]
            if not candidates:
                break
            move = rng.choice(candidates)
            board = apply_board_move(board, move, side)
            prefix.append(move)
            side = 3 - side
        if len(prefix) != OPENING_PLIES or board_winner(board):
            continue
        key = canonical_opening(board)
        if key not in seen:
            seen.add(key)
            openings.append(tuple(prefix))
    raise ValueError("could not generate enough distinct near-center openings")


def _analysis_contract(board, analysis):
    legal = board == 0
    checks = {}
    for field in ("opponent_policy", "play_policy", "raw_play_policy"):
        probabilities = np.asarray(analysis[field])
        valid_shape = probabilities.shape == board.shape
        finite = bool(valid_shape and np.isfinite(probabilities).all())
        bounds = bool(finite and np.all((probabilities >= 0) & (probabilities <= 1)))
        illegal_zero = bool(valid_shape and np.all(probabilities[~legal] == 0))
        normalized = bool(finite and abs(float(probabilities[legal].sum()) - 1) <= 1e-6)
        checks[field] = {"shape": valid_shape, "finite": finite, "within_0_1": bounds,
                         "illegal_exactly_zero": illegal_zero, "normalized": normalized}
    coverage = np.asarray(analysis["coverage"])
    coverage_ok = bool(coverage.shape == board.shape and np.all(coverage[legal] > 0)
                       and np.all(coverage[~legal] == 0))
    move = analysis["move"]
    legal_move = bool(move is not None and board[tuple(move)] == 0)
    return {"passed": all(all(item.values()) for item in checks.values()) and coverage_ok and legal_move,
            "policies": checks, "all_empty_cells_covered": coverage_ok, "move_legal": legal_move,
            "window_count": analysis["window_count"], "model_batches": analysis["model_batches"]}


def _search(board, side, priors, search_seconds, max_nodes):
    started = time.monotonic()
    result = select_move(board, side, priors, max_nodes=max_nodes, time_limit=search_seconds,
                         depth=SEARCH_DEPTH, candidate_width=CANDIDATE_WIDTH)
    return result, time.monotonic() - started


def _capture_padding_probe(board, side, opponent, play):
    images = []
    def capture(module, arguments):
        images.extend(arguments[0].detach().cpu().numpy().copy())
    handle = opponent.register_forward_pre_hook(capture)
    try:
        analysis = analyze_board(board, side, opponent, play, tactical=True)
    finally:
        handle.remove()
    checked_cells = 0
    correct = True
    corners = ((0, 0), (board.shape[0] - 1, board.shape[1] - 1))
    for center in corners:
        if center not in analysis["window_centers"]:
            correct = False
            continue
        index = analysis["window_centers"].index(center)
        window = extract_window(board, *center)
        actual = images[index]
        expected = encode_rgb(window.state)
        correct &= bool(np.allclose(actual, expected, atol=1e-7, rtol=0))
        padding = np.array([coordinate is None for coordinate in window.coordinates]).reshape(5, 5)
        gray = np.array(FORBIDDEN_RGB, dtype=np.float32) / 255
        correct &= bool(np.allclose(actual[:, padding], gray[:, None], atol=1e-7, rtol=0))
        checked_cells += int(padding.sum())
    return analysis, {"passed": correct, "checked_windows": len(corners),
                      "checked_padding_cells": checked_cells,
                      "method": "observed actual opponent-model RGB inputs through a read-only forward hook"}


def compliance_probes(opponent, play, *, search_seconds=0.1, max_nodes=2000):
    """System compliance probes; tactical/search outcomes are not raw-network accuracy."""
    shapes = []
    for rows, cols in ((6, 7), (16, 16), (20, 24)):
        board = np.zeros((rows, cols), dtype=np.uint8)
        board[0, 0], board[-1, -1], board[rows // 2, cols // 2] = BLACK, WHITE, BLACK
        board[0, -1], board[-1, 0] = FORBIDDEN, FORBIDDEN
        analysis, padding = _capture_padding_probe(board, BLACK, opponent, play)
        contract = _analysis_contract(board, analysis)
        search, seconds = _search(board, BLACK, analysis["raw_play_policy"], search_seconds, max_nodes)
        search_legal = search["move"] is not None and board[tuple(search["move"])] == 0
        shapes.append({"shape": [rows, cols], "contract": contract, "corner_padding": padding,
                       "search": search, "search_wall_seconds": seconds, "search_move_legal": bool(search_legal),
                       "passed": contract["passed"] and padding["passed"] and bool(search_legal)})

    fixtures = []
    center = (7, 7)
    three = np.zeros((16, 16), dtype=np.uint8)
    three[7, [5, 6, 8]] = WHITE
    three[[5, 6, 8], 7] = WHITE
    fixtures.append(("crossed_three_shared_extension", three, center))

    shared = np.zeros((16, 16), dtype=np.uint8)
    shared[7, [5, 6, 8, 9]] = WHITE
    shared[[5, 6, 8, 9], 7] = WHITE
    fixtures.append(("crossed_four_shared_winning_cell", shared, center))

    independent = np.zeros((16, 16), dtype=np.uint8)
    independent[5, 4:8] = WHITE
    independent[10, 4:8] = WHITE
    independent[5, 3] = independent[10, 3] = FORBIDDEN
    fixtures.append(("independent_four_winning_cells", independent, None))

    own = np.zeros((16, 16), dtype=np.uint8)
    own[7, [5, 6, 8, 9]] = BLACK
    own[[5, 6, 8, 9], 7] = BLACK
    fixtures.append(("own_crossed_four_immediate_win", own, center))

    tactics = []
    for name, board, expected in fixtures:
        analysis = analyze_board(board, BLACK, opponent, play, tactical=True)
        search, seconds = _search(board, BLACK, analysis["raw_play_policy"], search_seconds, max_nodes)
        enemy = analysis["threats"]["opponent"]
        if name == "crossed_three_shared_extension":
            correct = enemy["double_three"] and not enemy["double_kill"] and search["move"] == expected
        elif name == "crossed_four_shared_winning_cell":
            correct = (enemy["double_four"] and not enemy["double_kill"]
                       and enemy["shared_completion_cells"] == [center]
                       and analysis["move"] == center and search["move"] == center)
        elif name == "independent_four_winning_cells":
            kills = winning_cells(board, WHITE)
            correct = (len(kills) >= 2 and enemy["double_kill"]
                       and search["move"] in kills and search["proven_value"] == -1)
        else:
            correct = analysis["move"] == center and search["move"] == center and search["proven_value"] == 1
        contract = _analysis_contract(board, analysis)
        tactics.append({"name": name, "passed": bool(correct and contract["passed"]),
                        "expected_move": expected, "raw_move": analysis["raw_move"],
                        "tactical_move": analysis["move"], "tactical_kind": analysis["tactical_kind"],
                        "threats": analysis["threats"], "search": search,
                        "search_wall_seconds": seconds, "contract": contract,
                        "certificate_scope": "this board with BLACK to move"})

    terminal = np.zeros((16, 16), dtype=np.uint8)
    terminal[9, 5:11] = WHITE
    terminal_analysis = analyze_board(terminal, BLACK, opponent, play)
    terminal_search, seconds = _search(terminal, BLACK, np.zeros(terminal.shape), search_seconds, max_nodes)
    terminal_pass = (terminal_analysis["terminal"] and terminal_analysis["move"] is None
                     and terminal_analysis["window_count"] == 0 and terminal_search["move"] is None
                     and all(np.count_nonzero(terminal_analysis[field]) == 0
                             for field in ("opponent_policy", "play_policy", "raw_play_policy", "coverage")))
    return {"passed": all(item["passed"] for item in shapes + tactics) and bool(terminal_pass),
            "scope": "whole-board coverage, input padding, shared rules and search behavior",
            "shapes": shapes, "combination_threats": tactics,
            "terminal": {"passed": bool(terminal_pass), "winner": terminal_analysis["winner"],
                         "window_count": terminal_analysis["window_count"], "search": terminal_search,
                         "search_wall_seconds": seconds}}


def _new_usage():
    return {"turns": 0, "nodes": 0, "search_wall_seconds": 0.0, "analysis_wall_seconds": 0.0,
            "budget_exhaustions": 0, "completed_depths": Counter()}


def _finish_usage(usage):
    return {name: {**values, "completed_depths": dict(values["completed_depths"])}
            for name, values in usage.items()}


def play_one_game(opponent, play, rows, cols, prefix, ai_side, *, search_seconds, max_nodes, max_plies):
    board, side, opening = _start_board(rows, cols, prefix)
    transcript = list(opening)
    usage = {"unet": _new_usage(), "uniform": _new_usage()}
    searched_plies = 0
    started = time.monotonic()
    while searched_plies < max_plies and not board_winner(board) and legal_cells(board):
        is_ai = side == ai_side
        player = "unet" if is_ai else "uniform"
        analysis_seconds = 0.0
        analysis = None
        if is_ai:
            analysis_start = time.monotonic()
            analysis = analyze_board(board, side, opponent, play, tactical=True)
            analysis_seconds = time.monotonic() - analysis_start
            priors = analysis["raw_play_policy"]
        else:
            priors = (board == 0).astype(np.float64)
            priors /= priors.sum()
        search, search_seconds_used = _search(board, side, priors, search_seconds, max_nodes)
        move = search["move"]
        if move is None:
            raise AssertionError("search returned no action on a live board")
        next_board = apply_board_move(board, move, side)
        entry = {"ply_after_opening": searched_plies + 1, "side": side, "player": player,
                 "move": tuple(move), "opening": False, "search": search,
                 "analysis_wall_seconds": analysis_seconds, "search_wall_seconds": search_seconds_used,
                 "certificate_scope": "position before this move, from this side's view"}
        if analysis is not None:
            entry.update(raw_move=analysis["raw_move"], tactical_move=analysis["move"],
                         tactical_kind=analysis["tactical_kind"], window_count=analysis["window_count"])
        transcript.append(entry)
        item = usage[player]
        item["turns"] += 1
        item["nodes"] += search["nodes"]
        item["analysis_wall_seconds"] += analysis_seconds
        item["search_wall_seconds"] += search_seconds_used
        item["budget_exhaustions"] += int(search["budget_exhausted"])
        item["completed_depths"][str(search["completed_depth"])] += 1
        board, side = next_board, 3 - side
        searched_plies += 1
    won = board_winner(board)
    if won:
        outcome = "win" if won == ai_side else "loss"
    elif not np.any(board == 0):
        outcome = "draw"
    else:
        outcome = "unfinished"
    return {"ai_side": ai_side, "outcome": outcome, "winner": int(won),
            "searched_plies_after_opening": searched_plies, "opening": [list(move) for move in prefix],
            "transcript": transcript, "final_board": board.tolist(), "usage": _finish_usage(usage),
            "elapsed_seconds": time.monotonic() - started}


def paired_games(opponent, play, *, rows=16, cols=16, games=4, seed=20260906,
                 search_seconds=0.1, max_nodes=2000, max_plies=80, progress=None):
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (rows, cols, games, max_nodes, max_plies)):
        raise ValueError("dimensions, games and node/ply limits must be integers")
    if min(rows, cols) < 6 or games < 0 or games % 2 or max_nodes < 0 or max_plies < 1:
        raise ValueError("dimensions >=6, games nonnegative/even, nodes >=0 and max_plies >=1 required")
    prefixes = generate_paired_openings(rows, cols, games // 2, seed)
    records = []
    totals = {"win": 0, "draw": 0, "loss": 0, "unfinished": 0}
    by_side = {str(BLACK): dict(totals), str(WHITE): dict(totals)}
    for pair_index, prefix in enumerate(prefixes, 1):
        for ai_side in (BLACK, WHITE):
            game = play_one_game(opponent, play, rows, cols, prefix, ai_side,
                                 search_seconds=search_seconds, max_nodes=max_nodes, max_plies=max_plies)
            game["pair"] = pair_index
            records.append(game)
            totals[game["outcome"]] += 1
            by_side[str(ai_side)][game["outcome"]] += 1
            if progress:
                progress(f"game={len(records)}/{games} pair={pair_index} ai_side={ai_side} "
                         f"outcome={game['outcome']} searched_plies={game['searched_plies_after_opening']}")
    return {"games": len(records), "pairs": len(prefixes), "unique_shape_compatible_D4_openings": len(prefixes),
            "outcomes": totals, "by_ai_side": by_side, "records": records,
            "opening_seed": seed, "opening_plies_after_black_center": OPENING_PLIES,
            "max_plies_scope": "searched plies after the five-stone opening",
            "termination": "only actual >=5 or a board with no empty cells ends a game; horizon is unfinished"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="completed directory containing opponent.pt and play.pt")
    parser.add_argument("--output", required=True, help="new JSON report; existing reports are preserved")
    parser.add_argument("--games", type=int, default=4, help="even game count; 0 runs probes only")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--search-seconds", type=float, default=0.1)
    parser.add_argument("--max-nodes", type=int, default=2000)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args(argv)
    if min(args.rows, args.cols) < 6 or args.games < 0 or args.games % 2 or args.max_plies < 1:
        parser.error("rows/cols must be >=6, games nonnegative/even, max-plies positive")
    if args.threads < 1 or args.max_nodes < 0 or not np.isfinite(args.search_seconds) or args.search_seconds < 0:
        parser.error("threads must be positive; node/time budgets must be nonnegative and finite")
    output = Path(args.output).resolve()
    if output.suffix.lower() != ".json" or output.exists():
        parser.error("--output must be a new .json file; weights and previous reports cannot be overwritten")
    import torch
    torch.set_num_threads(args.threads)
    directory = Path(args.models).resolve()
    opponent, opponent_meta = load_model(directory / "opponent.pt", "opponent")
    play, play_meta = load_model(directory / "play.pt", "play")
    if any(metadata.get("trained") is not True or metadata.get("stage") != 3
           for metadata in (opponent_meta, play_meta)):
        raise ValueError("completed, trained stage-3 checkpoints are required")
    started = time.monotonic()
    probes = compliance_probes(opponent, play, search_seconds=args.search_seconds, max_nodes=args.max_nodes)
    print(f"whole-board compliance probes passed={probes['passed']}", flush=True)
    games = paired_games(opponent, play, rows=args.rows, cols=args.cols, games=args.games,
                         seed=args.seed, search_seconds=args.search_seconds, max_nodes=args.max_nodes,
                         max_plies=args.max_plies, progress=lambda message: print(message, flush=True))
    report = {
        "models": {"directory": str(directory), "opponent": opponent_meta, "play": play_meta},
        "config": vars(args),
        "common_search_limits": {"depth": SEARCH_DEPTH, "candidate_width": CANDIDATE_WIDTH,
                                 "max_nodes": args.max_nodes, "tree_time_seconds": args.search_seconds},
        "interpretation": {
            "comparison": "same full-board rules and search implementation; U-Net priors versus uniform priors",
            "budget_accounting": "both players have identical caps; actual explored nodes, depth and wall times are logged separately",
            "time_scope": "tree budget excludes some validation, immediate-tactic and ordering overhead; measured search wall time includes it",
            "unfinished": "ply-limited live games are counted separately from draws",
            "proof": "proven_value certifies its particular position for the acting side, never a universal black-first guarantee",
            "opponent_model": "synthetic teacher-policy preference",
        },
        "compliance": probes, "paired_games": games, "elapsed_seconds": time.monotonic() - started,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    print(json.dumps({"report": str(output), "compliance_passed": probes["passed"],
                      "outcomes": games["outcomes"], "pairs": games["pairs"]}, ensure_ascii=False), flush=True)
    if not probes["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()