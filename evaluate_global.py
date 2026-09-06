"""Paired global+local+value versus local-only whole-board search evaluation.

Both modes use the same search implementation and budget. Neural value estimates
never end a game or act as proof. Live games at the ply limit remain truncated.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import time

import numpy as np

from board_rules import normalize_board, board_winner, apply_board_move, winning_cells
from board_search import select_move
from board_threats import analyze_threats
from evaluate_unet_board import generate_paired_openings, _start_board, canonical_opening, OPENING_PLIES
from global_inference import analyze_global, evaluate_global_value, load_global_model
from unet_board import analyze_board
from unet_pipeline import load_model

SEARCH_DEPTH = 5
CANDIDATE_WIDTH = 12
MODES = ("global", "local")


def policy_contract(board, analysis):
    board = normalize_board(board)
    legal = board == 0
    terminal = bool(board_winner(board) or not legal.any())
    fields = ["opponent_policy", "play_policy", "raw_play_policy"]
    fields += [field for field in ("global_policy", "combined_policy") if field in analysis]
    checks = {}
    for name in fields:
        p = np.asarray(analysis[name])
        shape = p.shape == board.shape
        finite = bool(shape and np.isfinite(p).all())
        checks[name] = {
            "shape": shape, "finite": finite,
            "within_0_1": bool(finite and ((p >= 0) & (p <= 1)).all()),
            "illegal_exactly_zero": bool(shape and (p[~legal] == 0).all()),
            "normalized_or_terminal_zero": bool(finite and
                (np.count_nonzero(p) == 0 if terminal else abs(float(p.sum()) - 1) < 1e-6)),
        }
    coverage = np.asarray(analysis["coverage"])
    covered = bool(coverage.shape == board.shape and (coverage[~legal] == 0).all()
                   and ((coverage == 0).all() if terminal else (coverage[legal] > 0).all()))
    value = analysis.get("value")
    value_ok = (value is None) if terminal or "global_policy" not in analysis else (
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and -1 <= value <= 1)
    return {"passed": all(all(check.values()) for check in checks.values()) and covered and value_ok,
            "policies": checks, "coverage": covered, "value_is_valid_estimate_or_absent": value_ok,
            "terminal": terminal}


def policy_summary(policy, board, move=None):
    p = np.asarray(policy, dtype=np.float64)
    legal = np.flatnonzero(np.asarray(board).reshape(-1) == 0)
    order = sorted(legal, key=lambda index: (-p.reshape(-1)[index], int(index)))
    positive = p[p > 0]
    result = {
        "sum": float(p.sum()), "legal_cells": len(legal),
        "entropy_nats": float(-(positive * np.log(positive)).sum()),
        "top5": [{"move": list(map(int, np.unravel_index(index, p.shape))),
                  "probability": float(p.reshape(-1)[index])} for index in order[:5]],
    }
    if move is not None:
        chosen = float(p[tuple(move)])
        result.update(chosen_probability=chosen,
                      chosen_rank=1 + int(np.count_nonzero(p.reshape(-1)[legal] > chosen)),
                      chosen_tie_count=int(np.count_nonzero(p.reshape(-1)[legal] == chosen)))
    return result


def _analysis(opponent, play, global_model, board, side, mode):
    started = time.monotonic()
    result = analyze_board(board, side, opponent, play, tactical=True)
    if mode == "global":
        result.update(analyze_global(board, side, result, global_model))
    contract = policy_contract(board, result)
    if not contract["passed"]:
        raise AssertionError(f"{mode} analysis violated the full-board contract: {contract}")
    return result, time.monotonic() - started, contract


def _search(board, side, analysis, mode, global_model, search_seconds, max_nodes, value_weight):
    options = {}
    if mode == "global":
        options = {"value_evaluator": lambda position, actor: evaluate_global_value(position, actor, global_model),
                   "value_weight": value_weight}
    priors = analysis["combined_policy" if mode == "global" else "raw_play_policy"]
    started = time.monotonic()
    result = select_move(board, side, priors, depth=SEARCH_DEPTH, candidate_width=CANDIDATE_WIDTH,
                         time_limit=search_seconds, max_nodes=max_nodes, **options)
    return result, time.monotonic() - started


def _threat_evidence(board, side):
    return {"self": analyze_threats(board, side), "opponent": analyze_threats(board, 3 - side)}


def probe_positions():
    fixtures = []
    own = np.zeros((16, 16), dtype=np.uint8)
    own[7, [5, 6, 8, 9]] = 1
    fixtures.append(("immediate_win", own, 1, [(7, 7)], 1))
    block = own.copy()
    block[block == 1] = 2
    fixtures.append(("unique_block", block, 1, [(7, 7)], None))
    shared = block.copy()
    shared[[5, 6, 8, 9], 7] = 2
    fixtures.append(("overlapping_four_shared_defense", shared, 1, [(7, 7)], None))
    kills = np.zeros((16, 16), dtype=np.uint8)
    kills[3, 5:9] = kills[12, 5:9] = 2
    kills[3, 4] = kills[12, 4] = 3
    fixtures.append(("cross_window_independent_enemy_kills", kills, 1, [(3, 9), (12, 9)], -1))
    attack = kills.copy()
    attack[attack == 2] = 1
    fixtures.append(("cross_window_independent_own_kills", attack, 1, [(3, 9), (12, 9)], 1))
    terminal = np.zeros((16, 16), dtype=np.uint8)
    terminal[8, 4:10] = 2
    fixtures.append(("terminal_six_in_row", terminal, 1, [], -1))
    return fixtures


def compliance_probes(opponent, play, global_model, *, search_seconds=1.0, max_nodes=2000, value_weight=40.0):
    records = []
    for shape in ((6, 7), (16, 16), (20, 24)):
        board = np.zeros(shape, dtype=np.uint8)
        board[0, 0], board[-1, -1], board[shape[0] // 2, shape[1] // 2] = 1, 2, 1
        board[0, -1], board[-1, 0] = 3, 3
        for mode in MODES:
            analysis, seconds, contract = _analysis(opponent, play, global_model, board, 1, mode)
            search, search_wall = _search(board, 1, analysis, mode, global_model, search_seconds, max_nodes, value_weight)
            legal = search["move"] is not None and board[tuple(search["move"])] == 0
            records.append({"name": "full_board_contract", "shape": list(shape), "mode": mode,
                            "passed": bool(contract["passed"] and legal), "board": board.tolist(),
                            "contract": contract, "search": search, "global_value": analysis.get("value"),
                            "analysis_wall_seconds": seconds, "search_wall_seconds": search_wall})
    for name, board, side, expected_moves, expected_proof in probe_positions():
        for mode in MODES:
            analysis, seconds, contract = _analysis(opponent, play, global_model, board, side, mode)
            search, search_wall = _search(board, side, analysis, mode, global_model, search_seconds, max_nodes, value_weight)
            expected = [tuple(move) for move in expected_moves]
            correct = search["move"] in expected if expected else search["move"] is None
            if expected_proof is not None:
                correct = correct and search["proven_value"] == expected_proof
            evidence = analysis["threats"]
            if name == "overlapping_four_shared_defense":
                correct = correct and evidence["opponent"]["double_four"] and not evidence["opponent"]["double_kill"]
            if "independent" in name:
                target = "opponent" if "enemy" in name else "self"
                correct = correct and evidence[target]["double_kill"] and set(
                    map(tuple, evidence[target]["winning_cells"])) == set(expected)
            summaries = {field: policy_summary(analysis[field], board, search["move"])
                         for field in ("raw_play_policy", "global_policy", "combined_policy") if field in analysis}
            records.append({"name": name, "mode": mode, "side": side, "board": board.tolist(),
                            "passed": bool(correct and contract["passed"]), "expected_moves": expected_moves,
                            "expected_search_proven_value": expected_proof, "contract": contract,
                            "search": search, "priors": summaries, "threats": evidence,
                            "global_value": analysis.get("value"), "value_source": analysis.get("value_source", "not_used"),
                            "analysis_wall_seconds": seconds, "search_wall_seconds": search_wall})
    return {"passed": all(record["passed"] for record in records), "records": records,
            "scope": "system rules, legal policies and shared full-board tactics; not raw neural prediction accuracy",
            "value_scope": "global_value is an estimate; only search.proven_value can report a position-specific certificate"}


def _usage():
    return {"moves": 0, "nodes": 0, "value_evaluations": 0, "budget_exhaustions": 0,
            "analysis_wall_seconds": 0.0, "search_wall_seconds": 0.0, "completed_depths": Counter()}


def play_one_game(opponent, play, global_model, rows, cols, prefix, global_side, *,
                  search_seconds=1.0, max_nodes=2000, max_plies=96, value_weight=40.0, progress=None):
    board, side, opening = _start_board(rows, cols, prefix)
    initial = board.copy()
    transcript, usage = list(opening), {mode: _usage() for mode in MODES}
    started = time.monotonic()
    searched = 0
    while searched < max_plies and not board_winner(board) and np.any(board == 0):
        mode = "global" if side == global_side else "local"
        analysis, analysis_seconds, contract = _analysis(opponent, play, global_model, board, side, mode)
        search, search_wall = _search(board, side, analysis, mode, global_model, search_seconds, max_nodes, value_weight)
        move = search["move"]
        if move is None:
            raise AssertionError("search returned no move on a live board")
        after = apply_board_move(board, move, side)
        priors = {field: policy_summary(analysis[field], board, move)
                  for field in ("opponent_policy", "play_policy", "raw_play_policy", "global_policy", "combined_policy")
                  if field in analysis}
        transcript.append({
            "opening": False, "ply_after_opening": searched + 1, "side": side, "mode": mode,
            "move": tuple(move), "board_before": board.tolist(),
            "board_sha256": hashlib.sha256(board.tobytes()).hexdigest(),
            "prior_used": "combined_policy" if mode == "global" else "raw_play_policy",
            "priors": priors, "global_value": analysis.get("value"),
            "value_source": analysis.get("value_source", "not_used"),
            "search": search, "certificate_scope": "board_before with this side to move",
            "threats_before": analysis["threats"], "threats_after": _threat_evidence(after, side),
            "winner_after": int(board_winner(after)), "contract": contract,
            "window_count": int(analysis["window_count"]),
            "analysis_wall_seconds": analysis_seconds, "search_wall_seconds": search_wall,
        })
        used = usage[mode]
        used["moves"] += 1
        used["nodes"] += int(search["nodes"])
        used["value_evaluations"] += int(search.get("value_evaluations", 0))
        used["budget_exhaustions"] += int(search["budget_exhausted"])
        used["analysis_wall_seconds"] += analysis_seconds
        used["search_wall_seconds"] += search_wall
        used["completed_depths"][str(search["completed_depth"])] += 1
        board, side, searched = after, 3 - side, searched + 1
        if progress and searched % 8 == 0:
            progress(f"ply={searched}/{max_plies} last_mode={mode} move={tuple(move)} nodes={search['nodes']}")
    winner = int(board_winner(board))
    terminal = bool(winner or not np.any(board == 0))
    outcome = ("win" if winner == global_side else "loss") if winner else "draw" if terminal else "truncated"
    stop_reason = "actual_five_or_more" if winner else "board_full" if terminal else "ply_limit"
    return {"global_side": global_side, "outcome_for_global": outcome, "winner": winner, "terminal": terminal,
            "truncated": not terminal, "stop_reason": stop_reason, "next_side": 0 if terminal else side,
            "searched_plies_after_opening": searched, "initial_board": initial.tolist(),
            "opening_prefix": [list(move) for move in prefix], "transcript": transcript, "final_board": board.tolist(),
            "usage": {mode: {**data, "completed_depths": dict(data["completed_depths"])} for mode, data in usage.items()},
            "elapsed_seconds": time.monotonic() - started}


def paired_games(opponent, play, global_model, *, rows=16, cols=16, games=4, seed=20260906,
                 search_seconds=1.0, max_nodes=2000, max_plies=96, value_weight=40.0, progress=None):
    for value in (rows, cols, games, max_nodes, max_plies):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("dimensions and game/node/ply counts must be integers")
    if min(rows, cols) < 6 or max(rows, cols) > 128 or games < 0 or games % 2 or max_nodes < 0 or max_plies < 1:
        raise ValueError("dimensions 6..128, even nonnegative games, nonnegative nodes and positive max_plies required")
    if any(isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in (search_seconds, value_weight)):
        raise ValueError("search_seconds and value_weight must be finite and nonnegative")
    prefixes = generate_paired_openings(rows, cols, games // 2, seed)
    records = []
    totals = {key: 0 for key in ("win", "loss", "draw", "truncated")}
    by_side = {str(side): dict(totals) for side in (1, 2)}
    for pair, prefix in enumerate(prefixes, 1):
        for global_side in (1, 2):
            record = play_one_game(opponent, play, global_model, rows, cols, prefix, global_side,
                                   search_seconds=search_seconds, max_nodes=max_nodes,
                                   max_plies=max_plies, value_weight=value_weight,
                                   progress=(lambda line: progress(f"pair={pair} global_side={global_side} {line}")) if progress else None)
            record["pair"] = pair
            records.append(record)
            totals[record["outcome_for_global"]] += 1
            by_side[str(global_side)][record["outcome_for_global"]] += 1
            if progress:
                progress(f"game={len(records)}/{games} pair={pair} global_side={global_side} "
                         f"outcome={record['outcome_for_global']} searched_plies={record['searched_plies_after_opening']}")
    return {"games": len(records), "pairs": len(prefixes), "outcomes_for_global": totals,
            "by_global_side": by_side, "terminal_games": sum(record["terminal"] for record in records),
            "truncated_games": totals["truncated"], "records": records, "opening_seed": seed,
            "opening_plies_after_black_center": OPENING_PLIES,
            "unique_shape_preserving_D4_openings": len(prefixes),
            "color_assignment": "identical colored opening in each pair; global mode plays black then white",
            "max_plies_scope": "searched plies after black center plus four alternating opening moves"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="trained local opponent.pt/play.pt directory")
    parser.add_argument("--strategy", required=True, help="trained global checkpoint")
    parser.add_argument("--output", required=True, help="new JSON report; no overwrite")
    parser.add_argument("--games", type=int, default=4, help="even count; zero runs probes only")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--max-plies", type=int, default=96)
    parser.add_argument("--max-nodes", type=int, default=2000)
    parser.add_argument("--search-seconds", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=40.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args(argv)
    if (not 6 <= min(args.rows, args.cols) <= max(args.rows, args.cols) <= 128 or args.games < 0 or args.games % 2
            or args.max_plies < 1 or args.max_nodes < 0 or args.threads < 1
            or any(not math.isfinite(value) or value < 0 for value in (args.search_seconds, args.value_weight))):
        parser.error("invalid board dimensions, paired game count, or resource budgets")
    output = Path(args.output).resolve()
    if output.suffix.lower() != ".json" or output.exists():
        parser.error("--output must name a new .json report")
    import torch
    torch.set_num_threads(args.threads)
    directory, strategy = Path(args.models).resolve(), Path(args.strategy).resolve()
    opponent, opponent_meta = load_model(directory / "opponent.pt", "opponent")
    play, play_meta = load_model(directory / "play.pt", "play")
    global_model, global_meta = load_global_model(strategy)
    if any(meta.get("trained") is not True for meta in (opponent_meta, play_meta, global_meta)):
        raise ValueError("all three checkpoints must be explicitly marked trained")
    if any(meta.get("stage") != 3 for meta in (opponent_meta, play_meta)):
        raise ValueError("local checkpoints must have completed curriculum stage 3")
    started = time.monotonic()
    probes = compliance_probes(opponent, play, global_model, search_seconds=args.search_seconds,
                               max_nodes=args.max_nodes, value_weight=args.value_weight)
    print(f"global comparison probes passed={probes['passed']}", flush=True)
    games = paired_games(opponent, play, global_model, rows=args.rows, cols=args.cols,
                         games=args.games if probes["passed"] else 0,
                         seed=args.seed, search_seconds=args.search_seconds, max_nodes=args.max_nodes,
                         max_plies=args.max_plies, value_weight=args.value_weight,
                         progress=lambda line: print(line, flush=True))
    games["skipped_due_to_failed_probes"] = not probes["passed"]
    paths = {"opponent": directory / "opponent.pt", "play": directory / "play.pt", "global": strategy}
    report = {"format": "gomoku_global_evaluation_v1", "config": vars(args),
              "models": {role: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "metadata": meta} for (role, path), meta in
                         zip(paths.items(), (opponent_meta, play_meta, global_meta))},
              "common_search_limits": {"depth": SEARCH_DEPTH, "candidate_width": CANDIDATE_WIDTH,
                                       "max_nodes": args.max_nodes, "tree_time_seconds": args.search_seconds},
              "interpretation": {
                  "comparison": "same local models and new search: global policy+local policy+global leaf value versus local raw policy only",
                  "global_prior": "0.7 global + 0.3 local raw; continuous legal probabilities",
                  "budget": "identical tree caps; neural leaf costs consume global tree time; pre-search model/validation/tactic time is additional",
                  "neural_value": "actor-side estimate, never a proof, adjudication, or win probability",
                  "outcome": "only actual five-or-more or full board ends a game; live ply-limit cases stay truncated",
                  "sample_scope": "finite paired opening sample; not an independent test-set accuracy or universal strength guarantee"},
              "compliance": probes, "paired_games": games, "elapsed_seconds": time.monotonic() - started}
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    print(json.dumps({"report": str(output), "probes_passed": probes["passed"],
                      "outcomes_for_global": games["outcomes_for_global"]}, ensure_ascii=False), flush=True)
    return 0 if probes["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
