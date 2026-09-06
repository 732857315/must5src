"""Independent acceptance checks for the trained RGB U-Net pair.

The opponent model predicts the synthetic teacher's policy, not calibrated human
move frequencies. Tactical moves are used only as scoring labels, never to
replace a raw network action.
"""

import argparse
import gzip
import json
from numbers import Integral
from pathlib import Path
import time

import numpy as np
import torch

from arena import PlayerAB, evaluate as evaluate_arena, generate_openings
from game import BLACK, WHITE, EMPTY, FORBIDDEN, CELL_COUNT
from game import legal_moves, other_player, validate_state, winner, winning_moves
from train_unet import expand_actor_records, split_records
from unet_codec import encode_rgb, policy_rgb
from unet_curriculum import base_key
from unet_models import masked_probabilities
from unet_pipeline import analyze_position, load_model


def load_base_records(directory):
    """Read all three saved base datasets and verify their action-label contracts."""
    directory = Path(directory)
    records, seen = [], set()
    for stage in (1, 2, 3):
        path = directory / f"stage{stage}_base.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                state, side = record["state"], record["side"]
                validate_state(state)
                if isinstance(side, bool) or not isinstance(side, Integral) or side not in (BLACK, WHITE):
                    raise ValueError(f"{path}:{line_number}: invalid actor")
                if record.get("stage") != stage:
                    raise ValueError(f"{path}:{line_number}: stage does not match dataset")
                moves = legal_moves(state)
                if winner(state) != EMPTY or not moves:
                    raise ValueError(f"{path}:{line_number}: terminal base record has action labels")
                illegal = np.ones(CELL_COUNT, dtype=bool)
                illegal[moves] = False
                for field in ("target", "counter_target"):
                    target = np.asarray(record[field], dtype=np.float32)
                    if (target.shape != (CELL_COUNT,) or not np.isfinite(target).all()
                            or np.any(target < 0) or np.any(target > 1)
                            or not np.isclose(target.sum(), 1.0, atol=1e-5, rtol=1e-5)
                            or np.any(target[illegal] != 0)):
                        raise ValueError(f"{path}:{line_number}: invalid {field}")
                    record[field] = target
                for field in ("value", "value_kind", "outcome"):
                    if field not in record or "counter_" + field not in record:
                        raise ValueError(f"{path}:{line_number}: missing independent actor labels")
                key = base_key(state)
                if key in seen:
                    raise ValueError(f"{path}:{line_number}: duplicate D4/color-equivalent base board")
                seen.add(key)
                records.append(record)
    return records


def _predict_batches(model, records, batch_size, *, red_predictions=None):
    """Preserve raw model preferences, applying only legal-action masking."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            if red_predictions is None:
                images = [encode_rgb(record["state"]) for record in batch]
            else:
                images = [
                    policy_rgb(record["state"],
                               red_predictions[(record["state"], other_player(record["side"]))],
                               "red", levels=None)
                    for record in batch
                ]
            rgb = torch.from_numpy(np.stack(images))
            sides = torch.tensor([record["side"] for record in batch], dtype=torch.long)
            logits = model(rgb, sides)
            probabilities = masked_probabilities(logits, [record["state"] for record in batch])
            result.append(probabilities[:, 0].reshape(-1, CELL_COUNT).cpu().numpy())
    return np.concatenate(result) if result else np.zeros((0, CELL_COUNT), dtype=np.float32)


def probability_checks(probabilities, records, tolerance=1e-6):
    """Audit legal mass, exact illegal zeros, and all-zero terminal policies."""
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(records), CELL_COUNT):
        raise ValueError("one 25-cell probability vector is required per record")
    finite = bool(np.isfinite(probabilities).all())
    negative = int(np.count_nonzero(probabilities < 0))
    illegal_nonzero = terminal_nonzero = mass_failures = 0
    live = terminal = 0
    max_mass_error = 0.0
    for row, record in zip(probabilities, records):
        state = record["state"]
        moves = legal_moves(state)
        if winner(state) != EMPTY or not moves:
            terminal += 1
            terminal_nonzero += int(np.count_nonzero(row))
            continue
        live += 1
        allowed = np.zeros(CELL_COUNT, dtype=bool)
        allowed[moves] = True
        illegal_nonzero += int(np.count_nonzero(row[~allowed]))
        error = abs(float(row[allowed].sum()) - 1.0)
        if not np.isfinite(error) or error > tolerance:
            mass_failures += 1
        if np.isfinite(error):
            max_mass_error = max(max_mass_error, error)
    passed = finite and not any((negative, illegal_nonzero, terminal_nonzero, mass_failures))
    return {
        "passed": passed, "positions": len(records), "live_positions": live,
        "terminal_positions": terminal, "finite": finite, "negative_entries": negative,
        "illegal_nonzero_entries": illegal_nonzero, "terminal_nonzero_entries": terminal_nonzero,
        "normalization_failures": mass_failures, "max_legal_mass_error": max_mass_error,
        "tolerance": tolerance,
    }


def _empty_metrics():
    return {name: {"hits": 0, "count": 0} for name in ("teacher_top1", "immediate_win", "unique_defense")}


def tactical_metrics(probabilities, records):
    """Score unmodified greedy choices against real one-move wins/unique blocks."""
    total, by_stage, by_side = _empty_metrics(), {}, {}
    for row, record in zip(probabilities, records):
        state, side = record["state"], record["side"]
        if winner(state) != EMPTY or not legal_moves(state):
            continue
        choice = int(np.argmax(row))
        target = np.asarray(record["target"])
        own_wins = winning_moves(state, side)
        opponent_wins = winning_moves(state, other_player(side))
        outcomes = {"teacher_top1": bool(target[choice] >= target.max() - 1e-6)}
        if own_wins:
            outcomes["immediate_win"] = choice in own_wins
        elif len(opponent_wins) == 1:
            outcomes["unique_defense"] = choice == opponent_wins[0]
        groups = (total, by_stage.setdefault(str(record["stage"]), _empty_metrics()),
                  by_side.setdefault(str(side), _empty_metrics()))
        for group in groups:
            for name, hit in outcomes.items():
                group[name]["count"] += 1
                group[name]["hits"] += int(hit)
    for group in (total, *by_stage.values(), *by_side.values()):
        for item in group.values():
            item["rate"] = item["hits"] / item["count"] if item["count"] else None
    return {"all": total, "by_stage": by_stage, "by_side": by_side,
            "unique_defense_definition": "exactly one opponent winning cell and no own immediate win"}


def _prediction_lookup(records, predictions):
    lookup = {}
    for record, prediction in zip(records, predictions):
        key = (record["state"], record["side"])
        if key in lookup:
            raise ValueError("duplicate actor record in acceptance set")
        lookup[key] = prediction
    return lookup


def terminal_probes():
    """Both colors act on wins, a draw, and a fully forbidden board."""
    draw_cells = "OXXXX" "XOOXO" "XOXXO" "OOXOO" "XXXOO"
    draw = sum({"X": BLACK, "O": WHITE}[cell] << (2 * i) for i, cell in enumerate(draw_cells))
    states = [
        sum(BLACK << (2 * i) for i in range(5)),
        sum(WHITE << (2 * i) for i in (0, 5, 10, 15, 20)),
        sum(BLACK << (2 * i) for i in range(5)) | (FORBIDDEN << 48),
        draw,
        (1 << 50) - 1,
    ]
    return [{"state": state, "side": side} for state in states for side in (BLACK, WHITE)]


def evaluate_models(opponent_model, play_model, records, batch_size=128):
    """Evaluate both roles on held-out actor labels without changing their actions."""
    if not records:
        raise ValueError("acceptance set must not be empty")
    opponent = _predict_batches(opponent_model, records, batch_size)
    red = _prediction_lookup(records, opponent)
    play = _predict_batches(play_model, records, batch_size, red_predictions=red)
    probes = terminal_probes()
    terminal_opponent = _predict_batches(opponent_model, probes, batch_size)
    terminal_play = _predict_batches(
        play_model, probes, batch_size,
        red_predictions=_prediction_lookup(probes, terminal_opponent),
    )
    report = {}
    for name, probabilities, terminal in (
        ("opponent", opponent, terminal_opponent), ("play", play, terminal_play),
    ):
        checks = probability_checks(probabilities, records)
        report[name] = {
            "probabilities": checks,
            "probabilities_by_stage": {
                str(stage): probability_checks(
                    probabilities[[i for i, item in enumerate(records) if item["stage"] == stage]],
                    [item for item in records if item["stage"] == stage],
                ) for stage in sorted({item["stage"] for item in records})
            },
            "terminal_mask": probability_checks(terminal, probes),
            "tactics": tactical_metrics(probabilities, records),
        }
    report["passed_probability_contracts"] = all(
        report[role]["probabilities"]["passed"] and report[role]["terminal_mask"]["passed"]
        for role in ("opponent", "play")
    )
    return report


class RawUNetPlayer:
    name = "raw_play_unet_with_predicted_red"
    tactical_assistance = False

    def __init__(self, opponent_model, play_model):
        self.opponent_model = opponent_model
        self.play_model = play_model

    def move(self, state, side):
        return analyze_position(state, side, self.opponent_model, self.play_model)["move"]


class DepthTwoNoCNN(PlayerAB):
    def __init__(self):
        super().__init__(None, "depth2_alpha_beta_no_cnn", depth=2, hint=False)

    def _cnn_logits(self, state, side):
        return np.zeros(CELL_COUNT, dtype=np.float64)


def paired_match(opponent_model, play_model, games=24, seed=20260906, opening_plies=4):
    if isinstance(games, bool) or not isinstance(games, Integral) or games < 0 or games % 2:
        raise ValueError("games must be zero or a positive even integer")
    if games == 0:
        return {"enabled": False, "games": 0, "pairs": 0}
    openings = generate_openings(games // 2, seed=seed, plies=opening_plies)
    player = RawUNetPlayer(opponent_model, play_model)
    baseline = DepthTwoNoCNN()
    stats = evaluate_arena(player, baseline, openings)
    return {
        "enabled": True, "games": stats["games"], "pairs": stats["pairs"],
        "win": stats["W"], "draw": stats["D"], "loss": stats["L"],
        "wins_as_black": stats["Wb"], "wins_as_white": stats["Ww"],
        "unique_openings": stats["unique_openings"], "unique_transcripts": stats["unique_transcripts"],
        "seed": seed, "opening_plies_after_black_center": opening_plies,
        "openings": [list(prefix) for prefix in openings],
        "network_tactical_assistance": False, "network_search": False,
        "baseline": "PlayerAB depth 2, no CNN or learned priors",
        "board_scope": "plain 5x5; identical opening reused with players exchanging colors",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="completed training directory with weights and 3 base datasets")
    parser.add_argument("--output", required=True, help="new JSON acceptance report")
    parser.add_argument("--games", type=int, default=24, help="even number of paired games; 0 skips arena")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.games < 0 or args.games % 2 or args.batch_size < 1 or args.threads < 1:
        parser.error("games must be nonnegative/even; batch size and threads must be positive")
    output = Path(args.output).resolve()
    if output.suffix.lower() != ".json":
        parser.error("--output must be a JSON file, never a model weight path")
    if output.exists():
        parser.error("--output already exists; use a new report path")
    directory = Path(args.models).resolve()
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    opponent, opponent_meta = load_model(directory / "opponent.pt", "opponent")
    play, play_meta = load_model(directory / "play.pt", "play")
    if any(metadata.get("trained") is not True or metadata.get("stage") != 3
           for metadata in (opponent_meta, play_meta)):
        raise ValueError("acceptance requires completed, trained stage-3 checkpoints")
    bases = load_base_records(directory)
    training_base, held_out_base = split_records(bases)
    if {base_key(item["state"]) for item in training_base} & {base_key(item["state"]) for item in held_out_base}:
        raise AssertionError("training and validation base groups overlap")
    records = expand_actor_records(held_out_base)
    report = {
        "models": {"directory": str(directory), "opponent": opponent_meta, "play": play_meta},
        "data": {"base_samples": len(bases), "held_out_base_samples": len(held_out_base),
                 "held_out_actor_labels": len(records), "canonical_base_unique": True,
                 "split": "same stable base_key split as training, then both actors expanded",
                 "evaluation_set": "validation used for model selection; not a new untouched test set"},
        "interpretation": {
            "opponent": "synthetic teacher-policy prediction, not calibrated human move frequencies",
            "tactics": "raw legal greedy action scored against true immediate wins/unique defenses",
            "arena": "paired observed results on a small plain-board opening suite, not a proof of strength",
        },
        "acceptance": evaluate_models(opponent, play, records, args.batch_size),
    }
    print(f"Policy checks complete: {len(held_out_base)} held-out boards / {len(records)} actor labels", flush=True)
    report["paired_arena"] = paired_match(opponent, play, args.games, args.seed, args.opening_plies)
    report["elapsed_seconds"] = time.monotonic() - started
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    print(json.dumps({"report": str(output),
                      "passed_probability_contracts": report["acceptance"]["passed_probability_contracts"],
                      "paired_arena": report["paired_arena"]}, ensure_ascii=False), flush=True)
    if not report["acceptance"]["passed_probability_contracts"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()