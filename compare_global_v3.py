"""Compare completed v2/v3 checkpoints on identical audited v3 validation boards."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from global_inference import load_global_model
from global_model import encode_global, masked_global_policy
from train_global import batches, policy_metrics


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "training_runs/global_v3"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_stats(predictions, rows, target_name="value", valid_name="value_valid"):
    selected = [(float(prediction), float(row[target_name]), row)
                for prediction, row in zip(predictions, rows)
                if row.get(valid_name, False)]
    if not selected:
        return dict(total=0, mse=None, zero_baseline_mse=None)
    prediction = np.array([item[0] for item in selected])
    target = np.array([item[1] for item in selected])
    classification = np.where(prediction > .25, 1, np.where(prediction < -.25, -1, 0))
    groups = defaultdict(list)
    for guess, expected, row in selected:
        groups[row["game_id"]].append((guess - expected) ** 2)
    by_target = {}
    for outcome in (-1, 0, 1):
        mask = target == outcome
        by_target[str(outcome)] = dict(total=int(mask.sum()),
                                      mean_prediction=float(prediction[mask].mean()) if mask.any() else None,
                                      prediction_counts={str(value): int(np.count_nonzero(classification[mask] == value))
                                                         for value in (-1, 0, 1)})
    return dict(total=len(selected), groups=len(groups), mse=float(np.mean((prediction - target) ** 2)),
                zero_baseline_mse=float(np.mean(target ** 2)), mae=float(np.mean(np.abs(prediction - target))),
                group_balanced_mse=float(np.mean([np.mean(losses) for losses in groups.values()])),
                outcome_accuracy=float(np.mean(classification == target)),
                sign_accuracy=float(np.mean(np.sign(prediction) == target)), mean_prediction=float(prediction.mean()),
                confident_wrong=int(np.count_nonzero((prediction * target < 0) & (np.abs(prediction) >= .75))),
                saturated_predictions=int(np.count_nonzero(np.abs(prediction) >= .95)), by_target=by_target)


def predict(model, rows, zero_auxiliary):
    probabilities, values, ordered = [], [], []
    model.eval()
    with torch.inference_mode():
        for (inputs, boards, _, _, _), group in batches(rows, 64, auxiliary_dropout=float(zero_auxiliary),
                                                       input_mode=model.input_mode):
            logits, result = model(inputs)
            probabilities.extend(masked_global_policy(logits, boards)[:, 0].numpy())
            values.extend(result.numpy())
            ordered.extend(group)
    return probabilities, np.asarray(values), ordered


def summarize(probabilities, values, rows, zero_auxiliary):
    combined = probabilities if zero_auxiliary else [
        .7 * p + .3 * np.asarray(row["local_policy"]) for p, row in zip(probabilities, rows)]
    def bucket(indices):
        selected = [rows[i] for i in indices]
        prediction = values[indices]
        proof_rows = [dict(row, value=row.get("search_proven_value"), value_valid=row.get("search_proven_value") is not None)
                      for row in selected]
        unproven_terminal = [dict(row, terminal_value_valid=row.get("terminal_value_valid", False)
                                 and row.get("search_proven_value") is None) for row in selected]
        informative = [i for i in indices if rows[i]["policy_source"] != "all_legal_proved_loss"]
        return dict(records=len(selected), groups=len({row["game_id"] for row in selected}),
                    policy=policy_metrics([probabilities[i] for i in indices], selected),
                    combined_policy=policy_metrics([combined[i] for i in indices], selected),
                    informative_policy=policy_metrics([probabilities[i] for i in informative], [rows[i] for i in informative]),
                    training_objective_value=value_stats(prediction, selected),
                    proof_value=value_stats(prediction, proof_rows),
                    played_terminal_value=value_stats(prediction, selected, "terminal_value", "terminal_value_valid"),
                    unproven_played_terminal_value=value_stats(prediction, unproven_terminal, "terminal_value", "terminal_value_valid"))
    answer = {source: bucket([i for i, row in enumerate(rows) if row["source"] == source])
              for source in sorted({row["source"] for row in rows})}
    opening = [i for i, row in enumerate(rows) if row["source"] == "whole_board_search" and row.get("ply", 100) <= 8]
    answer["real_opening_through_ply8"] = bucket(opening)
    answer["all"] = bucket(list(range(len(rows))))
    return answer


def opening_probes(model):
    empty = np.zeros((16, 16), dtype=np.uint8)
    center = empty.copy()
    center[8, 8] = 1
    early = center.copy()
    early[7, 8] = 2
    fixtures = [("empty_black_to_move", empty, 1), ("black_center_white_to_move", center, 2),
                ("two_stones_black_to_move", early, 1)]
    result = {}
    with torch.inference_mode():
        for name, board, side in fixtures:
            zero = np.zeros(board.shape, dtype=np.float32)
            inputs = torch.from_numpy(encode_global(board, side, zero, zero, input_mode=model.input_mode)[None])
            _, value = model(inputs)
            result[name] = float(value[0])
    return result


def paired_game_delta(old_predictions, new_predictions, rows):
    losses = defaultdict(list)
    for old, new, row in zip(old_predictions, new_predictions, rows):
        if row["source"] == "whole_board_search" and row.get("terminal_value_valid", False):
            target = row["terminal_value"]
            losses[row["game_id"]].append((new - target) ** 2 - (old - target) ** 2)
    totals = np.array([np.sum(values) for values in losses.values()])
    counts = np.array([len(values) for values in losses.values()])
    rng = np.random.default_rng(20260908)
    indices = rng.integers(0, len(totals), size=(2000, len(totals)))
    samples = totals[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return dict(groups=len(totals), records=int(counts.sum()), delta_v3_minus_v2=float(totals.sum() / counts.sum()),
                paired_game_bootstrap_95_percent_interval=list(map(float, np.quantile(samples, [.025, .975]))),
                bootstrap_replicates=2000, interpretation="Negative delta improves played-continuation MSE; not a minimax guarantee.")


def main():
    summary_path = RUN / "summary.json"
    if not summary_path.exists():
        raise RuntimeError("Wait for global_v3 summary.json; active training is not evaluated")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["trained"] and summary["epochs_completed"] == 100
    audit = json.loads((RUN / "data_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] and audit["cache_audited"] and sha(RUN / "dataset.jsonl.gz") == audit["dataset_sha256"]
    split = json.loads((RUN / "split.json").read_text(encoding="utf-8"))
    selected = set(split["validation_groups"])
    with gzip.open(RUN / "dataset.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [row for line in handle if (row := json.loads(line))["game_id"] in selected]
    real = [row for row in rows if row["source"] == "whole_board_search"]
    assert len(real) == 1152 and len({row["game_id"] for row in real}) == 28
    torch.set_num_threads(4)
    paths = {name: ROOT / "training_runs" / ("global_" + name) / "global.pt" for name in ("v2", "v3")}
    hashes = {name: sha(path) for name, path in paths.items()}
    frozen_hashes = audit["inputs"]["frozen_local_model_sha256"]
    assert all(sha(path) == digest for path, digest in frozen_hashes.items())
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), models={}, validation_records=len(rows),
                  real_validation_records=len(real), real_validation_groups=28, v3_best_epoch=summary["best_epoch"],
                  same_v2_real_validation=True, comparison="Identical boards, actors, policy/terminal/proof labels and frozen local features.")
    predictions = {}
    ordered_keys = None
    for name, path in paths.items():
        model, payload = load_global_model(path)
        model_results = dict(checkpoint_sha256=hashes[name], best_epoch=payload["best_epoch"], input_mode=model.input_mode)
        for zero_auxiliary in (False, True):
            mode = "zero_auxiliary" if zero_auxiliary else "with_auxiliary"
            probabilities, values, ordered = predict(model, rows, zero_auxiliary)
            keys = [row["board_key"] for row in ordered]
            if ordered_keys is None:
                ordered_keys = keys
            assert keys == ordered_keys
            model_results[mode] = summarize(probabilities, values, ordered, zero_auxiliary)
            if zero_auxiliary:
                predictions[name] = values
        model_results["opening_zero_auxiliary_estimates"] = opening_probes(model)
        result["models"][name] = model_results
    result["paired_zero_auxiliary_continuation_delta"] = paired_game_delta(predictions["v2"], predictions["v3"], ordered)
    assert all(sha(path) == hashes[name] for name, path in paths.items())
    assert all(sha(path) == digest for path, digest in frozen_hashes.items())
    result["weights_unchanged"] = True
    result["limitations"] = [
        "Played terminal outcomes label the actual bounded teacher continuation, not guaranteed minimax values.",
        "Only one of four strategic source-game groups is held out; background variants are not independent source games.",
        "All-legal loss policies are excluded from informative policy accuracy.",
        "Opening probes have no independently proved value target and are estimates only.",
        "Held-out metrics and checkpoint selection do not establish a match win rate."]
    (RUN / "checkpoint_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = dict(v3_best_epoch=result["v3_best_epoch"], real_validation_records=1152, models={},
                   paired_zero_auxiliary_continuation_delta=result["paired_zero_auxiliary_continuation_delta"])
    for name, row in result["models"].items():
        compact["models"][name] = dict(best_epoch=row["best_epoch"],
                    normal_real_policy_top1=row["with_auxiliary"]["whole_board_search"]["policy"]["top1"],
                    zero_aux_real_policy_top1=row["zero_auxiliary"]["whole_board_search"]["policy"]["top1"],
                    normal_real_continuation=row["with_auxiliary"]["whole_board_search"]["played_terminal_value"],
                    zero_aux_real_continuation=row["zero_auxiliary"]["whole_board_search"]["played_terminal_value"],
                    zero_aux_real_unproven_continuation=row["zero_auxiliary"]["whole_board_search"]["unproven_played_terminal_value"],
                    zero_aux_real_proof=row["zero_auxiliary"]["whole_board_search"]["proof_value"],
                    zero_aux_strategic=row["zero_auxiliary"]["extra_verified_full_board"],
                    opening_zero_auxiliary_estimates=row["opening_zero_auxiliary_estimates"])
    (RUN / "checkpoint_comparison_summary.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
