"""Bounded in-memory diagnostics for the whole-board value head.

Reads existing labels/checkpoint and saves JSON only. No model weights are saved.
Short fixed-subset fitting is a learnability test, not a replacement trained model.
"""
import argparse
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from board_rules import normalize_board
from global_inference import load_global_model
from global_model import GlobalBoardNet, encode_global, global_loss
from train_global import batches


REPRESENTATIONS = ("absolute_rgb", "relative_rgb", "role_planes")


def encode_diagnostic(record, representation="absolute_rgb", swap_colors=False):
    """Zero-auxiliary encoding; optional color change always changes actor too."""
    if representation not in REPRESENTATIONS:
        raise ValueError("Unknown diagnostic representation")
    board = normalize_board(record["board"])
    side = int(record["side"])
    if side not in (1, 2):
        raise ValueError("Explicit actor must be 1 or 2")
    palette = np.array([0, 2, 1, 3], dtype=np.uint8)
    if swap_colors:
        board, side = palette[board], 3 - side
    zero = np.zeros(board.shape, dtype=np.float32)
    if representation == "absolute_rgb":
        return encode_global(board, side, zero, zero)
    normalized = palette[board] if side == 2 else board
    encoded = encode_global(normalized, 1, zero, zero)
    if representation == "role_planes":
        encoded[0] = normalized == 1
        encoded[1] = normalized == 2
        encoded[2] = normalized == 3
    return encoded


def select_balanced(records, sample_count=32, seed=0):
    """Equal quotas by actual actor and proven value; never fabricate labels."""
    if sample_count < 4 or sample_count % 4:
        raise ValueError("sample_count must be a positive multiple of four")
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for record in records:
        proof = record.get("search_proven_value")
        if record["value_valid"] and proof in (-1, 1) and record["value"] == proof:
            groups[(record["side"], int(proof))].append(record)
    result = []
    for key in ((1, -1), (1, 1), (2, -1), (2, 1)):
        group = groups[key]
        if len(group) < sample_count // 4:
            raise ValueError("Insufficient balanced proof samples for " + repr(key))
        indices = rng.permutation(len(group))[:sample_count // 4]
        result.extend(group[index] for index in indices)
    return result


def value_statistics(predictions, records):
    predictions = np.asarray(predictions, dtype=np.float64)
    valid = np.array([record["value_valid"] for record in records], dtype=bool)
    predictions = predictions[valid]
    labels = np.array([record["value"] for record in records], dtype=np.float64)[valid]
    if not len(labels):
        return dict(count=0)
    classified = np.where(predictions > .25, 1, np.where(predictions < -.25, -1, 0))
    return dict(count=len(labels), mse=float(np.mean((predictions-labels)**2)),
                zero_baseline_mse=float(np.mean(labels**2)), mean=float(predictions.mean()),
                std=float(predictions.std()), mean_absolute=float(np.abs(predictions).mean()),
                predicted_draw_fraction=float(np.mean(classified == 0)),
                outcome_accuracy=float(np.mean(classified == labels)),
                sign_accuracy_non_draw=(float(np.mean(np.sign(predictions[labels != 0]) == labels[labels != 0]))
                                        if np.any(labels != 0) else None),
                label_counts=dict(Counter(str(int(label)) for label in labels)),
                mean_prediction_by_value={str(label): float(predictions[labels == label].mean())
                                          for label in (-1, 0, 1) if np.any(labels == label)})


def predict_checkpoint(model, records, zero_auxiliary):
    predictions = []
    ordered = []
    model.eval()
    with torch.inference_mode():
        for (inputs, _, _, _, _), group in batches(
                records, 32, auxiliary_dropout=float(zero_auxiliary),
                input_mode=getattr(model, "input_mode", "absolute_rgb")):
            predictions.extend(model(inputs)[1].cpu().numpy())
            ordered.extend(group)
    return np.asarray(predictions), ordered


def stratified_checkpoint(model, records, train_games):
    output = {}
    for zero_auxiliary in (False, True):
        values, ordered = predict_checkpoint(model, records, zero_auxiliary)
        strata = defaultdict(list)
        for index, row in enumerate(ordered):
            split = "train" if row["game_id"] in train_games else "validation"
            size = "x".join(map(str, np.asarray(row["board"]).shape))
            strata[split].append(index)
            strata[split + "/" + size].append(index)
            if row.get("search_proven_value") is not None:
                strata[split + "/search_proof"].append(index)
            if row["value_valid"] and row["value"] != 0 and row["game_terminal"]:
                remaining = row["game_length"] - row["ply"]
                category = "near_end_le4" if remaining <= 4 else "mid_end_5to12" if remaining <= 12 else "early_gt12"
                strata[split + "/" + category].append(index)
        output["zero_auxiliary" if zero_auxiliary else "with_auxiliary"] = {
            name: value_statistics(values[indices], [ordered[index] for index in indices])
            for name, indices in strata.items()}
    return output


def gradient_diagnostic(model, records):
    model.eval()
    input_mode = getattr(model, "input_mode", "absolute_rgb")
    inputs = torch.from_numpy(np.stack([encode_diagnostic(row, input_mode) for row in records]))
    boards = np.stack([row["board"] for row in records])
    target = np.stack([row["target_policy"] for row in records])
    labels = [row["value"] for row in records]
    logits, values = model(inputs)
    parts = global_loss(logits, values, target, labels, [True]*len(records), boards,
                        return_components=True)
    shared = tuple(model.stem.parameters()) + tuple(model.spatial.parameters())
    policy_grad = torch.cat([grad.flatten() for grad in torch.autograd.grad(
        parts["policy_loss"], shared, retain_graph=True)])
    value_grad = torch.cat([grad.flatten() for grad in torch.autograd.grad(
        parts["value_loss"], shared, retain_graph=True)])
    head_grad = torch.cat([grad.flatten() for grad in torch.autograd.grad(
        parts["value_loss"], tuple(model.value_head.parameters()))])
    pn, vn = float(policy_grad.norm()), float(value_grad.norm())
    return dict(samples=len(records), policy_loss=float(parts["policy_loss"].detach()),
                value_loss=float(parts["value_loss"].detach()), policy_shared_gradient_l2=pn,
                value_shared_gradient_l2=vn, value_head_gradient_l2=float(head_grad.norm()),
                weighted_value_to_policy_gradient_ratio=.3 * vn / max(pn, 1e-12),
                shared_gradient_cosine=float(F.cosine_similarity(policy_grad, value_grad, dim=0)))


def fit_small(records, held_out, representation, *, steps, seed, joint=False, pretrained=None):
    """Fit only this fixed tiny set in memory, reporting separate held-out data."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = copy.deepcopy(pretrained) if pretrained is not None else GlobalBoardNet()
    inputs = torch.from_numpy(np.stack([encode_diagnostic(row, representation, swap)
                                        for swap in (False, True) for row in records]))
    targets = torch.tensor([row["value"] for _ in range(2) for row in records], dtype=torch.float32)
    policies = torch.tensor(np.stack([row["target_policy"] for _ in range(2) for row in records]))
    boards = np.stack([normalize_board(row["board"]) for _ in range(2) for row in records])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=1e-4)
    test_inputs = torch.from_numpy(np.stack([encode_diagnostic(row, representation) for row in held_out]))
    fit_rows = list(records) * 2
    history = []

    def evaluate(step):
        model.eval()
        with torch.inference_mode():
            predicted = model(inputs)[1].cpu().numpy()
            test_predicted = model(test_inputs)[1].cpu().numpy()
        row = dict(step=step, fixed_train=value_statistics(predicted, fit_rows),
                   held_out_near_end=value_statistics(test_predicted, held_out))
        history.append(row)
        return row

    initial = evaluate(0)
    started = time.monotonic()
    check_steps = {min(10, steps), min(30, steps), min(60, steps), steps}
    for step in range(1, steps + 1):
        model.train()
        selection = torch.from_numpy(rng.choice(len(inputs), min(32, len(inputs)), replace=False))
        optimizer.zero_grad(set_to_none=True)
        logits, predicted = model(inputs[selection])
        if joint:
            loss = global_loss(logits, predicted, policies[selection], targets[selection],
                               torch.ones(len(selection), dtype=torch.bool), boards[selection.numpy()],
                               value_weight=.3)
        else:
            loss = F.mse_loss(predicted, targets[selection])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite diagnostic loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step in check_steps:
            evaluate(step)
    return dict(representation=representation, objective="policy_plus_0.3_value" if joint else "value_only",
                initialization="existing_checkpoint" if pretrained is not None else "fresh_same_seed",
                steps=steps, unique_base_samples=len(records), color_augmented_fit_rows=len(inputs),
                held_out_rows=len(held_out), elapsed_seconds=time.monotonic()-started,
                initial=initial, final=history[-1], history=history)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="training_runs/global_v1")
    parser.add_argument("--output", default="training_runs/global_value_diagnostic")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 300 or args.samples > 64 or args.threads < 1:
        parser.error("Diagnostic is bounded to 1..300 steps and at most 64 base samples")
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    weight_path = source / "global.pt"
    original_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    with gzip.open(source / "dataset.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    split = json.loads((source / "split.json").read_text(encoding="utf-8"))
    train_games = set(split.get("train_groups", split["train_games"]))
    training = [row for row in rows if row["game_id"] in train_games and np.asarray(row["board"]).shape == (16,16)]
    chosen = select_balanced(training, args.samples, args.seed)
    held_out = [row for row in rows if row["game_id"] not in train_games
                and np.asarray(row["board"]).shape == (16,16) and row["value_valid"]
                and row["value"] != 0 and row["game_terminal"] and row["game_length"]-row["ply"] <= 4]
    if not held_out:
        raise ValueError("Need actual held-out near-terminal examples")
    model, metadata = load_global_model(weight_path)
    result = dict(configuration=vars(args), checkpoint_sha256=original_hash,
                  checkpoint_best_epoch=metadata.get("best_epoch"),
                  checkpoint_statistics=stratified_checkpoint(model, rows, train_games),
                  checkpoint_gradients=gradient_diagnostic(model, chosen),
                  fixed_samples=[dict(game_id=row["game_id"], ply=row["ply"], side=row["side"],
                                      value=row["value"], proof=row["search_proven_value"]) for row in chosen],
                  limitations=["Short fixed-subset fitting diagnoses learnability, not playing strength.",
                               "Held-out near-terminal games are never optimized.",
                               "Relative encodings are diagnostic prototypes, not compatible with existing checkpoint inputs.",
                               "No model weights are saved or original code changed."],
                  experiments=[])
    (output / "checkpoint_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(event="checkpoint_checked", gradients=result["checkpoint_gradients"])), flush=True)
    plans = [
        ("absolute_rgb", False, None),
        ("relative_rgb", False, None),
        ("role_planes", False, None),
        ("absolute_rgb", True, None),
        (getattr(model, "input_mode", "absolute_rgb"), False, model),
    ]
    for representation, joint, pretrained in plans:
        experiment = fit_small(chosen, held_out, representation, steps=args.steps, seed=args.seed,
                               joint=joint, pretrained=pretrained)
        result["experiments"].append(experiment)
        (output / "diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(dict(event="experiment", representation=representation,
                              objective=experiment["objective"], initialization=experiment["initialization"],
                              elapsed_seconds=experiment["elapsed_seconds"], final=experiment["final"])), flush=True)
    if hashlib.sha256(weight_path.read_bytes()).hexdigest() != original_hash:
        raise AssertionError("Original checkpoint changed while running diagnostic")
    result.update(original_checkpoint_unchanged=True, elapsed_seconds=time.monotonic()-started)
    (output / "diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
