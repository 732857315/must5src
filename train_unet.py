"""Three-stage supervised curriculum, canonical base samples and online augmentation."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from game import BLACK, WHITE, EMPTY
from unet_codec import BLACK_RGB, WHITE_RGB, EMPTY_RGB, FORBIDDEN_RGB, RED_RGB
from unet_models import OpponentUNet5x5, PlayUNet5x5, masked_probabilities, policy_loss
from unet_pipeline import CHECKPOINT_FORMAT


def split_records(records):
    """One stable assignment for an entire D4 plus color-exchange equivalence class."""
    from unet_curriculum import base_key
    train, validation = [], []
    for record in records:
        key = str(base_key(record["state"]))
        bucket = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big") % 5
        (validation if bucket == 0 else train).append(record)
    if not train or not validation:
        raise ValueError("Dataset needs both training and independent validation equivalence classes")
    return train, validation



def expand_actor_records(records):
    """Expand labels only AFTER splitting base boards, keeping both actors together."""
    expanded = []
    for record in records:
        expanded.append(dict(record))
        counter = dict(record)
        counter["side"] = 3 - record["side"]
        for field in ("target", "value", "value_kind", "outcome"):
            counter[field] = record["counter_" + field]
        counter["counterfactual_actor"] = True
        expanded.append(counter)
    return expanded

def encode_records(records):
    states = torch.tensor([int(record["state"]) for record in records], dtype=torch.int64)
    shifts = 2 * torch.arange(25, dtype=torch.int64)
    cells = ((states[:, None] >> shifts) & 3).reshape(-1, 5, 5)
    sides = torch.tensor([int(record["side"]) for record in records], dtype=torch.long)
    targets = torch.from_numpy(np.stack([record["target"] for record in records]).astype(np.float32)).reshape(-1, 1, 5, 5)
    return {"cells": cells, "sides": sides, "targets": targets, "states": states}


def rgb_batch(cells, opponent=None):
    palette = torch.tensor([EMPTY_RGB, BLACK_RGB, WHITE_RGB, FORBIDDEN_RGB], dtype=torch.float32) / 255
    rgb = palette[cells].permute(0, 3, 1, 2).contiguous()
    if opponent is not None:
        alpha = opponent * (cells == EMPTY)[:, None]
        color = torch.tensor(RED_RGB, dtype=torch.float32)[None, :, None, None] / 255
        rgb = rgb * (1 - alpha) + color * alpha
    return rgb


def augmented_batch(encoded, indices, generator, *, augment=True):
    cells = encoded["cells"][indices].clone()
    sides = encoded["sides"][indices].clone()
    targets = encoded["targets"][indices].clone()
    opponent = encoded.get("opponent")
    opponent = None if opponent is None else opponent[indices].clone()
    if augment:
        symmetries = torch.randint(8, (len(indices),), generator=generator)
        for symmetry in range(8):
            selected = symmetries == symmetry
            if not selected.any():
                continue
            def transform(value):
                rotated = torch.rot90(value, symmetry % 4, dims=(-2, -1))
                return rotated.flip(-1) if symmetry >= 4 else rotated
            cells[selected] = transform(cells[selected])
            targets[selected] = transform(targets[selected])
            if opponent is not None:
                opponent[selected] = transform(opponent[selected])
        swap = torch.rand(len(indices), generator=generator) < 0.5
        selected_cells = cells[swap]
        cells[swap] = torch.where((selected_cells == BLACK) | (selected_cells == WHITE), 3 - selected_cells, selected_cells)
        sides[swap] = 3 - sides[swap]
    shifts = 2 * torch.arange(25, dtype=torch.int64)
    states = (cells.reshape(-1, 25) << shifts).sum(dim=1)
    return rgb_batch(cells, opponent), sides, targets, states


def attach_opponent_predictions(encoded, model, batch_size):
    """Freeze red inputs predicted on current boards; never use observed future labels."""
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(encoded["sides"]), batch_size):
            sl = slice(start, start + batch_size)
            logits = model(rgb_batch(encoded["cells"][sl]), 3 - encoded["sides"][sl])
            predictions.append(masked_probabilities(logits, encoded["states"][sl]))
    encoded["opponent"] = torch.cat(predictions).clone()


def evaluate(model, encoded, records, batch_size):
    model.eval()
    correct, loss_sum, total = 0, 0.0, 0
    by_source, by_side, by_value, by_stage, by_pattern = {}, {}, {}, {}, {}
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(records)))
            rgb, sides, targets, states = augmented_batch(encoded, indices, None, augment=False)
            logits = model(rgb, sides)
            probabilities = masked_probabilities(logits, states).flatten(1)
            choices = probabilities.argmax(1)
            # Soft targets can support every legal move; only teacher maxima count.
            flat_targets = targets.flatten(1)
            chosen_target = flat_targets.gather(1, choices[:, None]).squeeze(1)
            hits = chosen_target >= flat_targets.max(1).values - 1e-6
            loss_sum += float(policy_loss(logits, targets, states)) * len(indices)
            correct += int(hits.sum())
            total += len(indices)
            for offset, hit in enumerate(hits.tolist()):
                record = records[start + offset]
                groups = ((by_source, str(record.get("source", "unknown"))),
                          (by_side, str(record["side"])),
                          (by_value, str(record.get('outcome', 'unknown'))),
                          (by_stage, str(record['stage'])),
                          (by_pattern, str(record.get('pattern', 'unclassified'))))
                for mapping, key in groups:
                    item = mapping.setdefault(key, {"correct": 0, "count": 0})
                    item["correct"] += int(hit)
                    item["count"] += 1
    for mapping in (by_source, by_side, by_value, by_stage, by_pattern):
        for item in mapping.values():
            item["accuracy"] = item["correct"] / item["count"]
    return {"count": total, "expert_top1": correct / total, "cross_entropy": loss_sum / total,
            "source_balanced_top1": sum(item['accuracy'] for item in by_source.values()) / len(by_source),
            "by_source": by_source, "by_side": by_side, "by_outcome": by_value,
            "by_stage": by_stage, "by_pattern": by_pattern}


def checkpoint(model, path, metadata):
    payload = {"format": CHECKPOINT_FORMAT, "role": model.role, "base_channels": model.base_channels,
               "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
               "input_contract": "absolute_rgb_v1; play uses continuous predicted opponent red",
               "canonicalization": "D4 + color exchange with side exchange", **metadata}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def fit(model, train_records, validation_records, *, opponent, stage, args, directory, generator):
    encoded_train, encoded_validation = encode_records(train_records), encode_records(validation_records)
    if model.role == "play":
        attach_opponent_predictions(encoded_train, opponent, args.batch_size)
        attach_opponent_predictions(encoded_validation, opponent, args.batch_size)
    initial_metrics = evaluate(model, encoded_validation, validation_records, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_score, best_loss, best_epoch, stale = -1.0, math.inf, 0, 0
    stage_path = directory / f"stage{stage}_{model.role}.pt"
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_records), generator=generator)
        loss_sum = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            rgb, sides, targets, states = augmented_batch(encoded_train, indices, generator)
            optimizer.zero_grad(set_to_none=True)
            loss = policy_loss(model(rgb, sides), targets, states)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        metrics = evaluate(model, encoded_validation, validation_records, args.batch_size)
        entry = {"epoch": epoch, "training_loss": loss_sum / len(order), **metrics}
        history.append(entry)
        score = metrics["source_balanced_top1"]
        better = score > best_score + 1e-8 or (
            abs(score - best_score) <= 1e-8 and metrics["cross_entropy"] < best_loss)
        if better:
            best_score, best_loss, best_epoch, stale = score, metrics["cross_entropy"], epoch, 0
            checkpoint(model, stage_path, {"trained": True, "stage": stage, "epoch": epoch,
                       "training_samples": len(train_records), "validation": metrics})
        else:
            stale += 1
        print(f"stage={stage} model={model.role} epoch={epoch}/{args.epochs} "
              f"loss={entry['training_loss']:.4f} val_top1={metrics['expert_top1']:.3%} "
              f"balanced_best={best_score:.3%} elapsed={time.monotonic()-started:.1f}s", flush=True)
        with (directory / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": stage, "role": model.role, **entry}) + "\n")
        if stale and stale % 6 == 0:
            for group in optimizer.param_groups:
                group["lr"] = max(group["lr"] * 0.5, 5e-5)
        if epoch >= 24 and stale >= args.patience:
            break
    payload = torch.load(stage_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return {"stage": stage, "role": model.role, "best_epoch": best_epoch, "epochs_run": len(history),
            "elapsed_seconds": time.monotonic() - started, "initial_validation": initial_metrics,
            "best_validation": payload["validation"]}


def distribution(records):
    def count(key):
        return dict(Counter(str(key(record)) for record in records))
    return {"count": len(records), "side": count(lambda r: r["side"]),
            "source": count(lambda r: r.get("source", "unknown")),
            "outcome": count(lambda r: r.get('outcome', 'unknown')),
            "value_kind": count(lambda r: r.get('value_kind', 'unknown')),
            "pattern": count(lambda r: r.get('pattern', 'unclassified')),
            "forbidden_count": count(lambda r: sum(((r["state"] >> (2*i)) & 3) == 3 for i in range(25)))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-stage", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--teacher-temperature", type=float, default=0.025,
                        help="soft-label temperature for quiet heuristic positions; exact labels are unchanged")
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args(argv)
    if min(args.samples_per_stage, args.epochs, args.batch_size, args.base_channels, args.threads, args.patience) < 1:
        parser.error("sample counts, epochs, channels, threads and patience must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if not math.isfinite(args.teacher_temperature) or args.teacher_temperature <= 0:
        parser.error("teacher temperature must be finite and positive")
    directory = Path(args.output_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        parser.error("output directory is not empty; use a new directory to preserve previous training")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    from unet_curriculum import build_curriculum_dataset, base_key, dataset_report
    from unet_patterns import catalogue_report
    catalogue = catalogue_report()
    (directory / "pattern_catalogue.json").write_text(json.dumps(catalogue, ensure_ascii=False, indent=2), encoding="utf-8")
    opponent, play = OpponentUNet5x5(args.base_channels), PlayUNet5x5(args.base_channels)
    print(f"parameters: opponent={sum(p.numel() for p in opponent.parameters())}, "
          f"play={sum(p.numel() for p in play.parameters())}; device=cpu threads={args.threads}", flush=True)
    all_records, used, reports = [], set(), []
    start_all = time.monotonic()
    for stage in (1, 2, 3):
        started = time.monotonic()
        records = build_curriculum_dataset(stage, args.samples_per_stage, args.seed + stage,
                                           exclude_keys=used, teacher_temperature=args.teacher_temperature)
        keys = [base_key(record["state"]) for record in records]
        if len(set(keys)) != len(keys) or used.intersection(keys):
            raise AssertionError("Equivalent boards leaked into the base dataset more than once")
        used.update(keys)
        all_records.extend(records)
        with gzip.open(directory / f"stage{stage}_base.jsonl.gz", "wt", encoding="utf-8") as stream:
            for record in records:
                serial = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in record.items()}
                stream.write(json.dumps(serial) + "\n")
        print(f"stage={stage} dataset_ready unique={len(records)} elapsed={time.monotonic()-started:.1f}s "
              f"distribution={json.dumps(distribution(records), ensure_ascii=False)}", flush=True)
        # Keep earlier skills while introducing the next forbidden-cell curriculum.
        current_train, current_validation = split_records(records)
        earlier = [record for record in all_records if record["stage"] != stage]
        if earlier:
            prior_train, prior_validation = split_records(earlier)
            rng = random.Random(args.seed + 100 * stage)
            replay = rng.sample(prior_train, min(len(prior_train), len(current_train) // 2))
            training = current_train + replay
            validation = current_validation + prior_validation
        else:
            training, validation = current_train, current_validation
        report = {"stage": stage, "base_distribution": distribution(records),
                  "pattern_coverage": dataset_report(records).get("pattern_coverage"),
                  "training_base_with_replay": len(training), "validation_base": len(validation)}
        training, validation = expand_actor_records(training), expand_actor_records(validation)
        report.update(training_actor_labels=len(training), validation_actor_labels=len(validation))
        for model in (opponent, play):
            report[model.role] = fit(model, training, validation, opponent=opponent, stage=stage,
                                      args=args, directory=directory, generator=generator)
        reports.append(report)
        summary = {"config": vars(args), "stages": reports, "unique_base_samples": len(all_records),
                   "equivalence": "D4 rotations/reflections + black/white and predicted-side exchange",
                   "exhaustive_all_boards": False, "base_contains_both_actor_labels": True, "elapsed_seconds": time.monotonic() - start_all}
        (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _, held_out_base = split_records(all_records)
    held_out = expand_actor_records(held_out_base)
    final_encoded = encode_records(held_out)
    final_metrics = {"opponent": evaluate(opponent, final_encoded, held_out, args.batch_size)}
    attach_opponent_predictions(final_encoded, opponent, args.batch_size)
    final_metrics["play"] = evaluate(play, final_encoded, held_out, args.batch_size)
    for model in (opponent, play):
        checkpoint(model, directory / f"{model.role}.pt", {"trained": True, "stage": 3,
                   "unique_base_samples": len(all_records), "validation": final_metrics[model.role],
                   "teacher_temperature": args.teacher_temperature, "seed": args.seed,
                   "augmentation": "online random D4 and color+side exchange",
                   "label_semantics": "expert move distribution; not calibrated human frequencies"})
    summary["final_validation"] = final_metrics
    summary["validation_base_samples"] = len(held_out_base)
    summary["actor_labels"] = len(all_records) * 2
    summary["elapsed_seconds"] = time.monotonic() - start_all
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TRAINING_COMPLETE {directory} metrics={json.dumps(final_metrics)}", flush=True)


if __name__ == "__main__":
    main()
