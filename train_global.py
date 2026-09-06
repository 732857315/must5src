"""Train the independent full-board policy/value network from searched games."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import io
import math
import json
from pathlib import Path
import time

import numpy as np
import torch

from global_data import (build_global_dataset, global_dataset_report, split_global_records,
                         apply_value_supervision, record_group, load_extra_positions, physical_board_key,
                         GlobalDataset, TEACHER_ENGINES, VALUE_SUPERVISION_MODES)
from global_model import (GlobalBoardNet, encode_global, global_loss, masked_global_policy,
                          CHANNEL_NAMES, INPUT_MODES)
from unet_board import analyze_board
from unet_pipeline import load_model


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_value), encoding='utf-8')



def _local_hash_identity(hashes, name):
    """Compare local weights by role and digest, allowing directory relocation."""
    if not isinstance(hashes, dict) or len(hashes) != 2:
        raise ValueError(f"{name} must identify exactly opponent.pt and play.pt")
    result = {}
    for filename, checksum in hashes.items():
        if not isinstance(filename, str):
            raise ValueError(f"{name} has an invalid checkpoint path")
        basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        if basename not in ("opponent.pt", "play.pt") or basename in result:
            raise ValueError(f"{name} has a missing, duplicated or unknown local model role")
        if (not isinstance(checksum, str) or len(checksum) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in checksum)):
            raise ValueError(f"{name} contains an invalid SHA256")
        result[basename] = checksum.lower()
    return result


def initialize_global_weights(model, checkpoint_path, local_model_hashes):
    """Initialize compatible model weights; no optimizer or scheduler is restored.

    The complete source payload is checked before changing the destination.
    Source identity is computed from exactly the bytes passed to torch.load.
    """
    if not isinstance(model, GlobalBoardNet):
        raise ValueError("Weight initialization requires a GlobalBoardNet")
    source = Path(checkpoint_path).resolve()
    contents = source.read_bytes()
    source_sha = hashlib.sha256(contents).hexdigest()
    payload = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "gomoku_global_v1":
        raise ValueError("Initialization requires a versioned gomoku_global_v1 checkpoint")
    if payload.get("role") != "global" or payload.get("trained") is not True:
        raise ValueError("Initialization requires trained global model weights")
    architecture = {}
    for name in ("base_channels", "token_dim", "attention_heads"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value != getattr(model, name):
            raise ValueError(f"Initialization checkpoint architecture differs: {name}")
        architecture[name] = value
    source_mode = payload.get("input_mode", "absolute_rgb")
    if source_mode != model.input_mode:
        raise ValueError("Initialization checkpoint input_mode differs from the requested training mode")
    if payload.get("input_channels") != list(CHANNEL_NAMES):
        raise ValueError("Initialization checkpoint input channel names/order differ")
    if "max_token_side" in payload:
        value = payload["max_token_side"]
        if isinstance(value, bool) or not isinstance(value, int) or value != model.max_token_side:
            raise ValueError("Initialization checkpoint max_token_side differs")
    expected_hashes = _local_hash_identity(local_model_hashes, "current local_model_sha256")
    recorded_hashes = _local_hash_identity(payload.get("local_model_sha256"),
                                            "initialization local_model_sha256")
    if recorded_hashes != expected_hashes:
        raise ValueError("Initialization checkpoint belongs to different local model weights")
    expected = model.state_dict()
    weights = payload.get("state_dict")
    if not isinstance(weights, dict) or set(weights) != set(expected):
        raise ValueError("Initialization checkpoint state_dict keys differ")
    for name, target in expected.items():
        tensor = weights[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.shape != target.shape or tensor.dtype != target.dtype
                or not torch.isfinite(tensor).all()):
            raise ValueError(f"Initialization checkpoint has incompatible or nonfinite tensor: {name}")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if "parameters" in payload:
        value = payload["parameters"]
        if isinstance(value, bool) or not isinstance(value, int) or value != parameters:
            raise ValueError("Initialization checkpoint parameter count differs")
    source_epoch = payload.get("best_epoch")
    if source_epoch is not None and (isinstance(source_epoch, bool)
            or not isinstance(source_epoch, int) or source_epoch < 1):
        raise ValueError("Initialization checkpoint best_epoch is invalid")
    model.load_state_dict(weights, strict=True)
    return dict(path=str(source), sha256=source_sha, format=payload["format"], role="global",
                input_mode=source_mode, architecture=architecture,
                local_model_role_sha256=recorded_hashes, source_best_epoch=source_epoch,
                mode="weights_only", optimizer_restored=False, scheduler_restored=False,
                interpretation="Initialize model weights only; optimizer, learning-rate schedule, epoch numbering and checkpoint selection start fresh.")


def append_tactical_positions(records, count, seed, sizes, progress_callback=None):
    """Append verified cases while retaining only real games in records.games."""
    from global_tactics_data import build_tactical_positions
    seen = {row["physical_key"] for row in records}
    additions, report = build_tactical_positions(count=count, seed=seed, sizes=sizes,
                                                seen_physical=seen, progress_callback=progress_callback)
    root = Path(__file__).resolve().parent
    fingerprint = dict(engine="continuous_four_proof",
                       source_sha256=hashlib.sha256((root / "board_forcing.py").read_bytes()).hexdigest(),
                       generator_sha256=hashlib.sha256((root / "global_tactics_data.py").read_bytes()).hexdigest(),
                       binary_sha256=None, binary_path=None)
    for row in additions:
        if record_group(row) != "constructed" or row["physical_key"] in seen:
            raise ValueError("Constructed positions must be independent from existing game groups")
        seen.add(row["physical_key"])
        row.update(teacher_engine="python_forcing", teacher_fingerprint=fingerprint.copy())
    records.extend(additions)
    return report



def load_cached_dataset(source, local_model_hashes):
    """Read a previous run without changing its data, report or checkpoints."""
    source = Path(source)
    cache = source / 'dataset.jsonl.gz' if source.is_dir() else source
    report_path = cache.parent / 'dataset_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report['local_model_sha256'] != local_model_hashes:
        raise ValueError('Cached local features belong to different local checkpoints')
    opener = gzip.open if cache.name.lower().endswith('.gz') else open
    with opener(cache, 'rt', encoding='utf-8') as handle:
        records = GlobalDataset([json.loads(line) for line in handle if line.strip()],
                                games=report.get('games'), generation=report.get('generation'))
    # Preserve groups for older reports lacking the optional complete game list.
    if 'games' not in report:
        records.games = global_dataset_report(list(records))['games']
    provenance = dict(path=str(cache.resolve()), sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                      report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())
    return records, report, provenance


def augment_record(record, turns, reflect, swap_colors, drop_auxiliary=False, *, input_mode="absolute_rgb", include_policy_constraints=False):
    """Transform board and all labels together, then regenerate coordinates."""
    def transform(array):
        result = np.rot90(np.asarray(array), turns)
        return np.ascontiguousarray(np.fliplr(result) if reflect else result)
    board = transform(record['board']).astype(np.uint8)
    side = int(record['side'])
    if swap_colors:
        board = np.array([0, 2, 1, 3], dtype=np.uint8)[board]
        side = 3 - side
    opponent, local, target = [transform(record[name]).astype(np.float32)
                               for name in ('opponent_policy', 'local_policy', 'target_policy')]
    if drop_auxiliary:
        opponent = np.zeros_like(opponent)
        local = np.zeros_like(local)
    item = (encode_global(board, side, opponent, local, input_mode=input_mode), board, target,
            float(record['value']), bool(record['value_valid']))
    if not include_policy_constraints:
        return item
    policy_mask = bool(record.get('policy_mask', True)) and record['policy_source'] != 'all_legal_proved_loss'
    losing = transform(record.get('losing_mask', np.zeros_like(record['board'], dtype=bool)))
    winning = transform(record.get('winning_mask', np.zeros_like(record['board'], dtype=bool)))
    if losing.dtype != np.bool_ or winning.dtype != np.bool_:
        raise ValueError('Constraint masks must be boolean before augmentation')
    if np.all(losing[board == 0]) and np.any(board == 0):
        policy_mask = False
    if not policy_mask:
        item = (*item[:2], np.zeros_like(target), *item[3:])
    return (*item, policy_mask, float(record.get('policy_weight', 1)), losing, winning)


def batches(records, batch_size, rng=None, auxiliary_dropout=0.0, *, input_mode="absolute_rgb", include_policy_constraints=False):
    buckets = defaultdict(list)
    for record in records:
        buckets[np.asarray(record['board']).shape].append(record)
    groups = []
    for rows in buckets.values():
        rows = rows.copy()
        if rng is not None:
            rng.shuffle(rows)
        groups.extend(rows[start:start + batch_size] for start in range(0, len(rows), batch_size))
    if rng is not None:
        rng.shuffle(groups)
    for group in groups:
        # Rectangles rotate by a shared parity so all items still share a shape.
        parity = int(rng.integers(2)) if rng is not None else 0
        items = []
        for record in group:
            if rng is None:
                options = (0, False, False, auxiliary_dropout == 1)
            else:
                options = (parity + 2 * int(rng.integers(2)), bool(rng.integers(2)),
                           bool(rng.integers(2)), bool(rng.random() < auxiliary_dropout))
            items.append(augment_record(record, *options, input_mode=input_mode,
                                        include_policy_constraints=include_policy_constraints))
        arrays = [np.stack(values) for values in zip(*items)]
        inputs, boards, policies, values, masks = arrays[:5]
        packed = (torch.from_numpy(inputs), boards, torch.from_numpy(policies),
                  torch.tensor(values, dtype=torch.float32), torch.from_numpy(masks))
        if include_policy_constraints:
            packed += tuple(torch.from_numpy(value) for value in arrays[5:])
        yield packed, group


def policy_metrics(probabilities, records):
    selected = [(p, r) for p, r in zip(probabilities, records)
                if r.get('policy_mask', True) and r['policy_source'] != 'all_legal_proved_loss']
    probabilities, records = ([p for p, _ in selected], [r for _, r in selected])
    successes = 0
    loss = 0.0
    per_size = defaultdict(lambda: [0, 0])
    per_source = defaultdict(lambda: [0, 0])
    for probability, record in zip(probabilities, records):
        target = np.asarray(record['target_policy']).reshape(-1)
        probability = np.asarray(probability).reshape(-1)
        correct = bool(target[int(probability.argmax())] >= target.max() - 1e-7)
        successes += int(correct)
        loss -= float(np.sum(target * np.log(np.maximum(probability, 1e-12))))
        for group, key in ((per_size, 'x'.join(map(str, np.asarray(record['board']).shape))),
                           (per_source, record['policy_source'])):
            group[key][0] += int(correct)
            group[key][1] += 1
    def summarize(group):
        return {key: dict(correct=counts[0], total=counts[1], top1=counts[0] / counts[1])
                for key, counts in group.items()}
    return dict(total=len(records), correct=successes,
                top1=successes / len(records) if records else None,
                cross_entropy=loss / len(records) if records else None, per_size=summarize(per_size),
                per_source=summarize(per_source))


def value_metrics(predictions, records):
    selected = [(float(value), float(record['value'])) for value, record in zip(predictions, records)
                if record['value_valid']]
    if not selected:
        return dict(total=0, mse=None, mae=None, outcome_accuracy=None,
                    zero_baseline_mse=None, target_counts={}, prediction_counts={}, mean_prediction=None)
    predictions, targets = np.asarray(selected).T
    classes = np.where(predictions > .25, 1, np.where(predictions < -.25, -1, 0))
    return dict(total=len(selected), mse=float(np.mean((predictions - targets) ** 2)),
                mae=float(np.mean(np.abs(predictions - targets))),
                outcome_accuracy=float(np.mean(classes == targets)),
                zero_baseline_mse=float(np.mean(targets ** 2)),
                target_counts={str(value): int(np.count_nonzero(targets == value)) for value in (-1, 0, 1)},
                prediction_counts={str(value): int(np.count_nonzero(classes == value)) for value in (-1, 0, 1)},
                mean_prediction=float(predictions.mean()))


def evaluate(model, records, batch_size, zero_auxiliary=False):
    model.eval()
    probabilities, values, ordered = [], [], []
    with torch.inference_mode():
        for (inputs, boards, _, _, _), group in batches(
                records, batch_size, auxiliary_dropout=float(zero_auxiliary),
                input_mode=getattr(model, "input_mode", "absolute_rgb")):
            logits, prediction = model(inputs)
            probabilities.extend(masked_global_policy(logits, boards)[:, 0].cpu().numpy())
            values.extend(prediction.cpu().numpy())
            ordered.extend(group)
    # Removing auxiliary information also removes the local inference blend.
    # Keep a normalized policy rather than retaining only its 70% contribution.
    combined = probabilities if zero_auxiliary else [
        .7 * probability + .3 * np.asarray(record['local_policy'])
        for probability, record in zip(probabilities, ordered)]
    def summarize(indices):
        rows = [ordered[index] for index in indices]
        predictions = [values[index] for index in indices]
        sources = sorted({row.get('value_source', 'unrecorded_legacy') for row in rows})
        by_value_source = {}
        for source in sources:
            selected = [i for i, row in enumerate(rows) if row.get('value_source', 'unrecorded_legacy') == source]
            by_value_source[source] = value_metrics([predictions[i] for i in selected], [rows[i] for i in selected])
        # Report real played outcomes even when a proof-only run masks them in
        # its loss. This exposes saturation on general continuation positions.
        terminal_rows = [dict(row, value=row.get('terminal_value', 0),
                              value_valid=row.get('terminal_value_valid', False)) for row in rows]
        return dict(policy=policy_metrics([probabilities[index] for index in indices], rows),
                    combined_policy=policy_metrics([combined[index] for index in indices], rows),
                    value=value_metrics(predictions, rows), value_by_source=by_value_source,
                    terminal_continuation_value=value_metrics(predictions, terminal_rows))
    result = summarize(list(range(len(ordered))))
    for kind in ("real_games", "constructed"):
        indices = [index for index, row in enumerate(ordered) if record_group(row) == kind]
        result[kind] = summarize(indices)
        if kind == "constructed":
            informative = [index for index in indices
                           if ordered[index]["policy_source"] != "all_legal_proved_loss"]
            rows = [ordered[index] for index in informative]
            result[kind]["informative_policy"] = policy_metrics([probabilities[index] for index in informative], rows)
            result[kind]["informative_combined_policy"] = policy_metrics([combined[index] for index in informative], rows)
            result[kind]["uninformative_proved_loss_records"] = len(indices)-len(informative)
    from global_policy_training import constraint_metrics
    result['constraints'] = dict(policy=constraint_metrics(probabilities, ordered),
                                 combined=constraint_metrics(combined, ordered))
    result['by_source'] = {source: summarize([i for i, row in enumerate(ordered)
                                             if row.get('source', 'unrecorded_legacy') == source])
                           for source in sorted({r.get('source', 'unrecorded_legacy') for r in ordered})}
    return result


def checkpoint_score(metrics, value_weight=0.3):
    """Select by policy alone when value training is disabled.

    Positive value weights retain the original fixed 0.02 value-error penalty.
    All-legal proved-loss labels still cannot inflate policy selection.
    """
    if (isinstance(value_weight, bool) or not isinstance(value_weight, (int, float))
            or not math.isfinite(value_weight) or value_weight < 0):
        raise ValueError("value_weight must be finite and nonnegative")
    policy = metrics["real_games"]["combined_policy"]
    if not policy["total"]:
        policy = metrics["constructed"]["informative_combined_policy"]
    if value_weight == 0:
        return policy["top1"] or 0.0
    return (policy["top1"] or 0.0) - .02 * (metrics["value"]["mse"] or 0.0)


def tactical_guard(initial, current, limit):
    """Check each tactical class independently for both deployed policies."""
    checks = {}
    for key in ('policy', 'combined_policy'):
        for source in ('immediate_win_set', 'forced_block'):
            previous = initial[key]['per_source'].get(source)
            present = current[key]['per_source'].get(source)
            if previous is None or previous['total'] == 0:
                continue
            if present is None or present['total'] != previous['total']:
                raise ValueError('Tactical validation population changed during training')
            baseline = previous['correct'] / previous['total']
            accuracy = present['correct'] / present['total']
            checks[key + ':' + source] = dict(total=previous['total'], initial=baseline,
                current=accuracy, change=accuracy-baseline, passed=accuracy >= baseline-limit)
    return dict(passed=all(row['passed'] for row in checks.values()), limit=limit, checks=checks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-models', default='training_runs/unet_curriculum_v2')
    parser.add_argument('--output-dir', default='training_runs/global_v1')
    parser.add_argument('--init-checkpoint',
                        help='Initialize compatible global model weights in a new output directory; optimizer and learning-rate schedule start fresh')
    parser.add_argument('--games', type=int, default=80)
    parser.add_argument('--sizes', default='6,10,16')
    parser.add_argument('--max-moves', type=int, default=64)
    parser.add_argument('--max-positions', type=int, default=4000)
    parser.add_argument('--tactical-positions', type=int, default=0,
                        help='Additional verified constructed cases, separate from played games')
    parser.add_argument('--teacher-depth', type=int, default=5)
    parser.add_argument('--teacher-seconds', type=float, default=.06)
    parser.add_argument('--teacher-nodes', type=int, default=512)
    parser.add_argument('--teacher-width', type=int, default=8)
    parser.add_argument('--teacher-engine', choices=TEACHER_ENGINES, default='python')
    parser.add_argument('--teacher-forcing-seconds', type=float, default=0.0)
    parser.add_argument('--teacher-forcing-nodes', type=int, default=10000)
    parser.add_argument('--teacher-forcing-depth', type=int, default=24)
    parser.add_argument('--value-supervision', choices=VALUE_SUPERVISION_MODES, default='proven_or_terminal')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--auxiliary-dropout', type=float, default=.3)
    parser.add_argument('--value-weight', type=float, default=.3)
    parser.add_argument('--input-mode', choices=INPUT_MODES, default='absolute_rgb')
    parser.add_argument('--seed', type=int, default=20260906)
    data_input = parser.add_mutually_exclusive_group()
    data_input.add_argument('--reuse-data', action='store_true', help='Reuse the output directory cache in memory')
    data_input.add_argument('--data-source', help='Read a previous run directory or dataset JSONL[.GZ] into a new output directory')
    parser.add_argument('--policy-constraints', action='append', default=[], metavar='JSONL',
                        help='Append independently verified action sets; preserve cached train/validation ownership')
    parser.add_argument('--constraint-repeats', type=int, default=1,
                        help='Training exposures per informative constraint per epoch; does not create new unique records')
    parser.add_argument('--tactical-regression-limit', type=float, default=.02,
                        help='Maximum absolute tactical validation accuracy drop for constraint checkpoint selection')
    parser.add_argument('--searched-policy-weight', type=float, default=1.0,
                        help='Scale ordinary bounded-search imitation CE; proved action losses keep unit weight')
    parser.add_argument('--extra-positions', action='append', default=[], metavar='JSONL',
                        help='Append externally proved full-board cases; repeat for multiple JSONL[.GZ] files')
    args = parser.parse_args(argv)
    if (args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or args.tactical_positions < 0
            or not 0 <= args.auxiliary_dropout <= 1):
        parser.error('epochs/batch-size/threads must be positive; auxiliary-dropout must be in [0,1]')
    if args.constraint_repeats < 1 or (args.constraint_repeats > 1 and not args.policy_constraints):
        parser.error('constraint-repeats must be positive and repeated sampling requires --policy-constraints')
    if not math.isfinite(args.tactical_regression_limit) or not 0 <= args.tactical_regression_limit <= 1:
        parser.error('tactical-regression-limit must be in [0,1]')
    if not math.isfinite(args.searched_policy_weight) or args.searched_policy_weight < 0:
        parser.error('searched-policy-weight must be finite and nonnegative')
    if args.policy_constraints and not args.data_source:
        parser.error('--policy-constraints requires --data-source with its frozen split.json and a new output directory')
    if not math.isfinite(args.value_weight) or args.value_weight < 0:
        parser.error('value-weight must be finite and nonnegative')
    if args.init_checkpoint:
        initialization_path = Path(args.init_checkpoint).resolve()
        output_path = Path(args.output_dir).resolve()
        if not initialization_path.is_file():
            parser.error('--init-checkpoint must name an existing checkpoint file')
        if args.reuse_data:
            parser.error('--init-checkpoint cannot overwrite a reused run; use --data-source with a new output directory')
        if initialization_path.parent == output_path or (output_path.exists() and
                (not output_path.is_dir() or any(output_path.iterdir()))):
            parser.error('--init-checkpoint requires a new empty output directory')
    if args.reuse_data and (args.extra_positions or args.tactical_positions):
        parser.error('Use --data-source and a new --output-dir when adding cases to a cached dataset')
    if args.data_source:
        source = Path(args.data_source)
        source_cache = source / 'dataset.jsonl.gz' if source.is_dir() else source
        output_path = Path(args.output_dir)
        if source_cache.resolve().parent == output_path.resolve():
            parser.error('--data-source requires a different output directory')
        if (output_path / 'global.pt').exists() or (output_path / 'dataset.jsonl.gz').exists():
            parser.error('--data-source requires a new output directory without an existing dataset or checkpoint')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model_paths = [Path(args.local_models) / name for name in ('opponent.pt', 'play.pt')]
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in model_paths}
    initialized_model, initialization = None, None
    if args.init_checkpoint:
        initialized_model = GlobalBoardNet(input_mode=args.input_mode)
        initialization = initialize_global_weights(initialized_model, args.init_checkpoint, hashes)
    selection_rule = ('real_game_policy_else_informative_constructed_policy_only'
                      if args.value_weight == 0 else
                      'real_game_policy_else_informative_constructed_policy_minus_0.02_value_mse')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / 'progress.jsonl'
    def log(event):
        event = dict(event, elapsed_seconds=round(time.monotonic() - started, 3))
        line = json.dumps(event, ensure_ascii=False, default=json_value)
        print(line, flush=True)
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')
    started = time.monotonic()
    configuration = dict(vars(args), initialization=initialization, checkpoint_selection=selection_rule)
    write_json(output / 'config.json', configuration)
    if initialization is not None:
        log(dict(event='weights_initialized', initialization=initialization))
    cache = output / 'dataset.jsonl.gz'
    sizes = [tuple(map(int, value.split('x'))) if 'x' in value else int(value)
             for value in args.sizes.split(',')]
    cached_report, provenance = {}, None
    tactical_report, extra_report = None, None
    if args.reuse_data or args.data_source:
        records, cached_report, provenance = load_cached_dataset(args.data_source or cache, hashes)
        feature_start = len(records)
        log(dict(event='data_reused', records=len(records), source=provenance))
    else:
        def generation_progress(event):
            if event['games_completed'] % 4 == 0 or event['games_completed'] == args.games:
                log(dict(event='games', games=event['games_completed'], records=event['records'],
                         last_game=event['game']))
        records = build_global_dataset(games=args.games, seed=args.seed, sizes=sizes,
                    max_moves=args.max_moves, max_positions=args.max_positions,
                    search_depth=args.teacher_depth, search_time_limit=args.teacher_seconds,
                    search_max_nodes=args.teacher_nodes, candidate_width=args.teacher_width,
                    teacher_engine=args.teacher_engine, forcing_seconds=args.teacher_forcing_seconds,
                    forcing_nodes=args.teacher_forcing_nodes, forcing_depth=args.teacher_forcing_depth,
                    value_supervision=args.value_supervision, progress_callback=generation_progress)
        feature_start = 0
    if args.tactical_positions:
        tactical_report = append_tactical_positions(
            records, args.tactical_positions, args.seed + 1, sizes,
            progress_callback=lambda event: log(dict(event='tactical_positions', **event)))
    if args.extra_positions:
        additions, extra_report = load_extra_positions(
            args.extra_positions, seen_physical={physical_board_key(row['board']) for row in records})
        records.extend(additions)
        log(dict(event='extra_positions', **extra_report))
    constraint_report = None
    preserved_split = None
    if args.policy_constraints:
        from global_policy_constraints import load_policy_constraints
        from global_policy_training import merge_constraints, preserved_policy_split
        additions, constraint_report = load_policy_constraints(args.policy_constraints)
        source = Path(args.data_source)
        split_path = (source if source.is_dir() else source.parent) / 'split.json'
        source_split = json.loads(split_path.read_text(encoding='utf-8'))
        constraint_report['merge'] = merge_constraints(records, additions)
        preserved_split = preserved_policy_split(records, source_split, seed=args.seed)
        constraint_report['source_split'] = dict(path=str(split_path.resolve()),
            sha256=hashlib.sha256(split_path.read_bytes()).hexdigest())
        log(dict(event='policy_constraints', **constraint_report))
    for record in records:
        record['policy_weight'] = args.searched_policy_weight if record['policy_source'] == 'searched_move' else 1.0
    # Convert value supervision in memory. A separate output receives a new
    # cache; --reuse-data leaves the existing cache and its report unchanged.
    apply_value_supervision(records, args.value_supervision)
    dataset_report = dict(cached_report)
    dataset_report.update(global_dataset_report(records))
    dataset_report.update(local_model_sha256=hashes, value_supervision=args.value_supervision,
                          cached_value_supervision=cached_report.get('value_supervision', args.value_supervision))
    if provenance is not None:
        dataset_report['data_source'] = provenance
    if tactical_report is not None:
        dataset_report['constructed_generation'] = tactical_report
    if extra_report is not None:
        dataset_report['extra_positions_import'] = extra_report
    if constraint_report is not None:
        dataset_report['policy_constraints_import'] = constraint_report
    if feature_start < len(records):
        opponent, _ = load_model(model_paths[0], 'opponent')
        local, _ = load_model(model_paths[1], 'play')
        for parameter in list(opponent.parameters()) + list(local.parameters()):
            parameter.requires_grad_(False)
        for index in range(feature_start, len(records)):
            record = records[index]
            result = analyze_board(record['board'], record['side'], opponent, local, tactical=False)
            record['opponent_policy'] = result['opponent_policy'].astype(np.float32)
            record['local_policy'] = result['raw_play_policy'].astype(np.float32)
            if (index + 1 - feature_start) % 200 == 0 or index + 1 == len(records):
                log(dict(event='local_features', complete=index + 1 - feature_start, total=len(records)-feature_start))
    if not args.reuse_data:
        with gzip.open(cache, 'wt', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, default=json_value) + '\n')
        write_json(output / 'dataset_report.json', dataset_report)
    write_json(output / 'training_dataset_report.json', dataset_report)
    train, validation = preserved_split or split_global_records(records, seed=args.seed)
    if constraint_report is not None:
        from global_policy_training import has_informative_constraints
        if not any(has_informative_constraints(row) for row in train) or not any(has_informative_constraints(row) for row in validation):
            raise ValueError('Constraint training requires informative evidence in independent training and validation source groups')
        selection_rule = 'heldout_constraint_quality_then_legacy_policy_with_tactical_guard'
        configuration['checkpoint_selection'] = selection_rule
        write_json(output / 'config.json', configuration)
    split = dict(train_groups=sorted({r['game_id'] for r in train}),
                 validation_groups=sorted({r['game_id'] for r in validation}),
                 train_games=sorted({r['game_id'] for r in train if record_group(r) == 'real_games'}),
                 validation_games=sorted({r['game_id'] for r in validation if record_group(r) == 'real_games'}),
                 train_constructed_cases=sorted({r['game_id'] for r in train if record_group(r) == 'constructed'}),
                 validation_constructed_cases=sorted({r['game_id'] for r in validation if record_group(r) == 'constructed'}),
                 train_extra_position_groups=sorted({r['game_id'] for r in train if r.get('source') == 'extra_verified_full_board'}),
                 validation_extra_position_groups=sorted({r['game_id'] for r in validation if r.get('source') == 'extra_verified_full_board'}),
                 train_records=len(train), validation_records=len(validation),
                 train_value_labels=sum(r['value_valid'] for r in train),
                 validation_value_labels=sum(r['value_valid'] for r in validation),
                 augmentation='D4 and simultaneous color/side exchange after game split',
                 auxiliary_policy_dropout=args.auxiliary_dropout,
                 value_supervision=args.value_supervision,
                 physical_board_overlap=len({r['physical_key'] for r in train} & {r['physical_key'] for r in validation}))
    write_json(output / 'split.json', split)
    baseline = policy_metrics([r['local_policy'] for r in validation], validation)
    for kind in ('real_games', 'constructed'):
        subset = [r for r in validation if record_group(r) == kind]
        baseline[kind] = policy_metrics([r['local_policy'] for r in subset], subset)
    log(dict(event='training_start', train_records=len(train), validation_records=len(validation),
             train_value_labels=split['train_value_labels'], validation_value_labels=split['validation_value_labels'],
             local_policy_baseline=baseline))
    epoch_records = list(train)
    if args.constraint_repeats > 1:
        epoch_records += [r for r in train if has_informative_constraints(r)] * (args.constraint_repeats - 1)
    log(dict(event='training_sampler', unique_train_records=len(train),
             train_exposures_per_epoch=len(epoch_records), constraint_repeats=args.constraint_repeats))
    model = initialized_model if initialized_model is not None else GlobalBoardNet(input_mode=args.input_mode)
    initial_metrics = evaluate(model, validation, args.batch_size) if constraint_report is not None else None
    if initial_metrics is not None:
        write_json(output / 'initial_validation.json', initial_metrics)
        log(dict(event='constraint_selection_baseline', validation=initial_metrics,
                 tactical_regression_limit=args.tactical_regression_limit))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * .1)
    history, best, best_score = [], None, ((-float('inf'), -float('inf')) if constraint_report is not None else -float('inf'))
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        components = defaultdict(float)
        component_rows = defaultdict(int)
        active_rows = defaultdict(int)
        skipped_updates = 0
        for (inputs, boards, policies, values, masks, policy_mask, policy_weights, losing_mask, winning_mask), group in batches(
                epoch_records, args.batch_size, rng, args.auxiliary_dropout, input_mode=model.input_mode,
                include_policy_constraints=True):
            optimizer.zero_grad(set_to_none=True)
            logits, predictions = model(inputs)
            parts = global_loss(logits, predictions, policies, values, masks, boards,
                               value_weight=args.value_weight, policy_mask=policy_mask,
                               policy_weights=policy_weights, losing_mask=losing_mask,
                               winning_mask=winning_mask, return_components=True)
            loss = parts['loss']
            counts = dict(ce_loss=int(parts['ce_count']), losing_loss=int(parts['losing_count']),
                          winning_loss=int(parts['winning_count']), value_loss=int(masks.sum()))
            for name, count in counts.items():
                components[name] += float(parts[name].detach()) * count
                component_rows[name] += count
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            for name in ('ce_count', 'losing_count', 'winning_count'):
                active_rows[name] += int(parts[name])
            contributes = bool((policy_mask & (policy_weights > 0)).any()
                               or parts['losing_count'] > 0 or parts['winning_count'] > 0
                               or (args.value_weight > 0 and masks.any()))
            if contributes:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            else:
                # Adam momentum and weight decay must not update an entirely
                # unlabelled batch, including all-actions-proved-loss rows.
                skipped_updates += 1
            total_loss += float(loss.detach()) * len(group)
            total += len(group)
        scheduler.step()
        metrics = evaluate(model, validation, args.batch_size)
        # A policy-only run must not be selected by its untrained value head.
        # These are validation metrics, not an independent match win rate.
        score = checkpoint_score(metrics, value_weight=args.value_weight)
        if constraint_report is not None:
            quality = metrics['constraints']['policy']['quality']
            score = (quality, score)
        row = dict(event='epoch', epoch=epoch, loss=total_loss / total,
                   loss_components={k: v / component_rows[k] if component_rows[k] else None for k, v in components.items()},
                   loss_component_rows=dict(component_rows),
                   policy_active_rows=dict(active_rows), skipped_optimizer_updates=skipped_updates,
                   validation=metrics)
        history.append(row)
        eligible = True
        if initial_metrics is not None:
            guard = tactical_guard(initial_metrics, metrics, args.tactical_regression_limit)
            eligible = guard['passed']
            row['checkpoint_tactical_guard_passed'] = eligible
            row['checkpoint_tactical_guard'] = guard
        if eligible and score > best_score:
            best_score, best = score, epoch
            checkpoint = dict(format='gomoku_global_v1', role='global', trained=True,
                              input_mode=model.input_mode, value_supervision=args.value_supervision,
                              teacher_engines=dataset_report['teacher_engines'],
                              teacher_fingerprints=dataset_report['teacher_fingerprints'],
                              checkpoint_selection=selection_rule, initialization=initialization,
                              base_channels=model.base_channels, token_dim=model.token_dim,
                              attention_heads=model.attention_heads, state_dict=model.state_dict(),
                              best_epoch=epoch, parameters=sum(p.numel() for p in model.parameters()),
                              input_channels=list(CHANNEL_NAMES), training_configuration=configuration,
                              unique_samples=len(records), train_samples=len(train), validation_samples=len(validation),
                              auxiliary_policy_dropout=args.auxiliary_dropout,
                              local_model_sha256=hashes, validation=metrics)
            torch.save(checkpoint, output / 'global.pt')
        log(row)
    if best is None:
        write_json(output / 'history.json', history)
        raise ValueError('No trained epoch met the preregistered tactical regression limit; no candidate selected')
    payload = torch.load(output / 'global.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(payload['state_dict'])
    final = evaluate(model, validation, args.batch_size)
    zero_auxiliary = evaluate(model, validation, args.batch_size, zero_auxiliary=True)
    after = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in model_paths}
    if after != hashes:
        raise AssertionError('Frozen local model checkpoint changed during global training')
    if initialization is not None:
        if hashlib.sha256(Path(initialization['path']).read_bytes()).hexdigest() != initialization['sha256']:
            raise AssertionError('Initialization checkpoint changed during training')
    summary = dict(trained=True, best_epoch=best, epochs_completed=len(history),
                   parameters=sum(p.numel() for p in model.parameters()), elapsed_seconds=time.monotonic() - started,
                   dataset={k: v for k, v in dataset_report.items() if k != 'games'}, split=split,
                   validation=final, validation_zero_auxiliary=zero_auxiliary,
                   local_policy_baseline=baseline, local_checkpoints_unchanged=True,
                   initialization=initialization, checkpoint_selection=selection_rule,
                   initialization_checkpoint_unchanged=True if initialization is not None else None,
                   checkpoint=str((output / 'global.pt').resolve()),
                   limitations=['Search imitation from bounded trajectories, not an unbeatable-play proof.',
                                'Held-out games select the checkpoint; no independent match claim.',
                                ('Value labels use explicit search certificates only.'
                                 if args.value_supervision == 'proven' else
                                 'Value labels describe the played continuation, not minimax truth unless search-proven.')])
    write_json(output / 'history.json', history)
    write_json(output / 'summary.json', summary)
    log(dict(event='complete', summary=summary))
    return summary


if __name__ == '__main__':
    main()
