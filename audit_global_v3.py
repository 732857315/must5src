"""Independent data/provenance audit for global_v3; never modifies training inputs."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from board_rules import apply_board_move, board_winner
from global_data import load_extra_positions, split_global_records


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "training_runs/global_v2"
RUN = ROOT / "training_runs/global_v3"
EXTRA = [ROOT / "training_runs" / name / "positions.jsonl.gz"
         for name in ("strategic_v3", "strategic_v3_more")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def identity(board, side):
    """Independently recompute D4 + joint actor/color identity."""
    original = np.asarray(board, dtype=np.uint8)
    relative = original.copy()
    relative[original == side] = 1
    relative[original == 3 - side] = 2
    choices = []
    for mirror in (False, True):
        view = relative[:, ::-1] if mirror else relative
        for turns in range(4):
            image = np.rot90(view, turns)
            choices.append((image.shape, bytes(image.ravel())))
    shape, cells = min(choices)
    return str(shape[0]) + "x" + str(shape[1]) + ":" + cells.hex()


def keys(row):
    one, two = identity(row["board"], 1), identity(row["board"], 2)
    return (one if row["side"] == 1 else two), min(one, two)


def summarize(rows):
    result = dict(records=len(rows), groups=len({r["game_id"] for r in rows}),
                  sources=dict(Counter(r.get("source") for r in rows)),
                  value_sources=dict(Counter(r["value_source"] for r in rows)),
                  values=dict(Counter(str(int(r["value"])) for r in rows if r["value_valid"])),
                  unknown_values=sum(not r["value_valid"] for r in rows))
    result["by_source"] = {}
    for source in sorted(result["sources"]):
        group = [r for r in rows if r.get("source") == source]
        result["by_source"][source] = dict(records=len(group), groups=len({r["game_id"] for r in group}),
                 values=dict(Counter(str(int(r["value"])) for r in group if r["value_valid"])),
                 policy_sources=dict(Counter(r["policy_source"] for r in group)))
    return result


def audit(cache_required=False):
    original = read_rows(BASE / "dataset.jsonl.gz")
    old_report = json.loads((BASE / "dataset_report.json").read_text(encoding="utf-8"))
    old_split = json.loads((BASE / "split.json").read_text(encoding="utf-8"))
    old_audit = json.loads((BASE / "data_audit.json").read_text(encoding="utf-8"))
    old_comparison = json.loads((BASE / "checkpoint_comparison.json").read_text(encoding="utf-8"))
    assert sha(BASE / "dataset.jsonl.gz") == old_audit["dataset_sha256"]
    assert sha(BASE / "global.pt") == old_comparison["models"]["v2"]["checkpoint_sha256"]
    frozen_hashes = old_report["local_model_sha256"]
    assert all(sha(path) == digest for path, digest in frozen_hashes.items())
    extra, import_report = load_extra_positions(EXTRA, seen_physical={keys(r)[1] for r in original})
    assert len(extra) == 1536 and import_report["duplicates_skipped"] == 0
    assert len({r["game_id"] for r in extra}) == 4

    # Bind every pair to its root certificate, its actual played prefix source,
    # and all legal defender replies. This checks certificate structure/labels;
    # it does not rerun or substitute for the recursive solver's proof logic.
    proof_groups = defaultdict(list)
    for row in extra:
        proof_groups[row["proof_file_resolved"]].append(row)
    reply_branches = 0
    for filename, pair in proof_groups.items():
        assert len(pair) == 2
        positive = next(r for r in pair if r["search_proven_value"] == 1)
        negative = next(r for r in pair if r["search_proven_value"] == -1)
        assert positive["game_id"] == negative["game_id"]
        with gzip.open(filename, "rt", encoding="utf-8") as handle:
            certificate = json.load(handle)
        board, side, result = np.asarray(certificate["board"], dtype=np.uint8), certificate["side"], certificate["result"]
        assert np.array_equal(board, positive["board"]) and side == positive["side"]
        assert result["proven_value"] == 1
        candidate = result["winning_candidate"]
        move = tuple(result["move"])
        assert tuple(candidate["move"]) == move and positive["target_policy"][move] == 1
        child = apply_board_move(board, move, side)
        assert np.array_equal(child, negative["board"]) and negative["side"] == 3 - side
        legal = {tuple(map(int, point)) for point in np.argwhere(child == 0)}
        replies = candidate["reply_proofs"]
        assert len(replies) == len(legal)
        assert {tuple(reply["move"]) for reply in replies} == legal
        assert all(reply["continuation"]["proven_value"] == 1 for reply in replies)
        assert negative["policy_source"] == "all_legal_proved_loss"
        assert np.allclose(negative["target_policy"][child == 0], 1 / len(legal))
        reply_branches += len(replies)

    cache = RUN / "dataset.jsonl.gz"
    split_path = RUN / "split.json"
    ready = cache.exists() and split_path.exists()
    if cache_required and not ready:
        raise RuntimeError("global_v3 cache/split is not complete; no report was written")
    if ready:
        rows = read_rows(cache)
        report = json.loads((RUN / "dataset_report.json").read_text(encoding="utf-8"))
        actual_split = json.loads(split_path.read_text(encoding="utf-8"))
        assert report["local_model_sha256"] == frozen_hashes
        assert report["data_source"]["sha256"] == sha(BASE / "dataset.jsonl.gz")
    else:
        rows = [dict(r) for r in original] + extra
        for row in rows:
            terminal = row.get("game_terminal", False)
            proof = row.get("search_proven_value")
            if terminal:
                winner = row["game_winner"]
                row.update(value=0. if winner == 0 else 1. if winner == row["side"] else -1.,
                           value_valid=True, value_source="terminal")
            elif proof is not None:
                row.update(value=float(proof), value_valid=True, value_source="search_proof")
            else:
                row.update(value=0., value_valid=False, value_source="unknown")
        report, actual_split = old_report, None

    actor_seen, physical_seen, owners = set(), set(), {}
    label_counts = Counter()
    terminal_proof_pairs, conflicts = Counter(), Counter()
    for row in rows:
        board, target = np.asarray(row["board"]), np.asarray(row["target_policy"])
        side = row["side"]
        assert board.shape == (16, 16) and board.dtype.kind in "iu"
        assert np.isin(board, [0, 1, 2, 3]).all() and side in (1, 2)
        assert np.count_nonzero(board == 1) - np.count_nonzero(board == 2) == (0 if side == 1 else 1)
        assert not board_winner(board) and np.any(board == 0)
        assert target.shape == board.shape and np.isfinite(target).all() and np.all(target >= 0)
        assert np.isclose(target.sum(), 1, atol=1e-6)
        assert np.all(target[board != 0] == 0)
        actor, physical = keys(row)
        assert row["board_key"] == actor and row["physical_key"] == physical
        assert actor not in actor_seen and physical not in physical_seen
        actor_seen.add(actor)
        physical_seen.add(physical)
        assert physical not in owners or owners[physical] == row["game_id"]
        owners[physical] = row["game_id"]
        if row.get("game_terminal"):
            expected = 0. if row["game_winner"] == 0 else 1. if row["game_winner"] == side else -1.
            assert row["value_valid"] and row["value"] == expected and row["value_source"] == "terminal"
            assert row["terminal_value_valid"] and row["terminal_value"] == expected
            label_counts["terminal"] += 1
            if row.get("search_proven_value") is not None:
                terminal_proof_pairs[row.get("source", "unknown")] += 1
                if expected != row["search_proven_value"]:
                    conflicts[row.get("source", "unknown")] += 1
        elif row.get("search_proven_value") is not None:
            assert row["value_valid"] and row["value"] == row["search_proven_value"]
            assert row["value_source"] == "search_proof"
            label_counts["proof"] += 1
        else:
            assert not row["value_valid"] and row["value"] == 0 and row["value_source"] == "unknown"
            label_counts["unknown"] += 1

    train, validation = split_global_records(rows, seed=20260908)
    train_ids, validation_ids = {r["game_id"] for r in train}, {r["game_id"] for r in validation}
    assert train_ids.isdisjoint(validation_ids)
    assert {r["physical_key"] for r in train}.isdisjoint(r["physical_key"] for r in validation)
    real_validation = [r for r in validation if r["source"] == "whole_board_search"]
    assert sorted({r["game_id"] for r in real_validation}) == old_split["validation_games"]
    assert len(real_validation) == 1152
    if actual_split is not None:
        assert sorted(train_ids) == actual_split["train_groups"]
        assert sorted(validation_ids) == actual_split["validation_groups"]
    # Values may change when recovering terminal supervision; all prior board,
    # policy, local-feature, search and source metadata must remain untouched.
    mutable = {"value", "value_valid", "value_source", "value_supervision"}
    original_by_key = {r["board_key"]: r for r in original}
    for row in rows:
        previous = original_by_key.get(row["board_key"])
        if previous is not None:
            assert {k: v for k, v in previous.items() if k not in mutable} == {k: v for k, v in row.items() if k not in mutable}
    assert len(rows) == len(original) + len(extra)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True, cache_audited=ready,
                  records=len(rows), independent_d4_actor_color_recomputation=True,
                  dedup=dict(unique_actor_keys=len(actor_seen), unique_physical_keys=len(physical_seen),
                             duplicates=0, cross_group_owners=0),
                  labels=dict(label_counts),
                  terminal_vs_proof=dict(dual_valid=sum(terminal_proof_pairs.values()),
                                         dual_valid_by_source=dict(terminal_proof_pairs),
                                         disagreements=sum(conflicts.values()), disagreements_by_source=dict(conflicts),
                                         selected_label_priority="played_terminal_then_search_proof"),
                  partitions=dict(all=summarize(rows), train=summarize(train), validation=summarize(validation)),
                  real_games=dict(started=report["games_started"], terminal=report["games_terminal"], truncated=report["games_truncated"],
                                  original_validation_groups_unchanged=28, original_validation_records_unchanged=1152),
                  extra_certificates=dict(pairs=len(proof_groups), legal_defender_branches=reply_branches,
                                          source_game_groups=4, label_structure_checked=True, recursive_solver_not_rerun=True),
                  inputs=dict(base_dataset_sha256=sha(BASE / "dataset.jsonl.gz"),
                              old_global_model_sha256=sha(BASE / "global.pt"),
                              frozen_local_model_sha256=frozen_hashes, extra_import=import_report),
                  old_data_and_models_unchanged=True, original_boards_policies_features_provenance_unchanged=len(original),
                  group_leakage=0, physical_leakage=0)
    if ready:
        result["dataset_sha256"] = sha(cache)
        result["split_sha256"] = sha(split_path)
        (RUN / "data_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-required", action="store_true")
    arguments = parser.parse_args()
    result = audit(arguments.cache_required)
    print(json.dumps({key: result[key] for key in ("passed", "cache_audited", "records", "labels", "real_games",
                                                    "extra_certificates", "terminal_vs_proof", "old_data_and_models_unchanged")}, ensure_ascii=False))
