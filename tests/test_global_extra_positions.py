"""External proof import and restoration of broad played-outcome supervision."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from global_data import (apply_value_supervision, canonical_board_key, global_dataset_report,
                         load_extra_positions, physical_board_key, split_global_records)
from train_global import evaluate, json_value, load_cached_dataset, main
from tests.test_train_global import EchoLocal, fixture


def extra_fixture(group="tree-a", variant=0):
    board = np.zeros((6, 7), dtype=np.uint8)
    board[0, 1], board[1, 4], board[3, 0], board[4, 3] = 1, 2, 3, 1
    if variant:
        board[5, variant] = 3
    target = np.zeros(board.shape)
    target[2, 5] = 1
    return dict(board=board.tolist(), side=2, target_policy=target.tolist(),
                search_proven_value=1, value_source="search_proof",
                proof_source="recursive-test-v1", group_id=group)


def write_rows(path, rows):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=json_value) + "\n")


class ExtraPositionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_jsonl_and_gzip_share_tree_group_but_never_count_as_games(self):
        first, second = self.root / "a.jsonl", self.root / "b.jsonl.gz"
        write_rows(first, [extra_fixture("tree-a"), extra_fixture("tree-b", 1)])
        write_rows(second, [extra_fixture("tree-a", 2), extra_fixture("tree-c", 3)])
        rows, imported = load_extra_positions([first, second])
        report = global_dataset_report(rows)
        self.assertEqual((len(rows), imported["groups"], report["games_started"]), (4, 3, 0))
        self.assertEqual(report["extra_position_records"], 4)
        self.assertEqual(report["sources"], {"extra_verified_full_board": 4})
        self.assertEqual(imported["files"][1]["sha256"], hashlib.sha256(second.read_bytes()).hexdigest())
        train, validation = split_global_records(rows, seed=3)
        self.assertTrue({r["game_id"] for r in train}.isdisjoint(r["game_id"] for r in validation))
        self.assertTrue({r["physical_key"] for r in train}.isdisjoint(r["physical_key"] for r in validation))
        self.assertTrue(all(r["value_valid"] and r["value"] == 1 and not r["terminal_value_valid"] for r in rows))

    def test_all_d4_and_joint_color_variants_are_one_physical_case(self):
        source = extra_fixture()
        variants = []
        for turns in range(4):
            for reflect in (False, True):
                row = copy.deepcopy(source)
                board, policy = np.rot90(row["board"], turns), np.rot90(row["target_policy"], turns)
                if reflect:
                    board, policy = np.fliplr(board), np.fliplr(policy)
                board = np.array([0, 2, 1, 3])[board]
                row.update(board=board.tolist(), target_policy=policy.tolist(), side=1,
                           board_key="fake", physical_key="fake", opponent_policy="untrusted")
                variants.append(row)
        path = self.root / "symmetry.jsonl"
        write_rows(path, [source] + variants)
        rows, report = load_extra_positions(path)
        self.assertEqual((len(rows), report["duplicates_skipped"]), (1, 8))
        self.assertEqual(rows[0]["board_key"], canonical_board_key(source["board"], 2))
        self.assertNotIn("opponent_policy", rows[0])
        blocked, report = load_extra_positions(path, seen_physical=[rows[0]["physical_key"]])
        self.assertEqual(blocked, [])
        self.assertEqual(report["duplicates_skipped"], 9)

    def test_duplicate_actor_conflicting_proof_is_rejected(self):
        row = extra_fixture()
        conflict = dict(row, search_proven_value=-1)
        path = self.root / "conflict.jsonl"
        write_rows(path, [row, conflict])
        with self.assertRaisesRegex(ValueError, "conflicting proof"):
            load_extra_positions(path)

    def test_schema_rejects_nonproof_and_terminal_or_illegal_policy(self):
        invalid = []
        for update in (dict(side=True), dict(side=1.0), dict(side=3), dict(group_id=""),
                       dict(proof_source=""), dict(search_proven_value=True),
                       dict(search_proven_value=.5), dict(value_source="heuristic"),
                       dict(value=-1), dict(value_valid=False), dict(game_terminal=True),
                       dict(terminal_value=1), dict(game_winner=0), dict(action=[2.0, 5]),
                       dict(action=[0, 1]), dict(target_policy=[1] * 42),
                       dict(target_policy=np.full((6, 7), float("nan")).tolist())):
            invalid.append(dict(extra_fixture(), **update))
        occupied = extra_fixture()
        occupied["target_policy"][2][5], occupied["target_policy"][3][0] = 0, 1
        invalid.append(occupied)
        won = extra_fixture()
        won["board"][0][:5] = [1] * 5
        invalid.append(won)
        full = extra_fixture()
        full["board"] = np.full((6, 7), 3).tolist()
        invalid.append(full)
        path = self.root / "bad.jsonl"
        for index, row in enumerate(invalid):
            with self.subTest(index=index):
                write_rows(path, [row])
                with self.assertRaisesRegex(ValueError, "bad.jsonl:1:"):
                    load_extra_positions(path)

    def test_flat_legal_loss_target_is_marked_uninformative(self):
        row = extra_fixture()
        mask = np.asarray(row["board"]) == 0
        row.update(search_proven_value=-1, target_policy=(mask / mask.sum()).reshape(-1).tolist())
        path = self.root / "loss.jsonl"
        write_rows(path, [row])
        rows, _ = load_extra_positions(path)
        self.assertEqual(rows[0]["target_policy"].shape, (6, 7))
        self.assertEqual(rows[0]["policy_source"], "all_legal_proved_loss")

    def test_certificate_and_source_provenance_hashes_are_retained_and_checked(self):
        proof = self.root / 'proof.json.gz'
        source = self.root / 'source.json'
        proof.write_bytes(gzip.compress(b'{"proof":"test fixture only"}'))
        source.write_text('{"history":[]}', encoding='utf-8')
        row = extra_fixture()
        row.update(proof_file=proof.name, proof_file_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
                   source_file=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   prefix_plies=30, variant_index=7, source_kind='reproved_match_prefix_with_remote_context',
                   policy_source='strategic_proof_move')
        path = self.root / 'proofs.jsonl'
        write_rows(path, [row])
        rows, _ = load_extra_positions(path)
        self.assertEqual(rows[0]['proof_file_resolved'], str(proof.resolve()))
        for key in ('proof_file', 'proof_file_sha256', 'source_file', 'source_sha256',
                    'prefix_plies', 'variant_index', 'source_kind', 'policy_source'):
            self.assertEqual(rows[0][key], row[key])
        proof.write_bytes(b'changed after certification')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            load_extra_positions(path)

    def test_two_extra_source_games_get_separate_train_and_validation_holdouts(self):
        from global_tactics_data import build_tactical_positions
        tactical, _ = build_tactical_positions(6, seed=312, sizes=(16,))
        path = self.root / 'extra.jsonl'
        write_rows(path, [extra_fixture('source-a'), extra_fixture('source-b', 1)])
        extras, _ = load_extra_positions(path)
        train, validation = split_global_records(tactical + extras, seed=20260908)
        for subset in (train, validation):
            self.assertEqual(sum(r.get('source') == 'extra_verified_full_board' for r in subset), 1)
            self.assertTrue(any(r.get('source') == 'constructed_verified_full_board' for r in subset))
        self.assertTrue({r['game_id'] for r in train}.isdisjoint(r['game_id'] for r in validation))

    def test_noninteger_board_and_nonfinite_negative_or_unnormalized_targets_rejected(self):
        cases = []
        row = extra_fixture()
        row["board"] = np.asarray(row["board"], dtype=float).tolist()
        cases.append(row)
        for value in (-1, .8, float("inf")):
            row = extra_fixture()
            row["target_policy"][2][5] = value
            cases.append(row)
        path = self.root / "bad.jsonl"
        for row in cases:
            write_rows(path, [row])
            with self.assertRaises(ValueError):
                load_extra_positions(path)


class ValueRestorationTests(unittest.TestCase):
    def test_retained_terminal_label_is_reversible_without_winner_metadata(self):
        rows = [dict(side=2, terminal_value=-1, terminal_value_valid=True,
                     search_proven_value=None),
                dict(side=1, terminal_value=None, terminal_value_valid=False,
                     value=0, value_valid=False, search_proven_value=None)]
        apply_value_supervision(rows, "proven")
        self.assertFalse(rows[0]["value_valid"])
        apply_value_supervision(rows, "proven_or_terminal")
        self.assertEqual((rows[0]["value"], rows[0]["value_source"]), (-1, "terminal"))
        self.assertFalse(rows[1]["value_valid"])

    def test_retained_terminal_metadata_must_match_winner_and_actor(self):
        for row in (dict(side=1, game_terminal=True, game_winner=1,
                         terminal_value=-1, terminal_value_valid=True),
                    dict(side=1, game_terminal=False, terminal_value=1, terminal_value_valid=True),
                    dict(side=1, game_terminal="false")):
            with self.subTest(row=row), self.assertRaises(ValueError):
                apply_value_supervision([row])

    def test_evaluation_reports_masked_continuations_zero_baseline_and_sources(self):
        real, extra = fixture(side=1), fixture(side=2)
        real.update(value=0, value_valid=False, value_source="unknown", source="whole_board_search",
                    terminal_value=-1, terminal_value_valid=True)
        extra.update(group_kind="constructed_position", source="extra_verified_full_board",
                     value=-1, value_source="search_proof", terminal_value=None, terminal_value_valid=False)
        metrics = evaluate(EchoLocal(), [real, extra], 2, zero_auxiliary=True)
        self.assertEqual(metrics["real_games"]["value"]["total"], 0)
        terminal = metrics["real_games"]["terminal_continuation_value"]
        self.assertEqual(terminal["total"], 1)
        self.assertEqual(terminal["zero_baseline_mse"], 1)
        self.assertEqual(terminal["target_counts"], {"-1": 1, "0": 0, "1": 0})
        self.assertEqual(metrics["by_source"]["extra_verified_full_board"]["value"]["total"], 1)
        self.assertEqual(metrics["constructed"]["value_by_source"]["search_proof"]["total"], 1)

    def test_new_output_reuses_frozen_cache_and_featurizes_only_new_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new, local = root / "old", root / "new", root / "local"
            old.mkdir()
            local.mkdir()
            hashes = {}
            for name in ("opponent.pt", "play.pt"):
                path = local / name
                path.write_bytes(b"test fixture, never loaded")
                hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = []
            for i in range(2):
                row = fixture(shape=(6, 7 + i), side=1, game_id=f"real-{i}")
                row.update(board_key=canonical_board_key(row["board"], 1), game_terminal=True,
                           game_winner=2, game_termination="win", search_proven_value=None,
                           source="whole_board_search", search_requested_depth=3, search_completed_depth=1)
                rows.append(row)
            apply_value_supervision(rows, "proven")
            cache = old / "dataset.jsonl.gz"
            write_rows(cache, rows)
            report = global_dataset_report(rows)
            report.update(local_model_sha256=hashes, value_supervision="proven")
            (old / "dataset_report.json").write_text(json.dumps(report), encoding="utf-8")
            before = {p.name: p.read_bytes() for p in old.iterdir()}
            extra = root / "extra.jsonl"
            write_rows(extra, [extra_fixture("case-a", 1), extra_fixture("case-b", 2)])
            loaded, loaded_report, _ = load_cached_dataset(old, hashes)
            self.assertEqual(len(loaded.games), 2)
            self.assertEqual(loaded_report["value_supervision"], "proven")
            def features(board, side, *args, **kwargs):
                legal = (np.asarray(board) == 0).astype(np.float32)
                legal /= legal.sum()
                return dict(opponent_policy=legal, raw_play_policy=legal)
            with patch("train_global.load_model", return_value=(SimpleNamespace(parameters=lambda: []), {})), \
                 patch("train_global.analyze_board", side_effect=features) as featurize, \
                 patch("train_global.GlobalBoardNet", side_effect=RuntimeError("stop before optimization")), \
                 patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "stop before optimization"):
                    main(["--data-source", str(old), "--output-dir", str(new),
                          "--local-models", str(local), "--extra-positions", str(extra),
                          "--value-supervision", "proven_or_terminal", "--epochs", "1", "--threads", "1"])
            self.assertEqual(featurize.call_count, 2)
            self.assertEqual(before, {p.name: p.read_bytes() for p in old.iterdir()})
            self.assertFalse((new / "global.pt").exists())
            with gzip.open(new / "dataset.jsonl.gz", "rt", encoding="utf-8") as handle:
                restored = [json.loads(line) for line in handle]
            self.assertEqual([r["value_source"] for r in restored], ["terminal", "terminal", "search_proof", "search_proof"])
            report = json.loads((new / "dataset_report.json").read_text(encoding="utf-8"))
            self.assertEqual((report["games_started"], report["extra_position_records"]), (2, 2))
            self.assertEqual(report["local_model_sha256"], hashes)


if __name__ == "__main__":
    unittest.main()
