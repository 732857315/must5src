"""Teacher provenance, proof-only labels, and constructed-case isolation."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from global_data import (apply_value_supervision, build_global_dataset, global_dataset_report,
                         record_group, split_global_records)
from train_global import append_tactical_positions, checkpoint_score, evaluate
from tests.test_train_global import EchoLocal, fixture


class GlobalTeacherTests(unittest.TestCase):
    def test_native_dispatch_preserves_budgets_and_actual_binary_fingerprint(self):
        fingerprint = dict(engine="native_test", source_sha256="a"*64, binary_sha256="b"*64,
                           binary_path="test-only.dll")
        def native(board, side, priors, **options):
            self.assertEqual(options, dict(max_nodes=17, time_limit=.02, depth=5,
                                           candidate_width=9, forcing_seconds=.03,
                                           forcing_nodes=111, forcing_depth=12))
            return dict(move=tuple(map(int, np.argwhere(board == 0)[0])), nodes=2,
                        completed_depth=1, proven_value=None, engine="native_test",
                        forcing=dict(nodes=3, proven_value=None))
        with patch("native_search.native_fingerprint", return_value=fingerprint), \
             patch("native_search.select_native_move", side_effect=native) as searched, \
             patch("global_data.select_move") as python:
            rows = build_global_dataset(games=1, sizes=(16,), max_moves=2, teacher_engine="native",
                                        search_depth=5, search_time_limit=.02, search_max_nodes=17,
                                        candidate_width=9, forcing_seconds=.03,
                                        forcing_nodes=111, forcing_depth=12)
        self.assertEqual(searched.call_count, 1)
        python.assert_not_called()
        self.assertEqual(rows[0]["search_engine"], "native_test")
        self.assertEqual(rows[0]["teacher_engine"], "native")
        self.assertEqual(rows[0]["teacher_fingerprint"]["source_sha256"], "a"*64)
        self.assertEqual(rows[0]["teacher_fingerprint"]["binary_sha256"], "b"*64)
        self.assertEqual(rows[0]["search_forcing"]["nodes"], 3)
        self.assertEqual(global_dataset_report(rows)["teacher_engines"], {"native": 1})
        self.assertEqual(rows.generation["teacher_fingerprint"], rows[0]["teacher_fingerprint"])

    def test_empty_native_request_never_builds_or_loads_a_library(self):
        with patch("native_search.native_fingerprint") as fingerprint:
            self.assertEqual(build_global_dataset(games=0, teacher_engine="native"), [])
        fingerprint.assert_not_called()

    def test_value_modes_preserve_actual_outcome_and_mask_unproven_terminals(self):
        rows = [
            dict(side=1, game_terminal=True, game_winner=1, search_proven_value=None),
            dict(side=2, game_terminal=True, game_winner=1, search_proven_value=None),
            dict(side=2, game_terminal=True, game_winner=0, search_proven_value=None),
            dict(side=1, game_terminal=False, game_winner=None, search_proven_value=0),
            dict(side=2, game_terminal=False, game_winner=None, search_proven_value=-1),
            dict(side=1, game_terminal=False, game_winner=None, search_proven_value=None),
        ]
        apply_value_supervision(rows)
        self.assertEqual([r["value"] for r in rows], [1, -1, 0, 0, -1, 0])
        self.assertEqual([r["value_valid"] for r in rows], [True, True, True, True, True, False])
        apply_value_supervision(rows, "proven")
        self.assertEqual([r["value_valid"] for r in rows], [False, False, False, True, True, False])
        self.assertEqual([r["terminal_value"] for r in rows], [1, -1, 0, None, None, None])
        self.assertEqual([r["terminal_value_valid"] for r in rows], [True, True, True, False, False, False])
        apply_value_supervision(rows)
        self.assertEqual([r["value"] for r in rows], [1, -1, 0, 0, -1, 0])

    def test_conflicting_proof_is_retained_separately_from_played_result(self):
        row = dict(side=2, game_terminal=True, game_winner=1, search_proven_value=1)
        apply_value_supervision([row])
        self.assertEqual((row["value"], row["terminal_value"]), (-1, -1))
        apply_value_supervision([row], "proven")
        self.assertEqual((row["value"], row["terminal_value"]), (1, -1))
        self.assertEqual(row["value_source"], "search_proof")

    def test_only_proofs_label_a_completed_game_in_proven_mode(self):
        board = np.full((6, 6), 3, dtype=np.uint8)
        board[0, :3] = [1, 2, 0]
        start = (board, 1, [(1, (0,0)), (2, (0,1))], "fixture")
        with patch("global_data._start_game", return_value=start), \
             patch("global_data.select_move", return_value=dict(move=(0,2), proven_value=None)):
            rows = build_global_dataset(games=1, sizes=(6,), max_moves=4, value_supervision="proven")
        self.assertFalse(rows[0]["value_valid"])
        self.assertEqual(rows[0]["terminal_value"], 0)
        report = global_dataset_report(rows)
        self.assertEqual(report["games_terminal"], 1)
        self.assertEqual(report["valid_values"], {})
        self.assertEqual(report["terminal_values"], {"0": 1})

    def test_invalid_engine_forcing_and_supervision_are_rejected(self):
        for config in (dict(teacher_engine="other"), dict(value_supervision="terminal_only"),
                       dict(forcing_seconds=.03), dict(teacher_engine="native", sizes=(65,)),
                       dict(teacher_engine="native", forcing_depth=4097),
                       dict(teacher_engine="native", forcing_seconds=float("nan"))):
            with self.subTest(config=config), self.assertRaises(ValueError):
                build_global_dataset(games=0, **config)


class ConstructedIntegrationTests(unittest.TestCase):
    def test_constructed_cases_do_not_become_truncated_games_or_leak_across_groups(self):
        rows = build_global_dataset(games=2, sizes=(6, 7), max_moves=2,
                                    search_time_limit=0, search_max_nodes=0)
        game_count = len(rows.games)
        report = append_tactical_positions(rows, 4, seed=73, sizes=(16,))
        self.assertEqual(report["constructed_cases"], 4)
        self.assertEqual(len(rows.games), game_count)
        apply_value_supervision(rows, "proven")
        for container in (rows, list(rows)):
            report = global_dataset_report(container)
            self.assertEqual(report["games_started"], 2)
            self.assertEqual(report["games_truncated"], 2)
            self.assertEqual(report["constructed_cases"], 4)
        train, validation = split_global_records(rows, seed=17)
        self.assertTrue({r["physical_key"] for r in train}.isdisjoint(r["physical_key"] for r in validation))
        self.assertEqual({record_group(r) for r in train}, {"real_games", "constructed"})
        self.assertEqual({record_group(r) for r in validation}, {"real_games", "constructed"})
        self.assertTrue(all(r["teacher_fingerprint"]["engine"] == "continuous_four_proof"
                            for r in rows if record_group(r) == "constructed"))

    def test_uniform_proved_loss_cannot_inflate_real_policy_or_checkpoint_score(self):
        real = fixture()
        real["value"] = 0
        constructed = copy.deepcopy(real)
        constructed.update(group_kind="constructed_position", source="constructed_verified_full_board",
                           policy_source="all_legal_proved_loss", value=-1)
        constructed["target_policy"] = (constructed["board"] == 0).astype(np.float32)
        constructed["target_policy"] /= constructed["target_policy"].sum()
        # The uniform network misses the real target. All-loss rows are excluded
        # from every policy metric, so their tied targets cannot inflate top-1.
        result = evaluate(EchoLocal(uniform=True), [real, constructed], 2, zero_auxiliary=True)
        self.assertEqual(result["policy"]["top1"], 0)
        self.assertEqual(result["policy"]["total"], 1)
        self.assertNotIn("all_legal_proved_loss", result["policy"]["per_source"])
        self.assertEqual(result["real_games"]["policy"]["top1"], 0)
        self.assertIsNone(result["constructed"]["policy"]["top1"])
        self.assertEqual(result["constructed"]["policy"]["total"], 0)
        self.assertEqual(result["constructed"]["informative_policy"]["total"], 0)
        self.assertEqual(result["constructed"]["uninformative_proved_loss_records"], 1)
        self.assertLessEqual(checkpoint_score(result), 0)

    def test_duplicate_constructed_board_is_rejected_before_append(self):
        row = fixture()
        row.update(physical_key="same-board", game_id="real-0")
        duplicate = dict(row, group_kind="constructed_position")
        rows = [row]
        with patch("global_tactics_data.build_tactical_positions", return_value=([duplicate], {})):
            with self.assertRaisesRegex(ValueError, "independent"):
                append_tactical_positions(rows, 1, 0, (16,))
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
