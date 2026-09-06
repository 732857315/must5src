import unittest
from unittest.mock import patch

import numpy as np

from board_rules import BLACK, WHITE, FORBIDDEN, board_winner, swap_board_colors
from global_data import (build_global_dataset, canonical_board_key, physical_board_key,
                         split_global_records, global_dataset_report)


def forced_win_start(shape, rng, game_index, max_moves):
    board = np.zeros((6, 6), dtype=np.uint8)
    opening = [(BLACK, (2, 1)), (WHITE, (0, 0)), (BLACK, (2, 2)),
               (WHITE, (5, 0)), (BLACK, (2, 3)), (WHITE, (5, 5))]
    for side, point in opening:
        board[point] = side
    return board, BLACK, opening, "fixture"


class GlobalDataTests(unittest.TestCase):
    def test_actor_key_respects_joint_color_swap_and_all_rectangular_d4_transforms(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        board[1, 2] = BLACK
        board[3, 4] = WHITE
        board[0, 1] = FORBIDDEN
        key = canonical_board_key(board, BLACK)
        self.assertNotEqual(key, canonical_board_key(board, WHITE))
        for turns in range(4):
            rotated = np.rot90(board, turns)
            for transformed in (rotated, np.fliplr(rotated)):
                self.assertEqual(canonical_board_key(transformed, BLACK), key)
                self.assertEqual(canonical_board_key(swap_board_colors(transformed), WHITE), key)
                self.assertEqual(physical_board_key(transformed), physical_board_key(board))
        self.assertNotEqual(physical_board_key(board), physical_board_key(np.pad(board, ((0, 1), (0, 0)))))

    def test_short_real_search_trajectories_have_legal_targets_fixed_masks_and_unknown_truncations(self):
        rows = build_global_dataset(games=3, seed=531, sizes=(6, (6, 7), 8), max_moves=6,
                                    search_time_limit=0, search_max_nodes=0, exploration_rate=0)
        self.assertTrue(rows)
        self.assertEqual(len({row["board_key"] for row in rows}), len(rows))
        masks = {}
        owners = {}
        for row in rows:
            board, side, target = row["board"], row["side"], row["target_policy"]
            self.assertEqual(board_winner(board), 0)
            self.assertEqual(target.shape, board.shape)
            self.assertEqual(target.dtype, np.float32)
            self.assertAlmostEqual(float(target.sum()), 1, places=6)
            self.assertTrue(np.all(target[board != 0] == 0))
            self.assertEqual(board[row["action"]], 0)
            black, white = int((board == BLACK).sum()), int((board == WHITE).sum())
            self.assertTrue(black == white if side == BLACK else black == white + 1)
            self.assertFalse(row["value_valid"])
            self.assertIsNone(row["game_winner"])
            self.assertEqual(row["game_termination"], "move_limit")
            self.assertEqual(row["search_requested_depth"], 3)
            self.assertNotIn("counter_target", row)
            if row["game_id"] in masks:
                np.testing.assert_array_equal(masks[row["game_id"]], board == FORBIDDEN)
            masks[row["game_id"]] = board == FORBIDDEN
            physical = row["physical_key"]
            self.assertEqual(owners.setdefault(physical, row["game_id"]), row["game_id"])
        report = global_dataset_report(rows)
        self.assertEqual(report["games_started"], 3)
        self.assertEqual(report["games_terminal"], 0)
        self.assertEqual(report["games_truncated"], 3)
        self.assertEqual(report["valid_values"], {})
        self.assertEqual(report["invalid_values"], len(rows))

    def test_real_terminal_outcome_is_backfilled_from_each_actual_actor_perspective(self):
        with patch("global_data._start_game", side_effect=forced_win_start):
            rows = build_global_dataset(games=1, seed=15, sizes=(6,), max_moves=9,
                                        search_time_limit=0, search_max_nodes=0, exploration_rate=0)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["side"] for row in rows], [BLACK, WHITE, BLACK])
        self.assertEqual([row["value"] for row in rows], [1, -1, 1])
        self.assertTrue(all(row["value_valid"] and row["value_source"] == "terminal" for row in rows))
        self.assertTrue(all(row["game_winner"] == BLACK for row in rows))
        self.assertEqual(global_dataset_report(rows)["games_terminal"], 1)

    def test_sample_cap_still_finishes_current_game_before_labeling_retained_positions(self):
        with patch("global_data._start_game", side_effect=forced_win_start):
            rows = build_global_dataset(games=4, seed=15, sizes=(6,), max_moves=9, max_positions=1,
                                        search_time_limit=0, search_max_nodes=0, exploration_rate=0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows.games), 1)
        self.assertEqual(rows.games[0]["search_calls"], 3)
        self.assertEqual((rows[0]["value"], rows[0]["value_valid"]), (1, True))

    def test_true_full_board_draw_is_valid_and_distinct_from_truncation(self):
        board = np.full((6, 6), FORBIDDEN, dtype=np.uint8)
        board[0, 0] = BLACK
        board[0, 1] = WHITE
        board[0, 2] = 0
        opening = [(BLACK, (0, 0)), (WHITE, (0, 1))]
        with patch("global_data._start_game", return_value=(board, BLACK, opening, "fixture")):
            rows = build_global_dataset(games=1, sizes=(6,), max_moves=4,
                                        search_time_limit=0, search_max_nodes=0)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["value"], rows[0]["value_valid"], rows[0]["game_termination"]), (0, True, "draw"))

    def test_search_proof_can_label_a_truncated_position_but_unknown_cannot(self):
        def search(board, side, priors, **kwargs):
            point = tuple(map(int, np.argwhere(board == 0)[0]))
            return dict(move=point, proven_value=1, completed_depth=3, nodes=4)
        with patch("global_data.select_move", side_effect=search):
            rows = build_global_dataset(games=1, sizes=(6,), max_moves=2)
        self.assertEqual((rows[0]["value"], rows[0]["value_valid"], rows[0]["value_source"]), (1, True, "search_proof"))
        self.assertFalse(rows[0]["game_terminal"])

    def test_cross_game_physical_duplicates_have_one_owner(self):
        with patch("global_data._start_game", side_effect=forced_win_start):
            rows = build_global_dataset(games=2, seed=31, sizes=(6,), max_moves=9,
                                        search_time_limit=0, search_max_nodes=0, exploration_rate=0)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["game_id"] for row in rows}), 1)
        self.assertEqual(rows.generation["duplicate_physical_owner"], 3)
        self.assertEqual(rows.games[1]["recorded_positions"], 0)

    def test_splits_preserve_game_groups_and_reject_physical_leakage(self):
        rows = build_global_dataset(games=3, seed=63, sizes=(6, 7, 8), max_moves=4,
                                    search_time_limit=0, search_max_nodes=0, exploration_rate=0)
        train, validation = split_global_records(rows, seed=3)
        self.assertTrue(train and validation)
        self.assertFalse({row["game_id"] for row in train} & {row["game_id"] for row in validation})
        self.assertFalse({row["physical_key"] for row in train} & {row["physical_key"] for row in validation})
        duplicate = dict(rows[0], game_id="different-game")
        with self.assertRaisesRegex(ValueError, "multiple game groups"):
            split_global_records([rows[0], duplicate])
        with self.assertRaises(ValueError):
            split_global_records([rows[0]])

    def test_seeded_openings_are_reproducible_with_node_zero_checks(self):
        kwargs = dict(games=2, seed=153, sizes=(6, 7), max_moves=5, search_time_limit=0,
                      search_max_nodes=0, exploration_rate=0.5)
        first, second = build_global_dataset(**kwargs), build_global_dataset(**kwargs)
        self.assertEqual([row["board_key"] for row in first], [row["board_key"] for row in second])
        self.assertEqual([row["action"] for row in first], [row["action"] for row in second])


    def test_progress_callback_runs_once_per_finalized_game_with_cumulative_counts(self):
        events = []
        rows = build_global_dataset(games=3, seed=244, sizes=(6, 7, 8), max_moves=4,
                                    search_time_limit=0, search_max_nodes=0, exploration_rate=0,
                                    progress_callback=events.append)
        self.assertEqual(len(events), 3)
        self.assertEqual([event["games_completed"] for event in events], [1, 2, 3])
        self.assertEqual(events[-1]["records"], len(rows))
        self.assertEqual([event["records"] for event in events],
                         sorted(event["records"] for event in events))
        self.assertTrue(all(event["elapsed_seconds"] >= 0 for event in events))
        for event, game in zip(events, rows.games):
            self.assertEqual(event["game"], game)
        events[0]["game"]["shape"][0] = 999
        self.assertEqual(rows.games[0]["shape"], [6, 6])

    def test_callback_respects_sample_cap_and_no_game_requests(self):
        events = []
        with patch("global_data._start_game", side_effect=forced_win_start):
            rows = build_global_dataset(games=4, seed=15, sizes=(6,), max_moves=9, max_positions=1,
                                        search_time_limit=0, search_max_nodes=0, exploration_rate=0,
                                        progress_callback=events.append)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["game"]["terminal"])
        self.assertEqual(events[0]["records"], 1)
        self.assertTrue(rows[0]["value_valid"])
        build_global_dataset(games=0, progress_callback=events.append)
        self.assertEqual(len(events), 1)
        with self.assertRaises(ValueError):
            build_global_dataset(games=0, progress_callback=7)

    def test_search_proofs_already_use_the_record_actor_perspective(self):
        for side in (BLACK, WHITE):
            for proof in (-1, 0, 1):
                board = np.zeros((6, 6), dtype=np.uint8)
                board[3, 3] = BLACK
                opening = [(BLACK, (3, 3))]
                if side == BLACK:
                    board[0, 0] = WHITE
                    opening.append((WHITE, (0, 0)))
                def search(grid, actor, priors, **kwargs):
                    self.assertEqual(actor, side)
                    point = tuple(map(int, np.argwhere(grid == 0)[0]))
                    return dict(move=point, proven_value=proof, completed_depth=3, nodes=4)
                with self.subTest(side=side, proof=proof), \
                     patch("global_data._start_game", return_value=(board, side, opening, "fixture")), \
                     patch("global_data.select_move", side_effect=search):
                    rows = build_global_dataset(games=1, sizes=(6,), max_moves=len(opening) + 1)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["side"], side)
                self.assertEqual(rows[0]["value"], proof)
                self.assertEqual(rows[0]["search_proven_value"], proof)
                self.assertTrue(rows[0]["value_valid"])
                self.assertEqual(rows[0]["value_source"], "search_proof")
                self.assertFalse(rows[0]["game_terminal"])

    def test_invalid_config_and_invalid_search_actions_are_rejected(self):
        for kwargs in (dict(games=-1), dict(search_depth=2), dict(search_time_limit=float("nan")),
                       dict(exploration_rate=1.1), dict(max_moves=1), dict(sizes=(5,))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_global_dataset(**kwargs)
        with patch("global_data.select_move", return_value=dict(move=(0.5, 1), proven_value=None)):
            with self.assertRaises(ValueError):
                build_global_dataset(games=1, sizes=(6,), max_moves=2)
        self.assertEqual(build_global_dataset(games=0), [])
        self.assertEqual(build_global_dataset(max_positions=0), [])


if __name__ == "__main__":
    unittest.main()
