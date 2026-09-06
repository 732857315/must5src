import json
import unittest

import numpy as np

from board_rules import BLACK, WHITE, FORBIDDEN, apply_board_move, winning_cells, swap_board_colors
from board_threats import analyze_threats


class BoardThreatTests(unittest.TestCase):
    def test_one_open_four_is_one_core_with_two_completions(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        board[7, 6:10] = BLACK
        report = analyze_threats(board, BLACK)
        self.assertEqual(report["four_count"], 1)
        self.assertFalse(report["double_four"])
        self.assertTrue(report["open_four"])
        self.assertTrue(report["double_kill"])
        self.assertEqual(report["winning_cells"], [(7, 5), (7, 10)])
        self.assertEqual(report["four_defense_cells"], [])
        self.assertEqual(report["fours"][0]["stones"], [(7, 6), (7, 7), (7, 8), (7, 9)])

    def test_adjacent_windows_merge_one_contiguous_three_core(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        board[7, 6:9] = BLACK
        report = analyze_threats(board, BLACK)
        self.assertEqual(report["three_count"], 1)
        self.assertFalse(report["double_three"])
        self.assertEqual(report["threes"][0]["extension_cells"], [(7, 5), (7, 9)])

    def test_broken_three_has_its_internal_extension(self):
        board = np.zeros((8, 11), dtype=np.uint8)
        board[4, (3, 4, 6)] = WHITE
        report = analyze_threats(board, WHITE)
        self.assertEqual(report["three_count"], 1)
        self.assertEqual(report["threes"][0]["extension_cells"], [(4, 5)])
        grown = analyze_threats(apply_board_move(board, (4, 5), WHITE), WHITE)
        self.assertTrue(grown["open_four"])

    def test_cross_window_crossing_cores_form_structural_double_three(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        board[7, 6:9] = BLACK
        board[6:9, 7] = BLACK
        report = analyze_threats(board, BLACK)
        self.assertEqual(report["three_count"], 2)
        self.assertTrue(report["double_three"])
        self.assertEqual({tuple(item["direction"]) for item in report["threes"]}, {(0, 1), (1, 0)})
        self.assertEqual(report["winning_cells"], [])
        self.assertTrue(report["structural_only"])

    def test_crossing_fours_are_counted_by_global_core(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        board[7, 6:10] = WHITE
        board[6:10, 7] = WHITE
        report = analyze_threats(board, WHITE)
        self.assertEqual(report["four_count"], 2)
        self.assertTrue(report["double_four"])
        self.assertEqual(len(report["winning_cells"]), 4)
        self.assertEqual(report["four_defense_cells"], [])

    def test_two_directions_with_one_shared_winning_cell_have_one_defense(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        for r, c in ((7, 5), (7, 6), (7, 8), (7, 9), (5, 7), (6, 7), (8, 7), (9, 7)):
            board[r, c] = BLACK
        report = analyze_threats(board, BLACK)
        self.assertEqual(report["four_count"], 2)
        self.assertTrue(report["double_four"])
        self.assertFalse(report["double_kill"])
        self.assertEqual(report["winning_cells"], [(7, 7)])
        self.assertEqual(report["shared_completion_cells"], [(7, 7)])
        self.assertEqual(report["four_defense_cells"], [(7, 7)])
        defended = apply_board_move(board, (7, 7), WHITE)
        self.assertEqual(analyze_threats(defended, BLACK)["winning_cells"], [])

    def test_distinct_same_direction_cores_can_also_share_one_kill_point(self):
        board = np.zeros((7, 12), dtype=np.uint8)
        board[3, (3, 4, 5, 7, 8)] = WHITE
        report = analyze_threats(board, WHITE)
        self.assertEqual(report["four_count"], 2)
        self.assertEqual(report["winning_cells"], [(3, 6)])
        self.assertFalse(report["double_kill"])
        self.assertEqual(report["four_defense_cells"], [(3, 6)])

    def test_boundaries_forbidden_and_enemy_cells_break_open_three(self):
        for blocker in (WHITE, FORBIDDEN):
            board = np.zeros((8, 10), dtype=np.uint8)
            board[4, 3:6] = BLACK
            board[4, 2] = blocker
            self.assertEqual(analyze_threats(board, BLACK)["three_count"], 0)
        board = np.zeros((6, 7), dtype=np.uint8)
        board[0, 0:3] = BLACK
        self.assertEqual(analyze_threats(board, BLACK)["three_count"], 0)
        board = np.zeros((6, 7), dtype=np.uint8)
        board[4, (0, 1, 2, 4)] = BLACK
        board[4, 3] = FORBIDDEN
        self.assertEqual(analyze_threats(board, BLACK)["four_count"], 0)

    def test_diagonal_patterns_do_not_wrap_around_board_edges(self):
        board = np.zeros((7, 8), dtype=np.uint8)
        for row, col in ((1, 6), (2, 7), (3, 0), (4, 1)):
            board[row, col] = BLACK
        report = analyze_threats(board, BLACK)
        self.assertEqual((report["four_count"], report["three_count"]), (0, 0))

    def test_analysis_is_color_symmetric_json_serializable_and_does_not_mutate(self):
        board = np.zeros((9, 13), dtype=np.uint8)
        board[3, 4:8] = BLACK
        board[4:7, 10] = WHITE
        board[0, :] = FORBIDDEN
        saved = board.copy()
        black = analyze_threats(board, BLACK)
        white = analyze_threats(swap_board_colors(board), WHITE)
        for key in ("fours", "threes", "winning_cells", "four_defense_cells"):
            self.assertEqual(black[key], white[key])
        json.dumps(black)
        np.testing.assert_array_equal(board, saved)


    def test_one_three_two_upgrade_paths_have_two_adjacent_single_move_defenses(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        board[7, 6:9] = BLACK
        report = analyze_threats(board, BLACK)
        self.assertEqual(report["three_count"], 1)
        self.assertFalse(report["double_three"])
        self.assertEqual(len(report["threes"][0]["upgrade_paths"]), 2)
        self.assertEqual(report["threes"][0]["defense_cells"], [(7, 5), (7, 9)])
        self.assertEqual(report["three_defense_cells"], [(7, 5), (7, 9)])
        blockers = [path["blockers"] for path in report["threes"][0]["upgrade_paths"]]
        self.assertCountEqual(blockers, [[(7, 4), (7, 5), (7, 9)],
                                        [(7, 5), (7, 9), (7, 10)]])
        for move in report["three_defense_cells"]:
            self.assertEqual(analyze_threats(apply_board_move(board, move, WHITE), BLACK)["three_count"], 0)

    def test_crossed_broken_threes_share_a_single_defensive_key_point(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        for row, col in ((7, 5), (7, 6), (7, 8), (5, 7), (6, 7), (8, 7)):
            board[row, col] = WHITE
        report = analyze_threats(board, WHITE)
        self.assertTrue(report["double_three"])
        self.assertEqual(report["three_count"], 2)
        self.assertEqual(report["three_defense_cells"], [(7, 7)])
        defended = analyze_threats(apply_board_move(board, (7, 7), BLACK), WHITE)
        self.assertEqual(defended["three_count"], 0)

    def test_forbidden_far_endpoint_removes_only_its_upgrade_path(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        board[7, 6:9] = BLACK
        board[7, 4] = FORBIDDEN
        report = analyze_threats(board, BLACK)
        self.assertEqual(report["three_count"], 1)
        self.assertEqual(report["threes"][0]["extension_cells"], [(7, 9)])
        self.assertEqual(len(report["threes"][0]["upgrade_paths"]), 1)
        self.assertEqual(report["three_defense_cells"], [(7, 5), (7, 9), (7, 10)])
        self.assertEqual(report["threes"][0]["upgrade_paths"][0]["empty_endpoints"], [(7, 5), (7, 10)])

    def test_three_defense_intersection_matches_simulated_blocking_moves(self):
        for crossed in (False, True):
            board = np.zeros((7, 9), dtype=np.uint8)
            board[3, (2, 3, 5)] = BLACK
            if crossed:
                board[(1, 2, 4), 4] = BLACK
            before = analyze_threats(board, BLACK)
            expected = []
            for row, col in np.argwhere(board == 0):
                move = (int(row), int(col))
                defended = analyze_threats(apply_board_move(board, move, WHITE), BLACK)
                if defended["three_count"] == 0:
                    expected.append(move)
            self.assertEqual(before["three_defense_cells"], expected)

    def test_terminal_board_has_no_actionable_threats(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        board[5, 3:8] = BLACK
        report = analyze_threats(board, WHITE)
        self.assertTrue(report["terminal"])
        self.assertEqual(report["winner"], BLACK)
        self.assertEqual((report["fours"], report["threes"], report["winning_cells"]), ([], [], []))
        for side in (True, 0, 3, 1.0):
            with self.assertRaises(ValueError):
                analyze_threats(board, side)

    def test_all_immediate_winning_cells_match_the_full_board_rule(self):
        rng = np.random.default_rng(890)
        for _ in range(8):
            board = rng.choice(4, size=(9, 11), p=(0.55, 0.2, 0.2, 0.05)).astype(np.uint8)
            for side in (BLACK, WHITE):
                self.assertEqual(analyze_threats(board, side)["winning_cells"], winning_cells(board, side))


if __name__ == "__main__":
    unittest.main()
