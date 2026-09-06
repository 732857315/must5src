import unittest

import numpy as np

from board_rules import (BLACK, WHITE, FORBIDDEN, normalize_board, board_winner,
                         legal_cells, winning_cells, apply_board_move,
                         board_move_rejection_reason, tactical_candidates, swap_board_colors)


def independent_winner(board):
    height, width = board.shape
    for row in range(height):
        for col in range(width):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = []
                for step in range(5):
                    r, c = row + step * dr, col + step * dc
                    if not (0 <= r < height and 0 <= c < width):
                        break
                    cells.append(int(board[r, c]))
                if len(cells) == 5 and cells[0] in (BLACK, WHITE) and len(set(cells)) == 1:
                    return cells[0]
    return 0


class BoardRulesTests(unittest.TestCase):
    def test_normalization_copies_rectangular_board_and_checks_encoding(self):
        original = np.zeros((7, 11), dtype=np.int64)
        original[1, 4] = FORBIDDEN
        board = normalize_board(original)
        self.assertEqual(board.shape, (7, 11))
        self.assertEqual(board.dtype, np.uint8)
        board[1, 4] = BLACK
        self.assertEqual(original[1, 4], FORBIDDEN)
        for invalid in ([], [0, 1], [[0], [1, 2]], [[4]], [[-1]], [[1.0]], [[True]], np.zeros((0, 5), dtype=int)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_board(invalid)

    def test_five_across_window_boundaries_in_every_direction_and_six_also_wins(self):
        for side in (BLACK, WHITE):
            for start, direction in (((7, 3), (0, 1)), ((3, 7), (1, 0)),
                                     ((3, 3), (1, 1)), ((3, 11), (1, -1))):
                board = np.zeros((16, 16), dtype=np.uint8)
                for step in range(6):
                    r = start[0] + step * direction[0]
                    c = start[1] + step * direction[1]
                    board[r, c] = side
                    self.assertEqual(board_winner(board), side if step >= 4 else 0)

    def test_forbidden_cells_break_lines_and_never_count_as_stones(self):
        board = np.full((16, 16), FORBIDDEN, dtype=np.uint8)
        self.assertEqual(board_winner(board), 0)
        self.assertEqual(legal_cells(board), [])
        self.assertEqual(winning_cells(board, BLACK), [])
        board = np.zeros((16, 16), dtype=np.uint8)
        board[6, 3:9] = BLACK
        board[6, 6] = FORBIDDEN
        self.assertEqual(board_winner(board), 0)
        self.assertNotIn((6, 6), winning_cells(board, BLACK))
        self.assertEqual(winning_cells(board, BLACK), [])

    def test_board_edges_never_wrap_into_the_next_row_or_diagonal(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        board[2, 14:16] = BLACK
        board[3, 0:3] = BLACK
        board[4, 14] = WHITE
        board[5, 15] = WHITE
        board[6, 0] = WHITE
        board[7, 1] = WHITE
        board[8, 2] = WHITE
        self.assertEqual(board_winner(board), 0)

    def test_winning_cell_joins_both_sides_into_a_six(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        board[8, 4:7] = WHITE
        board[8, 8:10] = WHITE
        self.assertEqual(winning_cells(board, WHITE), [(8, 7)])
        changed = apply_board_move(board, (8, 7), WHITE)
        self.assertEqual(board_winner(changed), WHITE)
        self.assertEqual(board[8, 7], 0)

    def test_two_opponent_winning_cells_are_distinct_blocking_threats(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        for row in (4, 9):
            board[row, 4:8] = WHITE
            board[row, 3] = BLACK
        wins, blocks, rest = tactical_candidates(board, BLACK)
        self.assertEqual(wins, [])
        self.assertEqual(blocks, [(4, 8), (9, 8)])
        self.assertEqual(len(set(wins + blocks + rest)), len(legal_cells(board)))
        self.assertEqual(winning_cells(board, WHITE), blocks)

    def test_game_over_has_no_legal_or_tactical_moves(self):
        board = np.zeros((7, 9), dtype=np.uint8)
        board[0, :5] = BLACK
        self.assertEqual(legal_cells(board), [])
        self.assertEqual(winning_cells(board, WHITE), [])
        self.assertEqual(tactical_candidates(board, BLACK), ([], [], []))
        with self.assertRaisesRegex(ValueError, "棋局已结束"):
            apply_board_move(board, (4, 4), WHITE)

    def test_illegal_move_reasons_are_specific_and_original_is_unchanged(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        board[0, 0] = BLACK
        board[1, 1] = FORBIDDEN
        original = board.copy()
        for move, reason in (((0, 0), "已有棋子"), ((1, 1), "边界或禁下格"),
                             ((16, 0), "越界"), ((-1, 0), "越界"),
                             ((True, 1), "坐标无效"), ((1.0, 2), "坐标无效"), (None, "坐标无效")):
            self.assertIn(reason, board_move_rejection_reason(board, move))
            with self.assertRaisesRegex(ValueError, reason):
                apply_board_move(board, move, WHITE)
        np.testing.assert_array_equal(board, original)
        for side in (0, 3, True, 1.0, None):
            with self.assertRaises(ValueError):
                apply_board_move(board, (3, 3), side)

    def test_no_opening_center_is_forced_and_numpy_integer_coordinates_work(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        self.assertEqual(len(legal_cells(board)), 256)
        moved = apply_board_move(board, (np.int64(0), np.int64(15)), np.int64(BLACK))
        self.assertEqual(moved[0, 15], BLACK)
        self.assertEqual(int(board.sum()), 0)
        singleton = apply_board_move([[0]], (0, 0), WHITE)
        self.assertEqual(legal_cells(singleton), [])
        self.assertEqual(board_winner(singleton), 0)

    def test_color_swap_preserves_mask_and_empty_cells(self):
        board = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.uint8)
        swapped = swap_board_colors(board)
        np.testing.assert_array_equal(swapped, [[0, 2, 1, 3], [3, 1, 2, 0]])
        np.testing.assert_array_equal(swap_board_colors(swapped), board)

    def test_global_winning_moves_match_independent_five_cell_enumeration(self):
        rng = np.random.default_rng(780)
        for _ in range(8):
            board = rng.choice(4, size=(6, 7), p=(0.55, 0.20, 0.20, 0.05)).astype(np.uint8)
            if independent_winner(board):
                continue
            self.assertEqual(board_winner(board), 0)
            for side in (BLACK, WHITE):
                expected = []
                for row, col in np.argwhere(board == 0):
                    trial = board.copy()
                    trial[row, col] = side
                    if independent_winner(trial) == side:
                        expected.append((int(row), int(col)))
                self.assertEqual(winning_cells(board, side), expected)


if __name__ == "__main__":
    unittest.main()
