import unittest
from unittest.mock import patch

import numpy as np

from game import BLACK, WHITE, EMPTY, FORBIDDEN, CENTER_INDEX
from windows import extract_window, format_grid, iter_windows, pad_board, validate_grid
from window_inference import infer_window, predict_board


class WindowGeometryTests(unittest.TestCase):
    def test_padding_uses_two_bit_forbidden_value(self):
        board = [[0] * 5 for _ in range(5)]
        board[2][2] = BLACK
        padded = pad_board(board)
        self.assertEqual((len(padded), len(padded[0])), (9, 9))
        for row in range(9):
            for col in range(9):
                self.assertEqual(padded[row][col], board[row - 2][col - 2]
                                 if 2 <= row < 7 and 2 <= col < 7 else FORBIDDEN)
        self.assertEqual(len(pad_board(board, width=1)), 7)
        self.assertEqual(pad_board(board, width=0), validate_grid(board))

    def test_corner_window_has_sixteen_forbidden_cells(self):
        board = [[0] * 5 for _ in range(5)]
        board[0][1] = BLACK
        board[4][4] = WHITE
        window = extract_window(board, 0, 0)
        self.assertEqual(sum(cell == FORBIDDEN for row in window.grid for cell in row), 16)
        self.assertEqual(window.grid[2][3], BLACK)
        self.assertFalse(any(cell == WHITE for row in window.grid for cell in row))
        self.assertEqual(window.to_global(CENTER_INDEX), (0, 0))
        self.assertEqual(window.to_global(np.int64(CENTER_INDEX)), (0, 0))
        self.assertEqual(window.to_global(24), (2, 2))
        self.assertIn('禁下', window.move_hint(0))
        with self.assertRaisesRegex(ValueError, '禁下'):
            window.to_global(0)
        with self.assertRaisesRegex(ValueError, '禁下'):
            window.to_global(13)  # Existing black stone.

    def test_all_edges_match_explicit_padded_slices(self):
        board = [[0] * 8 for _ in range(6)]
        board[0][0], board[5][7], board[2][4] = BLACK, WHITE, FORBIDDEN
        padded = pad_board(board)
        windows = list(iter_windows(board))
        self.assertEqual(len(windows), 48)
        covered = set()
        for window in windows:
            row, col = window.center
            expected = tuple(tuple(padded[r][col:col + 5]) for r in range(row, row + 5))
            self.assertEqual(window.grid, expected)
            self.assertEqual(window.coordinates[CENTER_INDEX], (row, col))
            covered.update(coordinate for coordinate in window.coordinates if coordinate is not None)
        self.assertEqual(covered, {(r, c) for r in range(6) for c in range(8)})

    def test_single_cell_board_is_still_a_five_by_five_window(self):
        window = extract_window([[EMPTY]], 0, 0)
        self.assertEqual(window.to_global(CENTER_INDEX), (0, 0))
        self.assertEqual(sum(c is None for c in window.coordinates), 24)

    def test_invalid_grids_and_centers_do_not_wrap(self):
        for board in ([], [[]], [[0, 1], [0]], [[4]], [[-1]], [[True]]):
            with self.subTest(board=board), self.assertRaises(ValueError):
                validate_grid(board)
        for row, col in ((-1, 0), (0, -1), (2, 0), (0, 2), (True, 0)):
            with self.subTest(center=(row, col)), self.assertRaises(ValueError):
                extract_window([[0, 0], [0, 0]], row, col)
        for width in (-1, True, 0.5):
            with self.assertRaises(ValueError):
                pad_board([[0]], width)

    def test_text_has_forbidden_marker_and_legend(self):
        text = format_grid(pad_board([[BLACK, WHITE, EMPTY, FORBIDDEN]], 1))
        self.assertIn('#', text)
        self.assertIn('边界/禁下', text)
        self.assertIn('2bit=11', text)


class WindowInferenceTests(unittest.TestCase):
    @staticmethod
    def fake_network(net, x):
        # Deliberately prefer forbidden padding and occupied cells above all empties.
        policy = np.where(x.reshape(-1) != EMPTY, 1e6, 0).astype(np.float32)
        return policy, np.array([0.0], dtype=np.float32)

    def test_local_logits_exclude_padding_and_occupied_cells(self):
        window = extract_window([[BLACK, EMPTY], [EMPTY, FORBIDDEN]], 0, 1)
        with patch('window_inference.ncnn_infer', side_effect=self.fake_network):
            policy, value = infer_window(None, window, WHITE)
        legal = [i for i, c in enumerate(sum(window.grid, ())) if c == EMPTY]
        self.assertEqual(set(np.flatnonzero(np.isfinite(policy))), set(legal))
        self.assertEqual(value, 0.0)

    def test_all_real_empty_cells_are_covered_and_boundaries_never_selected(self):
        board = [[0] * 8 for _ in range(6)]
        board[0][0], board[1][3], board[5][7] = BLACK, WHITE, FORBIDDEN
        with patch('window_inference.ncnn_infer', side_effect=self.fake_network) as network:
            result = predict_board(None, board, WHITE)
        self.assertEqual(network.call_count, 48)
        self.assertEqual(result.window_count, 48)
        self.assertEqual(result.policy.shape, (6, 8))
        self.assertAlmostEqual(result.policy.sum(), 1.0)
        self.assertTrue(np.all(result.policy[~result.legal_mask] == 0))
        self.assertTrue(np.all(result.policy[result.legal_mask] > 0))
        self.assertEqual(board[result.move[0]][result.move[1]], EMPTY)
        self.assertFalse(result.terminal)

    def test_global_turn_is_explicit_and_not_inferred_from_window_counts(self):
        board = [[BLACK, EMPTY], [EMPTY, EMPTY]]
        seen = []
        def capture(net, x):
            seen.append(x.copy())
            return self.fake_network(net, x)
        with patch('window_inference.ncnn_infer', side_effect=capture):
            predict_board(None, board, WHITE)
        self.assertTrue(all(2.0 in x and 3.0 in x for x in seen))
        self.assertTrue(all(1.0 not in x for x in seen))
        with self.assertRaises(ValueError):
            predict_board(None, board, FORBIDDEN)

    def test_terminal_global_line_skips_inference(self):
        board = [[0] * 8 for _ in range(7)]
        board[6][1:6] = [WHITE] * 5
        with patch('window_inference.ncnn_infer') as network:
            result = predict_board(None, board, BLACK)
        network.assert_not_called()
        self.assertIsNone(result.move)
        self.assertEqual(result.window_count, 0)
        self.assertEqual(result.policy.sum(), 0)
        self.assertFalse(result.legal_mask.any())
        self.assertTrue(result.terminal)

    def test_all_forbidden_board_has_no_move(self):
        with patch('window_inference.ncnn_infer') as network:
            result = predict_board(None, [[FORBIDDEN] * 3 for _ in range(2)], BLACK)
        network.assert_not_called()
        self.assertIsNone(result.move)


if __name__ == '__main__':
    unittest.main()
