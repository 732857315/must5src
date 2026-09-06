"""Native C proof/coordinate/budget regressions and independent small minimax."""

import ctypes
import json
import time
import unittest
from unittest.mock import patch

import numpy as np

import native_search
from native_search import native_library, native_fingerprint, select_native_move
from board_rules import board_winner, legal_cells, apply_board_move
from tests.test_board_forcing import HISTORICAL_WHITE_VCF, replay


def exact_value(board, side, cache):
    key = (board.tobytes(), side)
    if key in cache:
        return cache[key]
    winner = board_winner(board)
    if winner:
        return 1 if winner == side else -1
    legal = legal_cells(board)
    value = max((-exact_value(apply_board_move(board, move, side), 3 - side, cache) for move in legal), default=0)
    cache[key] = value
    return value


class NativeSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            native_library()
        except RuntimeError as exc:
            if 'requires clang and lld-link' in str(exc):
                raise unittest.SkipTest(str(exc))
            raise

    def test_empty_rectangular_and_single_cell_boards(self):
        for shape in ((1, 1), (6, 7), (7, 6), (16, 16), (64, 64)):
            board = np.zeros(shape, dtype=np.uint8)
            result = select_native_move(board, 1, max_nodes=0)
            self.assertEqual(result['move'], (shape[0] // 2, shape[1] // 2))
            self.assertIsNone(result['proven_value'])
            self.assertEqual(result['nodes'], 0)
            self.assertTrue(result['budget_exhausted'])
            self.assertEqual(np.count_nonzero(board), 0)

    def test_remote_empty_cells_survive_a_frontier_blocked_by_gray(self):
        board = np.full((9, 10), 3, dtype=np.uint8)
        board[0, 0] = 1
        board[8, 9] = 0
        result = select_native_move(board, 2, max_nodes=0)
        self.assertEqual(result['move'], (8, 9))
        result = select_native_move(board, 2, depth=3, max_nodes=100, time_limit=1)
        self.assertEqual(result['move'], (8, 9))
        self.assertEqual(result['proven_value'], 0)

    def test_large_finite_priors_do_not_overflow_before_normalization(self):
        board = np.full((7, 7), 3, dtype=np.uint8)
        board[0, 0] = board[6, 6] = 0
        priors = np.zeros(board.shape)
        priors[0, 0] = 1.
        priors[6, 6] = 1e308
        result = select_native_move(board, 1, priors, max_nodes=0)
        self.assertEqual(result['move'], (6, 6))

    def test_all_four_directions_and_black_white_match_global_coordinates(self):
        fixtures = [((8, 11), [(3, c) for c in range(3, 7)], (3, 2), (3, 7)),
                    ((9, 6), [(r, 3) for r in range(2, 6)], (1, 3), (6, 3)),
                    ((9, 11), [(r, r + 1) for r in range(2, 6)], (1, 2), (6, 7)),
                    ((9, 11), [(r, 8 - r) for r in range(2, 6)], (1, 7), (6, 2))]
        for shape, stones, blocked, target in fixtures:
            for side in (1, 2):
                with self.subTest(shape=shape, side=side, target=target):
                    board = np.zeros(shape, dtype=np.uint8)
                    for stone in stones:
                        board[stone] = side
                    board[blocked] = 3
                    result = select_native_move(board, side, max_nodes=0)
                    self.assertEqual(result['move'], target)
                    self.assertEqual(result['proven_value'], 1)
                    self.assertEqual(board_winner(apply_board_move(board, target, side)), side)

    def test_gray_is_neither_color_and_terminal_overlines_return_no_move(self):
        board = np.full((6, 8), 3, dtype=np.uint8)
        result = select_native_move(board, 1)
        self.assertIsNone(result['move'])
        self.assertEqual(result['proven_value'], 0)
        board[2, 1:7] = 2
        result = select_native_move(board, 1)
        self.assertIsNone(result['move'])
        self.assertEqual(result['proven_value'], -1)
        board[2, 3] = 3
        board[5, 7] = 0
        result = select_native_move(board, 1, max_nodes=0)
        self.assertEqual(result['move'], (5, 7))
        self.assertIsNone(result['proven_value'])

    def test_mandatory_defense_and_counterwin_precede_own_open_three(self):
        board = np.zeros((8, 11), dtype=np.uint8)
        board[2, 3:7] = 2
        board[2, 2] = 3
        board[5, 4:7] = 1
        result = select_native_move(board, 1, depth=1, max_nodes=30)
        self.assertEqual(result['move'], (2, 7))
        self.assertIsNone(result['proven_value'])
        board[2, 2] = 0
        result = select_native_move(board, 1)
        self.assertEqual(result['proven_value'], -1)
        board[5, 3] = 1
        result = select_native_move(board, 1)
        self.assertEqual(result['proven_value'], 1)
        self.assertEqual(board_winner(apply_board_move(board, result['move'], 1)), 1)

    def test_verified_upgrade_can_simultaneously_block_and_win(self):
        board = np.zeros((9, 11), dtype=np.uint8)
        board[4, 3:6] = 1
        board[0:4, 6] = 2
        result = select_native_move(board, 1, depth=1, max_nodes=100)
        self.assertEqual(result['move'], (4, 6))
        self.assertEqual(result['proven_value'], 1)

    def test_beam_and_node_exhaustion_cannot_become_proofs(self):
        board = np.zeros((8, 9), dtype=np.uint8)
        board[4, 4] = 1
        original = board.copy()
        for nodes in (0, 1, 7, 32):
            result = select_native_move(board, 2, depth=9, candidate_width=1, max_nodes=nodes, time_limit=2)
            self.assertLessEqual(result['nodes'], nodes)
            self.assertIsNone(result['proven_value'])
            self.assertEqual(board[result['move']], 0)
            np.testing.assert_array_equal(board, original)

    def test_zero_time_uses_legal_fallback_without_moving_input(self):
        board = np.zeros((10, 16), dtype=np.uint8)
        board[5, 8] = 1
        original = board.copy()
        result = select_native_move(board, 2, time_limit=0, max_nodes=10000)
        self.assertTrue(result['budget_exhausted'])
        self.assertEqual(result['completed_depth'], 0)
        self.assertEqual(result['nodes'], 0)
        self.assertIsNone(result['proven_value'])
        np.testing.assert_array_equal(board, original)

    def test_callback_failure_propagates_and_next_search_resets_context(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        board[2, 3] = 1
        original = board.copy()
        count = [0]
        def broken_clock():
            count[0] += 1
            if count[0] >= 3:
                raise RuntimeError('clock unavailable')
            return 0.
        with patch('native_search.time.monotonic', side_effect=broken_clock):
            with self.assertRaisesRegex(RuntimeError, 'clock unavailable'):
                select_native_move(board, 2, depth=3)
        np.testing.assert_array_equal(board, original)
        result = select_native_move(board, 2, depth=1, time_limit=1)
        self.assertEqual(board[result['move']], 0)

    def test_same_thread_context_reuses_different_board_sizes_safely(self):
        for shape in ((16, 16), (1, 2), (7, 13), (64, 64), (6, 7)):
            board = np.zeros(shape, dtype=np.uint8)
            result = select_native_move(board, 1, max_nodes=2, depth=1)
            self.assertEqual(board[result['move']], 0)
            self.assertLessEqual(result['nodes'], 2)

    def test_native_proofs_agree_with_independent_four_empty_minimax(self):
        rng = np.random.default_rng(9513)
        tested = proved = 0
        while tested < 20:
            board = rng.choice([1, 2, 3], size=(6, 7), p=[.45, .45, .1]).astype(np.uint8)
            board.reshape(-1)[rng.choice(board.size, 4, replace=False)] = 0
            if board_winner(board):
                continue
            tested += 1
            for side in (1, 2):
                original = board.copy()
                exact = exact_value(board, side, {})
                result = select_native_move(board, side, max_nodes=10000, time_limit=1, depth=6, candidate_width=64)
                if result['proven_value'] is not None:
                    proved += 1
                    self.assertEqual(result['proven_value'], exact)
                if result['proven_value'] == 1:
                    child = apply_board_move(board, result['move'], side)
                    self.assertEqual(-exact_value(child, 3 - side, {}), 1)
                np.testing.assert_array_equal(board, original)
        self.assertGreater(proved, 10)

    def test_optional_vcf_proof_preserves_real_winning_line(self):
        board = np.array([[int(x) for x in row] for row in HISTORICAL_WHITE_VCF.splitlines()], dtype=np.uint8)
        result = select_native_move(board, 2, time_limit=0, max_nodes=0,
                                    forcing_seconds=.2, forcing_nodes=5000, forcing_depth=16)
        self.assertEqual(result['proven_value'], 1)
        self.assertIsNotNone(result['forcing'])
        self.assertGreater(result['forcing']['proof_plies'], 3)
        self.assertEqual(board_winner(replay(board, 2, result['forcing'])), 2)

    def test_invalid_sizes_probabilities_and_configuration(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        for options in ({'max_nodes': True}, {'max_nodes': 2**31}, {'depth': 59}, {'candidate_width': 65},
                        {'time_limit': float('nan')}, {'forcing_seconds': -1},
                        {'forcing_nodes': -1}, {'forcing_depth': True}):
            with self.assertRaises(ValueError):
                select_native_move(board, 1, **options)
        for priors in (np.zeros((7, 6)), np.full(board.shape, float('inf')), np.full(board.shape, -.1)):
            with self.assertRaises(ValueError):
                select_native_move(board, 1, priors)
        with self.assertRaises(ValueError):
            select_native_move(np.zeros((65, 6), dtype=np.uint8), 1)
        with self.assertRaises(ValueError):
            select_native_move(board, True)

    def test_build_fingerprint_is_serializable_and_versioned(self):
        metadata = native_fingerprint()
        self.assertEqual(len(metadata['source_sha256']), 64)
        self.assertEqual(len(metadata['binary_sha256']), 64)
        self.assertTrue(metadata['binary_path'].endswith('.dll'))
        json.dumps(metadata)


if __name__ == '__main__':
    unittest.main()
