import unittest
from unittest.mock import patch
import numpy as np

from board_rules import winning_cells
from board_search import MATE, _Search, _BudgetExceeded, select_move


class BoardSearchTests(unittest.TestCase):
    def test_cross_window_immediate_win_precedes_network_preference(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        board[8, 6:10] = 1
        board[8, 5] = 3
        priors = np.zeros(board.shape)
        priors[0, 0] = 1
        result = select_move(board, 1, priors)
        self.assertEqual(result["move"], (8, 10))
        self.assertEqual(result["proven_value"], 1)
        self.assertEqual(result["nodes"], 0)

    def test_mandatory_defense_is_preserved_when_search_has_zero_budget(self):
        board = np.zeros((8, 13), dtype=np.uint8)
        board[2, 3:7] = 2
        board[2, 2] = 3
        result = select_move(board, 1, np.ones(board.shape), max_nodes=0)
        self.assertEqual(result["move"], (2, 7))
        self.assertTrue(result["budget_exhausted"])
        self.assertIsNone(result["proven_value"])

    def test_two_real_winning_cells_are_proved_loss(self):
        board = np.zeros((10, 12), dtype=np.uint8)
        board[4, 3:7] = 2
        result = select_move(board, 1, np.ones(board.shape))
        self.assertIn(result["move"], ((4, 2), (4, 7)))
        self.assertEqual(result["proven_value"], -1)

    def test_forming_open_four_is_a_valid_forcing_win_certificate(self):
        board = np.zeros((7, 11), dtype=np.uint8)
        board[3, 4:7] = 1
        prior = np.zeros(board.shape)
        prior[3, 3] = 1
        result = select_move(board, 1, prior, time_limit=3, depth=2)
        self.assertIn(result["move"], ((3, 3), (3, 7)))
        self.assertEqual(result["proven_value"], 1)

    def test_defends_open_three_before_it_becomes_unstoppable_four(self):
        board = np.zeros((7, 11), dtype=np.uint8)
        board[3, 4:7] = 2
        board[1, 1] = 1
        result = select_move(board, 1, np.ones(board.shape), time_limit=3, depth=2)
        self.assertIn(result["move"], ((3, 3), (3, 7)))

    def test_pruned_heuristic_search_does_not_claim_unbeatable_or_draw(self):
        board = np.zeros((8, 9), dtype=np.uint8)
        board[4, 4] = 1
        original = board.copy()
        result = select_move(board, 2, np.ones(board.shape), depth=2,
                             max_nodes=8, time_limit=2, candidate_width=2)
        self.assertIsNone(result["proven_value"])
        self.assertEqual(board[result["move"]], 0)
        np.testing.assert_array_equal(board, original)
        self.assertLessEqual(result["nodes"], 8)

    def test_prevents_cross_direction_double_three_at_shared_center(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        for point in ((8, 7), (8, 9), (7, 8), (9, 8)):
            board[point] = 2
        board[2, 2] = 1
        result = select_move(board, 1, np.ones(board.shape), depth=2, time_limit=3)
        self.assertEqual(result["move"], (8, 8))

    def test_one_intersection_can_defend_two_open_threes(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        board[8, 5:8] = 2
        board[5:8, 8] = 2
        board[2, 2] = 1
        result = select_move(board, 1, np.ones(board.shape), depth=2, time_limit=3)
        self.assertEqual(result["move"], (8, 8))
        self.assertIsNone(result["proven_value"])

    def test_complete_terminal_boards_return_no_move(self):
        board = np.full((6, 8), 3, dtype=np.uint8)
        result = select_move(board, 1, np.zeros(board.shape))
        self.assertIsNone(result["move"])
        self.assertEqual(result["proven_value"], 0)
        board[2, 1:6] = 2
        result = select_move(board, 1, np.zeros(board.shape))
        self.assertIsNone(result["move"])
        self.assertEqual(result["proven_value"], -1)

    def test_bad_configuration_is_rejected(self):
        board = np.zeros((6, 6), dtype=np.uint8)
        for options in ({"time_limit":float("nan")}, {"max_nodes":-1},
                        {"depth":0}, {"candidate_width":False}):
            with self.assertRaises(ValueError):
                select_move(board, 1, np.ones(board.shape), **options)
        with self.assertRaises(ValueError):
            select_move(board, 1, np.ones((5,5)))
        with self.assertRaises(ValueError):
            select_move(board, True, np.ones(board.shape))



class SearchValueAndForcingTests(unittest.TestCase):
    @staticmethod
    def engine(board, *, value_evaluator=None, max_nodes=100, value_weight=40.0):
        return _Search(board, 1, np.zeros(board.shape), max_nodes, float('inf'), 3,
                       value_evaluator=value_evaluator, value_weight=value_weight)

    @staticmethod
    def mandatory_defense_board():
        board = np.zeros((7, 11), dtype=np.uint8)
        board[1, 2:6] = 2
        board[1, 1] = 3
        return board

    def test_extreme_neural_values_never_become_search_certificates(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        board[2, 3] = 1
        for prediction in (-1., 1.):
            calls = []
            def evaluate(copy, side):
                calls.append(side)
                return prediction
            result = select_move(board, 2, np.ones(board.shape), depth=2, candidate_width=2,
                                 max_nodes=80, time_limit=2, value_weight=MATE * 10,
                                 value_evaluator=evaluate)
            self.assertTrue(calls)
            self.assertIsNone(result['proven_value'])
            self.assertLess(abs(result['heuristic_score']), MATE)
            self.assertEqual(board[result['move']], 0)

    def test_value_cache_and_perspective_include_actual_side(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        calls = []
        def evaluate(copy, side):
            calls.append(side)
            return 1 if side == 1 else -1
        engine = self.engine(board, value_evaluator=evaluate)
        self.assertEqual(engine.evaluate(board, 1), 40.)
        self.assertEqual(engine.evaluate(board, 2), -40.)
        self.assertEqual(engine.evaluate(board, 1), 40.)
        self.assertEqual(calls, [1, 2])

    def test_invalid_values_raise_clear_errors_without_changing_board(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        board[2, 3] = 1
        original = board.copy()
        for value in (float('nan'), float('inf'), -1.01, 1.01, None, [0, 1], True, '0.5'):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(ValueError, 'global value'):
                    select_move(board, 2, np.ones(board.shape), depth=1, candidate_width=1,
                                value_evaluator=lambda b, s: value)
                np.testing.assert_array_equal(board, original)
        for config in ({'value_weight': True}, {'value_weight': -1}, {'value_weight': float('nan')},
                       {'value_evaluator': 1}):
            with self.assertRaises(ValueError):
                select_move(board, 2, np.ones(board.shape), **config)

    def test_slow_value_call_exhausts_budget_and_preserves_fallback(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        board[2, 3] = 1
        original = board.copy()
        clock = [0.]
        def evaluate(copy, side):
            clock[0] = 2.
            return 1.
        with patch('board_search.time.monotonic', side_effect=lambda: clock[0]):
            result = select_move(board, 2, np.ones(board.shape), time_limit=1, depth=1,
                                 candidate_width=1, value_evaluator=evaluate)
        self.assertTrue(result['budget_exhausted'])
        self.assertEqual(result['completed_depth'], 0)
        self.assertEqual(result['value_evaluations'], 1)
        self.assertIsNone(result['proven_value'])
        self.assertEqual(board[result['move']], 0)
        np.testing.assert_array_equal(board, original)

    def test_forcing_upgrade_must_not_ignore_opponent_immediate_win(self):
        board = self.mandatory_defense_board()
        board[4, 3:6] = 1
        original = board.copy()
        engine = self.engine(board)
        # Even an incomplete supplied block list cannot defeat the actual
        # complete-board enemy-win verification inside the upgrade check.
        self.assertIsNone(engine.forcing_move(board, 1, []))
        np.testing.assert_array_equal(board, original)
        result = select_move(board, 1, np.ones(board.shape), max_nodes=1, depth=1)
        self.assertEqual(result['move'], (1, 6))
        self.assertIsNone(result['proven_value'])

    def test_upgrade_can_block_a_four_and_prove_its_own_two_winning_cells(self):
        board = np.zeros((9, 11), dtype=np.uint8)
        board[4, 3:6] = 1
        board[0:4, 6] = 2
        self.assertEqual(winning_cells(board, 2), [(4, 6)])
        original = board.copy()
        result = select_move(board, 1, np.ones(board.shape), depth=1, max_nodes=8)
        self.assertEqual(result['move'], (4, 6))
        self.assertEqual(result['proven_value'], 1)
        after = board.copy()
        after[4, 6] = 1
        self.assertEqual(winning_cells(after, 2), [])
        self.assertEqual(winning_cells(after, 1), [(4, 2), (4, 7)])
        np.testing.assert_array_equal(board, original)

    def test_forcing_checks_are_node_bounded_and_reversible(self):
        board = np.zeros((7, 11), dtype=np.uint8)
        board[3, 4:7] = 1
        original = board.copy()
        result = select_move(board, 1, np.ones(board.shape), max_nodes=0)
        self.assertTrue(result['budget_exhausted'])
        self.assertEqual(result['nodes'], 0)
        self.assertIsNone(result['proven_value'])
        np.testing.assert_array_equal(board, original)
        engine = self.engine(board, max_nodes=1)
        self.assertIsNotNone(engine.forcing_move(board, 1, []))
        self.assertEqual(engine.nodes, 1)
        np.testing.assert_array_equal(board, original)
        with self.assertRaises(_BudgetExceeded):
            engine.forcing_move(board, 1, [])
        np.testing.assert_array_equal(board, original)

    def test_forcing_move_restores_board_even_when_tactic_validation_raises(self):
        board = np.zeros((7, 11), dtype=np.uint8)
        board[3, 4:7] = 1
        original = board.copy()
        engine = self.engine(board)
        with patch.object(engine, 'tactics', side_effect=RuntimeError('validation failed')):
            with self.assertRaisesRegex(RuntimeError, 'validation failed'):
                engine.forcing_move(board, 1, [])
        np.testing.assert_array_equal(board, original)

    def test_leaf_mandatory_defense_extends_before_neural_evaluation(self):
        board = self.mandatory_defense_board()
        original = board.copy()
        seen = []
        def evaluate(copy, side):
            seen.append((copy.copy(), side))
            copy[:] = 3  # Evaluators receive a copy, never the search board.
            return 0.
        engine = self.engine(board, value_evaluator=evaluate)
        score, proof = engine.search(board, 1, 0, -float('inf'), float('inf'))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][1], 2)
        self.assertEqual(seen[0][0][1, 6], 1)
        self.assertGreaterEqual(engine.nodes, 2)
        self.assertIsNone(proof)
        np.testing.assert_array_equal(board, original)

    def test_quiescence_budget_or_callback_failure_unwinds_search_board(self):
        board = self.mandatory_defense_board()
        original = board.copy()
        engine = self.engine(board, max_nodes=1)
        with self.assertRaises(_BudgetExceeded):
            engine.search(board, 1, 0, -float('inf'), float('inf'))
        self.assertEqual(engine.nodes, 1)
        np.testing.assert_array_equal(board, original)
        def fail(copy, side):
            raise RuntimeError('model unavailable')
        engine = self.engine(board, value_evaluator=fail)
        with self.assertRaisesRegex(RuntimeError, 'model unavailable'):
            engine.search(board, 1, 0, -float('inf'), float('inf'))
        np.testing.assert_array_equal(board, original)


if __name__ == "__main__":
    unittest.main()
