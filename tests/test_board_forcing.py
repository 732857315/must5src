"""Proof replay and negative/counterattack cases for bounded continuous-four search."""

import json
import time
import unittest
from unittest.mock import patch

import numpy as np

from board_forcing import solve_forcing, _ForcingSearch, _BudgetExceeded
from board_rules import board_winner, winning_cells, legal_cells, apply_board_move


# First historical board_acceptance loss: white to play before searched ply 41.
HISTORICAL_WHITE_VCF = """0000000000000000
0000000000000000
0000002000000000
0000001000000000
0000021200000000
0012111200000000
0002211120000000
0010102210000000
0021111212000000
0021220000100000
0022210020000000
0020200000000000
0110010000000000
0000000000000000
0000000000000000
0000000000000000"""


def replay(board, side, result):
    position = board.copy()
    actor = side
    for step in result['principal_variation']:
        if step['side'] != actor:
            raise AssertionError('PV colors must alternate from the supplied actor')
        if step['kind'] == 'mandatory_defense':
            assert tuple(step['move']) in winning_cells(position, 3 - actor)
        position = apply_board_move(position, step['move'], actor)
        actor = 3 - actor
    return position


def exact_small(board, side):
    winner = board_winner(board)
    if winner:
        return 1 if winner == side else -1
    moves = legal_cells(board)
    if not moves:
        return 0
    return max(-exact_small(apply_board_move(board, move, side), 3 - side) for move in moves)


class ContinuousFourTests(unittest.TestCase):
    def test_immediate_win_has_legal_color_aware_line(self):
        board = np.zeros((6, 10), dtype=np.uint8)
        board[2, 3:7] = 1
        board[2, 2] = 3
        result = solve_forcing(board, 1)
        self.assertEqual(result['status'], 'win')
        self.assertEqual(result['move'], (2, 7))
        self.assertEqual(board_winner(replay(board, 1, result)), 1)
        json.dumps(result)

    def test_open_four_proof_replays_to_actual_five_and_mirrors_colors(self):
        for side in (1, 2):
            board = np.zeros((7, 11), dtype=np.uint8)
            board[3, 4:7] = side
            original = board.copy()
            result = solve_forcing(board, side, max_depth=1)
            self.assertEqual(result['proven_value'], 1)
            self.assertEqual(result['proof_plies'], 3)
            self.assertEqual(board_winner(replay(board, side, result)), side)
            np.testing.assert_array_equal(board, original)

    def test_immediate_counterwin_precedes_own_three(self):
        board = np.zeros((8, 11), dtype=np.uint8)
        board[2, 3:7] = 2
        board[5, 4:7] = 1
        result = solve_forcing(board, 1)
        self.assertEqual(result['proven_value'], -1)
        self.assertEqual(result['reason'], 'two_independent_enemy_wins')
        self.assertEqual(board_winner(replay(board, 1, result)), 2)

    def test_own_immediate_five_can_outrun_enemy_double_kill(self):
        board = np.zeros((8, 11), dtype=np.uint8)
        board[2, 3:7] = 2
        board[5, 3:7] = 1
        result = solve_forcing(board, 1)
        self.assertEqual(result['proven_value'], 1)
        self.assertEqual(result['proof_plies'], 1)
        self.assertEqual(board_winner(replay(board, 1, result)), 1)

    def test_defense_can_also_create_a_proved_open_four(self):
        board = np.zeros((9, 11), dtype=np.uint8)
        board[4, 3:6] = 1
        board[0:4, 6] = 2
        result = solve_forcing(board, 1, max_depth=1)
        self.assertEqual(result['move'], (4, 6))
        self.assertEqual(result['proven_value'], 1)
        self.assertEqual(board_winner(replay(board, 1, result)), 1)

    def test_forced_defense_chain_can_certify_loss_without_quiet_pruning(self):
        board = np.zeros((9, 11), dtype=np.uint8)
        board[2, 3:7] = 2
        board[2, 2] = 3
        board[6, 4:7] = 2
        result = solve_forcing(board, 1, max_depth=2)
        self.assertEqual(result['move'], (2, 7))
        self.assertEqual(result['proven_value'], -1)
        self.assertEqual(result['reason'], 'forced_defense_chain')
        self.assertEqual(board_winner(replay(board, 1, result)), 2)

    def test_omitted_quiet_defenses_are_unknown_even_with_enemy_open_threes(self):
        board = np.zeros((9, 11), dtype=np.uint8)
        board[2, 4:7] = 2
        board[6, 4:7] = 2
        result = solve_forcing(board, 1, max_depth=16)
        self.assertEqual(result['status'], 'unknown')
        self.assertIsNone(result['proven_value'])
        self.assertIsNone(result['move'])
        self.assertFalse(result['budget_exhausted'])

    def test_forbidden_cells_block_fives_and_forcing_lines(self):
        board = np.full((6, 8), 3, dtype=np.uint8)
        board[2, 1:7] = [1, 1, 3, 1, 0, 0]
        result = solve_forcing(board, 1)
        self.assertEqual(result['status'], 'unknown')
        self.assertIsNone(result['proven_value'])

    def test_terminal_overline_and_full_board_draw(self):
        board = np.zeros((7, 11), dtype=np.uint8)
        board[2, 2:8] = 2
        result = solve_forcing(board, 1)
        self.assertEqual(result['proven_value'], -1)
        self.assertIsNone(result['move'])
        self.assertEqual(result['principal_variation'], [])
        result = solve_forcing(np.full((6, 7), 3, dtype=np.uint8), 1)
        self.assertEqual(result['status'], 'draw')
        self.assertEqual(result['proven_value'], 0)

    def test_node_depth_and_time_limits_never_become_loss_or_draw(self):
        board = np.zeros((7, 11), dtype=np.uint8)
        board[3, 4:7] = 1
        original = board.copy()
        for options in ({'max_nodes': 0}, {'max_nodes': 1}, {'time_limit': 0}, {'max_depth': 0}):
            result = solve_forcing(board, 1, **options)
            self.assertEqual(result['status'], 'unknown')
            self.assertIsNone(result['proven_value'])
            self.assertLessEqual(result['nodes'], result['max_nodes'])
            np.testing.assert_array_equal(board, original)
        result = solve_forcing(board, 1, max_nodes=2)
        self.assertEqual(result['proven_value'], 1)
        self.assertEqual(result['nodes'], 2)

    def test_exception_unwinds_internal_search_board(self):
        board = np.zeros((7, 11), dtype=np.uint8)
        board[3, 4:7] = 1
        original = board.copy()
        engine = _ForcingSearch(board, 100, time.monotonic() + 10)
        original_facts = engine._facts
        calls = [0]
        def fail_on_child(side):
            calls[0] += 1
            if calls[0] == 2:
                raise _BudgetExceeded
            return original_facts(side)
        with patch.object(engine, '_facts', side_effect=fail_on_child):
            with self.assertRaises(_BudgetExceeded):
                engine.search(1, 8)
        np.testing.assert_array_equal(board, original)
        self.assertEqual(engine.cache, {})

    def test_proof_cache_hits_still_consume_nodes(self):
        board = np.zeros((7, 11), dtype=np.uint8)
        board[3, 4:7] = 1
        engine = _ForcingSearch(board, 3, time.monotonic() + 10)
        first = engine.search(1, 8)
        self.assertEqual(engine.nodes, 2)
        self.assertEqual(engine.search(1, 8), first)
        self.assertEqual(engine.nodes, 3)
        self.assertEqual(engine.cache_hits, 1)
        with self.assertRaises(_BudgetExceeded):
            engine.search(1, 8)

    def test_historical_long_forcing_win_and_forced_loss_have_replayable_lines(self):
        board = np.array([[int(cell) for cell in row] for row in HISTORICAL_WHITE_VCF.splitlines()], dtype=np.uint8)
        win = solve_forcing(board, 2, max_nodes=5000, max_depth=16, time_limit=1)
        self.assertEqual(win['proven_value'], 1)
        self.assertGreater(win['proof_plies'], 3)
        self.assertEqual(board_winner(replay(board, 2, win)), 2)
        child = apply_board_move(board, win['move'], 2)
        loss = solve_forcing(child, 1, max_nodes=5000, max_depth=16, time_limit=1)
        self.assertEqual(loss['proven_value'], -1)
        self.assertEqual(board_winner(replay(child, 1, loss)), 2)

    def test_certificates_agree_with_independent_complete_small_endgame_oracle(self):
        rng = np.random.default_rng(914)
        checked = proved = 0
        while checked < 20:
            board = rng.choice([1, 2, 3], size=(6, 7), p=[.4, .4, .2]).astype(np.uint8)
            board.reshape(-1)[rng.choice(board.size, 4, replace=False)] = 0
            if board_winner(board):
                continue
            checked += 1
            for side in (1, 2):
                result = solve_forcing(board, side, max_depth=6, max_nodes=1000, time_limit=1)
                if result['proven_value'] is not None:
                    proved += 1
                    self.assertEqual(result['proven_value'], exact_small(board, side))
        self.assertGreater(proved, 0)

    def test_invalid_boards_and_configuration_are_rejected(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        for options in ({'max_nodes': -1}, {'max_nodes': True}, {'max_depth': 1.5},
                        {'time_limit': float('nan')}, {'time_limit': -1}):
            with self.assertRaises(ValueError):
                solve_forcing(board, 1, **options)
        for side in (0, 3, True, 1.0):
            with self.assertRaises(ValueError):
                solve_forcing(board, side)
        with self.assertRaises(ValueError):
            solve_forcing(np.full((6, 7), 4), 1)


if __name__ == '__main__':
    unittest.main()
