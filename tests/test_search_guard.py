"""Exact child-proof filtering, shared budgets, and a real missed-defense case."""

import unittest
from unittest.mock import patch

import numpy as np

from board_rules import board_winner, apply_board_move
from board_forcing import solve_forcing
from board_threat_search import _ordered_moves
from search_guard import select_guarded_move
from tests.test_board_forcing import HISTORICAL_WHITE_VCF
import tests.test_board_threat_search as threat_tests


def initial(move, *, value=None, nodes=0, forcing=None):
    return dict(move=move, proven_value=value, nodes=nodes, completed_depth=7,
                forcing=forcing, budget_exhausted=False)


def reply(value, nodes=1):
    return dict(proven_value=value, nodes=nodes, principal_variation=[],
                status='unknown' if value is None else 'win' if value == 1 else 'loss',
                budget_exhausted=False)


class GuardTests(unittest.TestCase):
    def test_existing_native_win_bypasses_opponent_probes(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        board[2, 1:5] = 1
        with patch('search_guard.select_native_move', return_value=initial((2, 5), value=1)), \
             patch('search_guard.solve_forcing') as solver:
            result = select_guarded_move(board, 1)
        solver.assert_not_called()
        self.assertEqual(result['proven_value'], 1)
        self.assertEqual(result['move'], (2, 5))

    def test_child_proved_loss_establishes_a_real_parent_win(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        board[2, 1:5] = 1
        with patch('search_guard.select_native_move', return_value=initial((2, 5))):
            result = select_guarded_move(board, 1)
        self.assertEqual(result['proven_value'], 1)
        self.assertEqual(result['status'], 'child_forced_win')
        after = apply_board_move(board, result['move'], 1)
        self.assertEqual(board_winner(after), 1)

    def test_all_legal_children_must_be_certified_before_parent_loss(self):
        board = np.array([[0, 2, 2, 2, 2, 0]], dtype=np.uint8)
        with patch('search_guard.select_native_move', return_value=initial((0, 0), value=-1)):
            result = select_guarded_move(board, 1, time_limit=1, max_nodes=20, native_fraction=0)
        self.assertEqual(result['proven_value'], -1)
        self.assertEqual(result['status'], 'all_moves_proved_losing')
        self.assertEqual(len(result['guard_probes']), 2)
        self.assertEqual(result['unexamined_count'], 0)
        self.assertTrue(all(probe['opponent_result']['proven_value'] == 1 for probe in result['guard_probes']))
        position = board.copy()
        for step in result['principal_variation']:
            position = apply_board_move(position, step['move'], step['side'])
        self.assertEqual(board_winner(position), 2)

    def test_partial_losing_audit_does_not_propagate_initial_negative_certificate(self):
        board = np.array([[0, 2, 2, 2, 2, 0]], dtype=np.uint8)
        with patch('search_guard.select_native_move', return_value=initial((0, 0), value=-1)):
            result = select_guarded_move(board, 1, time_limit=1, max_nodes=1, native_fraction=0)
        self.assertIsNone(result['proven_value'])
        self.assertEqual(result['move'], (0, 5))
        self.assertEqual(result['status'], 'unexamined_budget_fallback')
        self.assertEqual(result['nodes'], 1)
        self.assertTrue(result['guard_changed'])
        self.assertEqual(result['completed_depth'], 0)

    def test_regression_rejects_8_1_and_keeps_unproved_12_5(self):
        board = np.array([[int(cell) for cell in row] for row in HISTORICAL_WHITE_VCF.splitlines()], dtype=np.uint8)
        board[12, 5] = 0
        board[10, 5] = 0
        board[10, 1] = 1
        original = board.copy()
        with patch('search_guard.select_native_move', return_value=initial((8, 1))):
            result = select_guarded_move(board, 1, time_limit=1, max_nodes=50000,
                                         probe_time_limit=.3, probe_max_nodes=10000, forcing_depth=32)
        self.assertEqual(result['move'], (12, 5))
        self.assertIn((8, 1), result['rejected_moves'])
        self.assertIsNone(result['proven_value'])
        self.assertEqual(result['status'], 'opponent_not_proved_winning')
        self.assertIsNone(result['selected_reply']['opponent_result']['proven_value'])
        self.assertEqual(result['completed_depth'], 0)
        for probe in result['guard_probes']:
            if probe['move'] == (8, 1):
                position = apply_board_move(board, (8, 1), 1)
                for step in probe['opponent_result']['principal_variation']:
                    position = apply_board_move(position, step['move'], step['side'])
                self.assertEqual(board_winner(position), 2)
        np.testing.assert_array_equal(board, original)

    def test_actual_last_empty_draw_stays_conservative_and_is_not_rejected(self):
        board = np.array([[1, 2, 1, 2, 0]], dtype=np.uint8)
        original = board.copy()
        with patch('search_guard.select_native_move', return_value=initial((0, 4))):
            result = select_guarded_move(board, 1, time_limit=1)
        self.assertEqual(result['move'], (0, 4))
        self.assertEqual(result['selected_reply']['opponent_result']['proven_value'], 0)
        self.assertIsNone(result['proven_value'])
        self.assertEqual(result['rejected_moves'], [])
        self.assertEqual(result['unexamined_count'], 0)
        self.assertEqual(board_winner(apply_board_move(board, (0, 4), 1)), 0)
        np.testing.assert_array_equal(board, original)

    def test_unknown_never_becomes_safe_draw_or_win(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        with patch('search_guard.select_native_move', return_value=initial((3, 3))), \
             patch('search_guard.solve_forcing', return_value=reply(None)):
            result = select_guarded_move(board, 1)
        self.assertIsNone(result['proven_value'])
        self.assertEqual(result['status'], 'opponent_not_proved_winning')
        self.assertFalse(result['guard_changed'])

    def test_remote_candidates_remain_legal_and_forbidden_cells_never_change(self):
        board = np.full((9, 10), 3, dtype=np.uint8)
        board[0, 0] = 1
        board[0, 1] = board[8, 9] = 0
        original = board.copy()
        inputs = []
        def inspect(child, actor, **options):
            inputs.append(child.copy())
            np.testing.assert_array_equal(child == 3, board == 3)
            self.assertEqual(np.count_nonzero(child == 2), 1)
            return reply(1 if child[0, 1] == 2 else None)
        with patch('search_guard.select_native_move', return_value=initial((0, 1))), \
             patch('search_guard.solve_forcing', side_effect=inspect):
            result = select_guarded_move(board, 2, time_limit=1)
        self.assertEqual(result['move'], (8, 9))
        self.assertEqual(len(inputs), 2)
        self.assertEqual(result['legal_count'], 2)
        np.testing.assert_array_equal(board, original)

    def test_shared_nodes_include_initial_vcf_without_double_counting(self):
        board = np.full((6, 7), 3, dtype=np.uint8)
        board[0, 0] = board[5, 6] = 0
        root_vcf = dict(proven_value=None, move=None, nodes=2)
        with patch('search_guard.select_native_move', return_value=initial((0, 0), nodes=6, forcing=root_vcf)), \
             patch('search_guard.solve_forcing', side_effect=[reply(1, 3), reply(None, 2)]) as solver:
            result = select_guarded_move(board, 1, max_nodes=20, native_fraction=.5, time_limit=1, probe_max_nodes=3)
        self.assertEqual(result['nodes'], 13)
        self.assertLessEqual(result['nodes'], 20)
        self.assertEqual([call.kwargs['max_nodes'] for call in solver.call_args_list], [3, 3])
        root_vcf = dict(proven_value=1, move=(0, 0), nodes=5, principal_variation=[])
        with patch('search_guard.select_native_move', return_value=initial((0, 0), value=1, nodes=5, forcing=root_vcf)):
            result = select_guarded_move(board, 1, max_nodes=20, native_fraction=.5)
        self.assertEqual(result['nodes'], 5)

    def test_each_probe_uses_only_remaining_wall_time(self):
        board = np.full((6, 7), 3, dtype=np.uint8)
        board[0, 0] = board[5, 6] = 0
        clock = [0.]
        allowances = []
        def native(*args, **kwargs):
            clock[0] = .12
            return initial((0, 0))
        def solver(*args, **kwargs):
            allowances.append(kwargs['time_limit'])
            clock[0] += kwargs['time_limit']
            return reply(1 if len(allowances) == 1 else None)
        with patch('search_guard.time.monotonic', side_effect=lambda: clock[0]), \
             patch('search_guard.select_native_move', side_effect=native), \
             patch('search_guard.solve_forcing', side_effect=solver):
            result = select_guarded_move(board, 1, time_limit=.2, probe_time_limit=.05)
        self.assertAlmostEqual(allowances[0], .05)
        self.assertAlmostEqual(allowances[1], .03)
        self.assertAlmostEqual(result['elapsed_seconds'], .2)
        self.assertTrue(result['budget_exhausted'])
        self.assertIsNone(result['proven_value'])

    def test_zero_budget_and_terminal_states_do_not_require_candidate_probes(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        with patch('search_guard.select_native_move', return_value=initial((3, 3))), \
             patch('search_guard.solve_forcing') as solver:
            result = select_guarded_move(board, 1, time_limit=0, max_nodes=0)
        solver.assert_not_called()
        self.assertEqual(result['status'], 'unexamined_budget_fallback')
        self.assertIsNone(result['proven_value'])
        board[0, :5] = 2
        with patch('search_guard.select_native_move') as native:
            result = select_guarded_move(board, 1)
        native.assert_not_called()
        self.assertIsNone(result['move'])
        self.assertEqual(result['proven_value'], -1)

    def test_probe_errors_propagate_without_mutating_input(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        original = board.copy()
        with patch('search_guard.select_native_move', return_value=initial((3, 3))), \
             patch('search_guard.solve_forcing', side_effect=RuntimeError('proof failure')):
            with self.assertRaisesRegex(RuntimeError, 'proof failure'):
                select_guarded_move(board, 1)
        np.testing.assert_array_equal(board, original)

    def test_quiet_search_is_disabled_by_default(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        with patch('search_guard.select_native_move', return_value=initial((3, 3))), \
             patch('search_guard.solve_forcing', return_value=reply(None)), \
             patch('search_guard.solve_threat') as quiet:
            result = select_guarded_move(board, 1)
        quiet.assert_not_called()
        self.assertIsNone(result['selected_reply']['opponent_threat_result'])
        self.assertEqual(result['threat_time_limit'], 0.)

    @threat_tests.simulated_leaf_schedule
    def test_actual_browser_white_30_is_rejected_by_quiet_certificate(self):
        board = threat_tests.browser_position()
        board[9, 5] = 0
        original = board.copy()
        losing_child = apply_board_move(board, (9, 5), 2)
        def actual_attack(position, actor, lines):
            # Fix only the known black attack after the actual white move.
            # The shared 20us/check clock stabilizes certificate discovery;
            # guard candidates, all defenders and every budget stay intact.
            if actor == 1 and np.array_equal(position, losing_child):
                return [(5, 8)]
            return _ordered_moves(position, actor, lines)
        with patch('search_guard.select_native_move', return_value=initial((9, 5))), \
             patch('board_threat_search._ordered_moves', new=actual_attack):
            result = select_guarded_move(board, 2, time_limit=.5, max_nodes=50000,
                                         threat_time_limit=.2, threat_max_nodes=20000, threat_width=16)
        self.assertIn((9, 5), result['rejected_moves'],
                      {key: result.get(key) for key in ('status', 'nodes', 'elapsed_seconds')})
        probe = next(p for p in result['guard_probes'] if p['move'] == (9, 5))
        self.assertIsNone(probe['opponent_result']['proven_value'])
        self.assertEqual(probe['opponent_threat_result']['proven_value'], 1)
        certificate = probe['opponent_threat_result']
        self.assertEqual(certificate['move'], (5, 8))
        self.assertEqual(certificate['winning_candidate']['certified_replies'], 225)
        self.assertLessEqual(certificate['nodes'], 20000)
        threat_tests.QuietThreatTests().verify_all_replies(losing_child, 1, certificate)
        self.assertEqual(probe['rejection_source'], 'quiet')
        position = apply_board_move(board, (9, 5), 2)
        for step in probe['opponent_threat_result']['principal_variation']:
            position = apply_board_move(position, step['move'], step['side'])
        self.assertEqual(board_winner(position), 1)
        self.assertIn(result['proven_value'], (None, -1))
        if result['proven_value'] == -1:
            self.assertEqual(len(result['rejected_moves']), result['legal_count'])
            self.assertTrue(all(p['rejection_source'] in ('vcf', 'quiet') for p in result['guard_probes']))
            position = board.copy()
            for step in result['principal_variation']:
                position = apply_board_move(position, step['move'], step['side'])
            self.assertEqual(board_winner(position), 1)
        self.assertLessEqual(result['nodes'], 50000)
        np.testing.assert_array_equal(board, original)

    def test_quiet_nodes_already_include_their_nested_vcf_work(self):
        board = np.array([[0, 2, 2, 2, 2, 0]], dtype=np.uint8)
        win = dict(proven_value=1, nodes=5, vcf_nodes=4,
                   principal_variation=[dict(side=2, move=(0, 5), kind='immediate_win')])
        unknown = dict(proven_value=None, nodes=2, vcf_nodes=1, principal_variation=[])
        with patch('search_guard.select_native_move', return_value=initial((0, 0))), \
             patch('search_guard.solve_forcing', return_value=reply(None)), \
             patch('search_guard.solve_threat', side_effect=[win, unknown]) as quiet:
            result = select_guarded_move(board, 1, time_limit=1, max_nodes=20,
                                         native_fraction=0, threat_time_limit=.2)
        self.assertEqual(result['nodes'], 9)
        self.assertEqual(result['move'], (0, 5))
        self.assertIsNone(result['proven_value'])
        self.assertEqual([c.kwargs['max_nodes'] for c in quiet.call_args_list], [19, 13])
        self.assertEqual(result['guard_probes'][0]['rejection_source'], 'quiet')

    def test_quiet_checks_share_remaining_wall_time(self):
        board = np.array([[0, 2, 2, 2, 2, 0]], dtype=np.uint8)
        clock = [0.]
        quiet_allowances = []
        def native(*args, **options):
            clock[0] = .1
            return initial((0, 0))
        def vcf(*args, **options):
            clock[0] += min(.02, options['time_limit'])
            return reply(None)
        def quiet(*args, **options):
            quiet_allowances.append(options['time_limit'])
            clock[0] += options['time_limit']
            return dict(nodes=2, proven_value=1, principal_variation=[])
        with patch('search_guard.time.monotonic', side_effect=lambda: clock[0]), \
             patch('search_guard.select_native_move', side_effect=native), \
             patch('search_guard.solve_forcing', side_effect=vcf), \
             patch('search_guard.solve_threat', side_effect=quiet):
            result = select_guarded_move(board, 1, time_limit=.2, threat_time_limit=.07)
        self.assertEqual(len(quiet_allowances), 1)
        self.assertAlmostEqual(quiet_allowances[0], .07)
        self.assertAlmostEqual(result['elapsed_seconds'], .2)
        self.assertIsNone(result['proven_value'])
        self.assertTrue(result['budget_exhausted'])

    def test_quiet_search_errors_and_over_budget_counts_are_not_hidden(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        original = board.copy()
        with patch('search_guard.select_native_move', return_value=initial((3, 3))), \
             patch('search_guard.solve_forcing', return_value=reply(None)), \
             patch('search_guard.solve_threat', side_effect=RuntimeError('quiet error')):
            with self.assertRaisesRegex(RuntimeError, 'quiet error'):
                select_guarded_move(board, 1, threat_time_limit=.2)
        with patch('search_guard.select_native_move', return_value=initial((3, 3))), \
             patch('search_guard.solve_forcing', return_value=reply(None)), \
             patch('search_guard.solve_threat', return_value=dict(nodes=50, proven_value=1)):
            with self.assertRaisesRegex(RuntimeError, 'node allocation'):
                select_guarded_move(board, 1, max_nodes=10, threat_time_limit=.2)
        np.testing.assert_array_equal(board, original)

    def test_bad_parameters_are_rejected(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        for options in ({'max_nodes': True}, {'time_limit': -1}, {'native_fraction': 1.1},
                        {'probe_max_nodes': 0}, {'probe_time_limit': 0}, {'forcing_depth': 1.5},
                        {'candidate_width': 65}, {'time_limit': float('nan')},
                        {'threat_time_limit': -1}, {'threat_time_limit': float('nan')},
                        {'threat_max_nodes': 0}, {'threat_width': True}):
            with self.assertRaises(ValueError):
                select_guarded_move(board, 1, **options)
        with self.assertRaises(ValueError):
            select_guarded_move(board, True)
        with self.assertRaises(ValueError):
            select_guarded_move(board, 1, np.ones((7, 6)))


class AttackProofTests(unittest.TestCase):
    def test_positive_attack_selects_certificate_move_and_retains_evidence(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        proof = dict(proven_value=1, move=(3, 4), nodes=7,
                     principal_variation=[dict(side=1, move=(3, 4), kind='quiet_move')])
        with patch('search_guard.select_native_move', return_value=initial((2, 2), nodes=3)), \
             patch('search_guard.solve_threat', return_value=proof) as attack, \
             patch('search_guard.solve_forcing') as defend:
            result = select_guarded_move(board, 1, max_nodes=100, time_limit=2,
                attack_time_limit=.2, attack_max_nodes=20,
                threat_quiet_plies=2, threat_total_plies=48)
        self.assertEqual(result['move'], (3, 4))
        self.assertEqual(result['proven_value'], 1)
        self.assertEqual(result['status'], 'attack_forced_win')
        self.assertEqual(result['attack_result'], proof)
        self.assertEqual(result['nodes'], 10)
        self.assertEqual(result['completed_depth'], 0)
        self.assertTrue(result['guard_changed'])
        self.assertEqual(attack.call_args.kwargs['max_quiet_plies'], 2)
        self.assertEqual(attack.call_args.kwargs['max_total_plies'], 48)
        defend.assert_not_called()

    def test_unknown_attack_cannot_displace_native_and_uses_shared_budget(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        proof = dict(proven_value=None, move=None, nodes=30, principal_variation=[])
        with patch('search_guard.select_native_move', return_value=initial((2, 2), nodes=2)), \
             patch('search_guard.solve_threat', return_value=proof) as attack, \
             patch('search_guard.solve_forcing', return_value=reply(None, nodes=1)) as defend:
            result = select_guarded_move(board, 1, max_nodes=35, time_limit=2,
                attack_time_limit=.2, attack_max_nodes=100)
        self.assertEqual(attack.call_args.kwargs['max_nodes'], 33)
        self.assertEqual(defend.call_args.kwargs['max_nodes'], 3)
        self.assertEqual(result['nodes'], 33)
        self.assertEqual(result['move'], (2, 2))
        self.assertIsNone(result['proven_value'])
        self.assertEqual(result['attack_result'], proof)
        self.assertEqual(result['completed_depth'], 7)

    def test_attack_budget_exhaustion_preserves_a_legal_unverified_move(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        proof = dict(proven_value=None, move=None, nodes=5, principal_variation=[])
        with patch('search_guard.select_native_move', return_value=initial((2, 2), nodes=0)), \
             patch('search_guard.solve_threat', return_value=proof), \
             patch('search_guard.solve_forcing') as defend:
            result = select_guarded_move(board, 1, max_nodes=5, time_limit=2,
                attack_time_limit=.2)
        self.assertEqual(result['move'], (2, 2))
        self.assertIsNone(result['proven_value'])
        self.assertEqual(result['status'], 'unexamined_budget_fallback')
        self.assertTrue(result['budget_exhausted'])
        defend.assert_not_called()

    def test_attack_does_not_allow_illegal_move_or_budget_overrun(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        for proof in (dict(proven_value=1, move=(8, 8), nodes=1, principal_variation=[]),
                      dict(proven_value=None, move=None, nodes=11, principal_variation=[])):
            with self.subTest(proof=proof), \
                 patch('search_guard.select_native_move', return_value=initial((2, 2), nodes=0)), \
                 patch('search_guard.solve_threat', return_value=proof):
                with self.assertRaises(RuntimeError):
                    select_guarded_move(board, 1, max_nodes=10, time_limit=2, attack_time_limit=.2)

    def test_zero_quiet_plies_runs_real_prover_with_supported_total_limit(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        with patch('search_guard.select_native_move', return_value=initial((2, 2), nodes=0)):
            result = select_guarded_move(board, 1, max_nodes=100, time_limit=1,
                attack_time_limit=.1, threat_quiet_plies=0, threat_total_plies=256)
        self.assertEqual(result['move'], (2, 2))
        self.assertIsNone(result['proven_value'])
        self.assertEqual(result['attack_result']['max_quiet_plies'], 0)
        self.assertLessEqual(result['nodes'], 100)

    def test_attack_off_by_default_and_invalid_limits_rejected(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        with patch('search_guard.select_native_move', return_value=initial((2, 2))), \
             patch('search_guard.solve_forcing', return_value=reply(None)), \
             patch('search_guard.solve_threat') as attack:
            result = select_guarded_move(board, 1)
        attack.assert_not_called()
        self.assertIsNone(result['attack_result'])
        for options in ({'attack_time_limit': -1}, {'attack_max_nodes': 0},
                        {'threat_quiet_plies': True}, {'threat_quiet_plies': -1},
                        {'threat_quiet_plies': 33}, {'threat_total_plies': 0},
                        {'threat_total_plies': 257}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                select_guarded_move(board, 1, **options)


if __name__ == '__main__':
    unittest.main()
