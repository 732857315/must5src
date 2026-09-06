"""Independent rule replay and completeness checks for quiet threat proofs."""
import unittest
import json
from itertools import count
from functools import wraps
from pathlib import Path
from unittest.mock import patch
import numpy as np

from board_rules import apply_board_move, board_winner, legal_cells, winning_cells
from board_forcing import solve_forcing
from board_threat_search import solve_threat, _ThreatSearch, _BudgetExceeded

# Actual first 30 plies of native_v1_probe/game_000002; no runtime report needed.
BROWSER_PREFIX = [(8,8),(9,9),(7,10),(9,8),(9,7),(7,9),(8,7),(8,10),
                  (7,7),(10,7),(10,8),(8,9),(6,9),(11,9),(10,9),(11,6),
                  (12,5),(11,8),(11,5),(9,6),(10,6),(12,4),(8,5),(11,10),
                  (11,7),(11,12),(11,11),(13,10),(12,9),(9,5)]


# Actual first 23 plies of guarded_global_v2_formal/game_000005.
FORMAL_PREFIX = [(8,8),(7,7),(8,6),(8,9),(9,7),(7,9),(10,8),(7,5),
                 (7,6),(9,6),(9,9),(7,8),(11,7),(8,10),(11,9),(12,10),
                 (10,7),(7,10),(7,11),(8,7),(13,7),(12,7),(12,6)]


def formal_position():
    board = np.zeros((16, 16), dtype=np.uint8)
    for ply, move in enumerate(FORMAL_PREFIX):
        board = apply_board_move(board, move, 1 + ply % 2)
    return board


def browser_position():
    board = np.zeros((16, 16), dtype=np.uint8)
    for ply, move in enumerate(BROWSER_PREFIX):
        board = apply_board_move(board, move, 1 + ply % 2)
    return board


def cross_position():
    board = np.full((7, 7), 3, dtype=np.uint8)
    board[3, :] = 0
    board[:, 3] = 0
    board[3, 1:3] = 1
    board[1:3, 3] = 1
    board[6, 6] = 0  # A distant legal defense must not disappear from coverage.
    return board


def simulated_leaf_schedule(test):
    """Exercise certificate semantics under a declared deterministic schedule.

    A 20us clock check models the diagnosed ~80 VCF nodes per 5ms scan.
    Node, leaf-time, total-time and ply caps are unchanged. This tests real
    branch proofs, not guaranteed discovery on every machine/schedule.
    Dedicated deadline/allocation/node-cap tests use their original clocks.
    """
    @wraps(test)
    def run(*args, **kwargs):
        ticks = count()
        with patch('board_threat_search.time.monotonic',
                   new=lambda: next(ticks) * .000020):
            return test(*args, **kwargs)
    return run


class QuietThreatTests(unittest.TestCase):
    @staticmethod
    def search_summary(result):
        return {key: result.get(key) for key in
                ('move', 'reason', 'nodes', 'elapsed_seconds', 'recursive_calls',
                 'quiet_iterations', 'candidates_checked')}

    def verify_line(self, board, actor, line, expected_winner):
        position = board.copy()
        for step in line:
            self.assertFalse(board_winner(position))
            self.assertEqual(step['side'], actor)
            position = apply_board_move(position, step['move'], actor)
            actor = 3 - actor
        self.assertEqual(board_winner(position), expected_winner)

    def verify_all_replies(self, board, side, result):
        self.assertEqual(result['proven_value'], 1)
        evidence = result['winning_candidate']
        if evidence is None:
            self.verify_line(board, side, result['principal_variation'], side)
            return
        child = apply_board_move(board, result['move'], side)
        if evidence['quiet']:
            self.assertEqual(winning_cells(child, side), [])
        legal = set(legal_cells(child))
        checked = {tuple(item['move']) for item in evidence['reply_proofs']}
        self.assertEqual(legal, checked)
        self.assertEqual(evidence['legal_reply_count'], len(legal))
        self.assertEqual(evidence['certified_replies'], len(legal))
        self.assertEqual(len(evidence['reply_proofs']), len(legal))
        for item in evidence['reply_proofs']:
            after = apply_board_move(child, item['move'], 3 - side)
            self.assertEqual(item['continuation']['proven_value'], 1)
            self.verify_line(after, side, item['continuation']['principal_variation'], side)
            if item.get('proof_source') == 'recursive':
                self.verify_all_replies(after, side, item['continuation'])
        self.verify_line(board, side, result['principal_variation'], side)

    @simulated_leaf_schedule
    def test_real_browser_quiet_5_8_wins_against_all_225_responses(self):
        board = browser_position()
        original = board.copy()
        direct = solve_forcing(board, 1, time_limit=.2, max_nodes=10000, max_depth=32)
        self.assertIsNone(direct['proven_value'])
        from board_threat_search import _ordered_moves
        def root_only(position, actor, lines):
            if actor == 1 and np.array_equal(position, board):
                return [(5, 8)]
            return _ordered_moves(position, actor, lines)
        # This named regression verifies the actual (5,8) certificate, not
        # whether heuristic ordering picks it before other winning roots.
        with patch('board_threat_search._ordered_moves', new=root_only):
            result = solve_threat(board, 1, time_limit=2, max_nodes=20000)
        self.assertEqual(result['move'], (5, 8), self.search_summary(result))
        self.assertEqual(result['winning_candidate']['legal_reply_count'], 225)
        self.verify_all_replies(board, 1, result)
        np.testing.assert_array_equal(board, original)
        self.assertLessEqual(result['nodes'], 20000)
        self.assertGreater(result['nodes'], result['vcf_nodes'])

    def test_root_trimming_cannot_claim_loss_or_safety(self):
        result = solve_threat(browser_position(), 1, time_limit=1, candidate_width=1)
        self.assertIsNone(result['proven_value'])
        self.assertEqual(result['status'], 'unknown')
        self.assertTrue(result['candidate_limit_reached'])
        self.assertIsNone(result['move'])

    @simulated_leaf_schedule
    def test_distant_reply_and_gray_mask_are_preserved_for_both_colors(self):
        for side in (1, 2):
            board = cross_position()
            board[board == 1] = side
            original = board.copy()
            result = solve_threat(board, side, time_limit=1, candidate_width=1)
            self.assertEqual(result['move'], (3, 3))
            self.verify_all_replies(board, side, result)
            self.assertIn((6, 6), {tuple(p['move']) for p in result['winning_candidate']['reply_proofs']})
            np.testing.assert_array_equal(board, original)

    def test_one_unknown_defense_prevents_certificate(self):
        board = cross_position()
        actual_solver = solve_forcing
        def continuation(after, side, **options):
            if after[6, 6] == 2:
                return dict(proven_value=None, nodes=1, principal_variation=[])
            return actual_solver(after, side, **options)
        with patch('board_threat_search.solve_forcing', side_effect=continuation):
            result = solve_threat(board, 1, time_limit=1, candidate_width=1)
        self.assertIsNone(result['proven_value'])
        self.assertEqual(result['candidate_attempts'][0]['status'], 'continuation_unknown')
        self.assertLess(result['candidate_attempts'][0]['certified_replies'], 9)

    def test_opponent_immediate_five_preempts_continuation_oracle(self):
        board = np.zeros((7, 7), dtype=np.uint8)
        board[0, 1:5] = 2
        with patch('board_threat_search._ordered_moves', return_value=[(3, 3)]), \
             patch('board_threat_search.solve_forcing') as solver:
            result = solve_threat(board, 1, time_limit=1, candidate_width=1)
        solver.assert_not_called()
        self.assertIsNone(result['proven_value'])
        self.assertEqual(result['reason'], 'opponent_immediate_wins')
        for move in result['opponent_winning_cells']:
            self.assertEqual(board_winner(apply_board_move(board, move, 2)), 2)

    def test_current_own_immediate_win_has_priority_over_enemy_five_threat(self):
        board = np.zeros((7, 7), dtype=np.uint8)
        board[0, 1:5] = 2
        board[6, 1:5] = 1
        result = solve_threat(board, 1)
        self.assertEqual(result['reason'], 'immediate_win')
        self.assertEqual(result['proven_value'], 1)
        self.verify_line(board, 1, result['principal_variation'], 1)

    def test_node_caps_cover_candidate_and_response_traversal(self):
        for cap in (0, 1, 3, 10):
            result = solve_threat(cross_position(), 1, max_nodes=cap, time_limit=1)
            self.assertLessEqual(result['nodes'], cap)
            self.assertIsNone(result['proven_value'])
            self.assertTrue(result['budget_exhausted'])

    def test_nested_vcf_receives_remaining_total_time_and_nodes(self):
        clock = [0.]
        allocations = []
        def continuation(*args, **options):
            allocations.append(options)
            clock[0] += options['time_limit']
            return dict(proven_value=1, nodes=1, principal_variation=[])
        with patch('board_threat_search.time.monotonic', side_effect=lambda: clock[0]), \
             patch('board_threat_search.solve_forcing', side_effect=continuation):
            result = solve_threat(cross_position(), 1, max_nodes=20, time_limit=.07,
                                  vcf_max_nodes=100, vcf_time_limit=.05, candidate_width=1, staged_vcf=False)
        self.assertEqual(len(allocations), 2)
        self.assertAlmostEqual(allocations[0]['time_limit'], .05)
        self.assertAlmostEqual(allocations[1]['time_limit'], .02)
        self.assertEqual([a['max_nodes'] for a in allocations], [17, 15])
        self.assertIsNone(result['proven_value'])
        self.assertTrue(result['budget_exhausted'])
        self.assertEqual(result['nodes'], 6)

    def test_terminal_draw_and_enemy_win_never_become_positive_certificate(self):
        for board, side, expected in (([[1,2,1,2,1]], 2, None), ([[2,2,2,2,2]], 1, None),
                                      ([[1,1,1,1,1]], 1, 1), ([[3,3,3,3,3]], 1, None)):
            result = solve_threat(board, side)
            self.assertEqual(result['proven_value'], expected)
            self.assertIsNone(result['move'])

    def test_failure_does_not_mutate_caller_board_or_swallow_solver_errors(self):
        board = cross_position()
        original = board.copy()
        with patch('board_threat_search.solve_forcing', side_effect=RuntimeError('solver failed')):
            with self.assertRaisesRegex(RuntimeError, 'solver failed'):
                solve_threat(board, 1)
        np.testing.assert_array_equal(board, original)
        with patch('board_threat_search.solve_forcing', return_value=dict(nodes=1000, proven_value=1)):
            with self.assertRaisesRegex(RuntimeError, 'node cap'):
                solve_threat(board, 1, max_nodes=20)

    @simulated_leaf_schedule
    def test_formal_forced_defense_preserves_one_quiet_layer_and_all_replies(self):
        board = formal_position()
        original = board.copy()
        result = solve_threat(board, 2, time_limit=3, max_nodes=20000,
                              max_quiet_plies=1, max_total_plies=7)
        self.assertEqual(result['move'], (13, 5))
        self.assertEqual(result['proven_value'], 1)
        self.assertEqual(result['quiet_plies_used'], 1)
        self.assertLessEqual(result['max_ply'], 7)
        root = result['winning_candidate']
        self.assertTrue(root['mandatory_defense'])
        self.assertEqual(root['quiet_cost'], 0)
        self.assertEqual(root['certified_replies'], 232)
        branch = next(p for p in root['reply_proofs'] if p['move'] == (6, 9))
        self.assertIsNone(branch['vcf_result']['proven_value'])
        self.assertEqual(branch['proof_source'], 'recursive')
        self.assertEqual(branch['continuation']['move'], (9, 10))
        self.assertEqual(branch['continuation']['winning_candidate']['certified_replies'], 230)
        self.verify_all_replies(board, 2, result)
        np.testing.assert_array_equal(board, original)

    @simulated_leaf_schedule
    def test_quiet_two_retains_fast_one_quiet_certificate_by_iterative_deepening(self):
        result = solve_threat(browser_position(), 1, time_limit=3, max_nodes=20000,
                              max_quiet_plies=2)
        self.assertEqual(result['proven_value'], 1, self.search_summary(result))
        self.assertEqual(result['quiet_plies_used'], 1)
        self.assertEqual(len(result['quiet_iterations']), 1)
        self.assertEqual(result['quiet_iterations'][0]['quiet_limit'], 1)
        self.assertEqual(result['winning_candidate']['certified_replies'], 225)
        self.verify_all_replies(browser_position(), 1, result)

    @simulated_leaf_schedule
    def test_quiet_and_total_ply_limits_prevent_deeper_certificate(self):
        board = formal_position()
        for quiet, total in ((0, 64), (1, 6), (2, 0)):
            result = solve_threat(board, 2, time_limit=2, max_nodes=20000,
                                  max_quiet_plies=quiet, max_total_plies=total)
            self.assertIsNone(result['proven_value'])
            self.assertLessEqual(result['max_ply'], total)
            self.assertTrue(result['total_ply_limited'] or result['quiet_limited'] or quiet == 0)
        board = cross_position()
        for total in (2, 3, 4):
            result = solve_threat(board, 1, time_limit=1, max_total_plies=total)
            self.assertIsNone(result['proven_value'])
            self.assertLessEqual(result['max_ply'], total)

    def test_missing_real_defense_in_ordering_cannot_produce_a_certificate(self):
        from board_threat_search import _ordered_moves
        def omit_remote(board, side, lines):
            order = _ordered_moves(board, side, lines)
            return [p for p in order if p != (6, 6)] if side == 2 else order
        with patch('board_threat_search._ordered_moves', side_effect=omit_remote):
            result = solve_threat(cross_position(), 1, time_limit=1, candidate_width=1)
        self.assertIsNone(result['proven_value'])
        self.assertEqual(result['candidate_attempts'][0]['status'], 'incomplete_reply_coverage')

    @simulated_leaf_schedule
    def test_recursive_unknown_at_one_true_defense_prevents_outer_proof(self):
        board = formal_position()
        actual_solver = solve_forcing
        def stop_one(after, side, **options):
            # Black's remote defense to the nested white(9,10) setup.
            if after[9, 10] == 2 and after[15, 15] == 1 and after[6, 9] == 1:
                return dict(proven_value=None, nodes=1, principal_variation=[])
            return actual_solver(after, side, **options)
        with patch('board_threat_search.solve_forcing', side_effect=stop_one):
            result = solve_threat(board, 2, time_limit=2, max_nodes=20000,
                                  max_quiet_plies=1, candidate_width=2)
        self.assertIsNone(result['proven_value'])
        self.assertGreater(result['recursive_calls'], 0)
        self.assertEqual(result['candidate_attempts'][0]['status'], 'continuation_unknown')

    def test_all_recursive_nodes_and_iterations_share_one_cap(self):
        board = formal_position()
        result = solve_threat(board, 2, max_nodes=100, time_limit=3, max_quiet_plies=2)
        self.assertIsNone(result['proven_value'])
        self.assertEqual(result['nodes'], 100)
        self.assertTrue(result['budget_exhausted'])
        self.assertGreater(result['recursive_calls'], 0)
        self.assertEqual(sum(x['nodes'] for x in result['quiet_iterations']), result['nodes'])

    def test_recursive_continuations_share_the_original_deadline(self):
        clock = [0.]
        allocations = []
        actual_solver = solve_forcing
        def timed_vcf(*args, **options):
            allocations.append((clock[0], options['time_limit']))
            self.assertLessEqual(options['time_limit'], .03-clock[0]+1e-12)
            result = actual_solver(*args, **options)
            clock[0] += min(.002, options['time_limit'])
            return result
        with patch('board_threat_search.time.monotonic', side_effect=lambda: clock[0]), \
             patch('board_threat_search.solve_forcing', side_effect=timed_vcf):
            result = solve_threat(formal_position(), 2, time_limit=.03,
                                  max_nodes=20000, max_quiet_plies=2)
        self.assertGreater(result['recursive_calls'], 0)
        self.assertGreater(len(allocations), 2)
        self.assertAlmostEqual(result['elapsed_seconds'], .03)
        self.assertIsNone(result['proven_value'])
        self.assertTrue(result['budget_exhausted'])

    def test_completed_proof_cache_does_not_cache_unknown_or_skip_budget(self):
        board = cross_position()
        engine = _ThreatSearch(board, 1, max_nodes=20000, time_limit=3,
                                candidate_width=1, forcing_depth=32,
                                vcf_max_nodes=5000, vcf_time_limit=.05, max_total_plies=64)
        with patch('board_threat_search.solve_forcing', return_value=dict(proven_value=None, nodes=1, principal_variation=[])):
            first = engine.search(board, 1)
        self.assertIsNone(first['proven_value'])
        self.assertEqual(engine.cache, {})
        second = engine.search(board, 1)
        self.assertEqual(second['proven_value'], 1)
        previous = engine.nodes
        cached = engine.search(board, 1)
        self.assertEqual(cached['proven_value'], 1)
        self.assertTrue(cached['cache_hit'])
        self.assertEqual(engine.nodes, previous+1)
        engine.max_nodes = engine.nodes
        with self.assertRaises(_BudgetExceeded):
            engine.search(board, 1)

    def test_short_leaf_scan_reaches_later_candidate_before_deadline(self):
        # A bad first candidate uses its entire leaf allowance. A later real
        # cross setup is cheap to prove, but the old 50 ms leaf hides it.
        from board_threat_search import _ordered_moves
        board = cross_position()
        def order(position, side, lines):
            if np.array_equal(position, board):
                return [(6, 6), (3, 3)]
            return _ordered_moves(position, side, lines)
        for staged in (False, True):
            clock = [0.]
            def leaf(position, side, **options):
                if position[6, 6] == 1:
                    clock[0] += options['time_limit']
                    return dict(proven_value=None, nodes=1, principal_variation=[])
                return solve_forcing(position, side, **options)
            with patch('board_threat_search._ordered_moves', side_effect=order), \
                 patch('board_threat_search.time.monotonic', side_effect=lambda: clock[0]), \
                 patch('board_threat_search.solve_forcing', side_effect=leaf):
                result = solve_threat(board, 1, time_limit=.04, candidate_width=2,
                                      staged_vcf=staged)
            if staged:
                self.assertEqual(result['move'], (3, 3))
                self.verify_all_replies(board, 1, result)
                self.assertEqual(result['quiet_iterations'][0]['vcf_stage'], 'fast')
            else:
                self.assertIsNone(result['proven_value'])
                self.assertTrue(result['budget_exhausted'])

    def test_larger_leaf_retry_can_prove_unknown_without_resetting_budget(self):
        clock = [0.]
        allocations = []
        def leaf(position, side, **options):
            allocations.append(options.copy())
            clock[0] += .001
            if options['max_nodes'] <= 500:
                return dict(proven_value=None, nodes=options['max_nodes'], principal_variation=[])
            return solve_forcing(position, side, **options)
        with patch('board_threat_search.time.monotonic', side_effect=lambda: clock[0]), \
             patch('board_threat_search.solve_forcing', side_effect=leaf):
            result = solve_threat(cross_position(), 1, max_nodes=1200, time_limit=1,
                                  candidate_width=1)
        self.verify_all_replies(cross_position(), 1, result)
        self.assertEqual([x['vcf_stage'] for x in result['quiet_iterations']], ['fast', 'configured'])
        self.assertIsNone(result['quiet_iterations'][0]['proven_value'])
        self.assertEqual(allocations[0]['max_nodes'], 500)
        self.assertLess(allocations[1]['max_nodes'], 700)
        self.assertGreaterEqual(result['nodes'], 500)
        self.assertLessEqual(result['nodes'], 1200)
        self.assertEqual(sum(x['nodes'] for x in result['quiet_iterations']), result['nodes'])
        self.assertAlmostEqual(result['elapsed_seconds'], len(allocations)*.001)

    def test_fast_stage_never_increases_user_leaf_caps_or_duplicates_same_caps(self):
        allocations = []
        def unknown(*args, **options):
            allocations.append(options)
            return dict(proven_value=None, nodes=1, principal_variation=[])
        with patch('board_threat_search.solve_forcing', side_effect=unknown):
            result = solve_threat(cross_position(), 1, candidate_width=1,
                                  vcf_max_nodes=100, vcf_time_limit=.002)
        self.assertEqual(len(result['quiet_iterations']), 1)
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0]['max_nodes'], 100)
        self.assertEqual(allocations[0]['time_limit'], .002)

    @simulated_leaf_schedule
    def test_real_v3_failure_simulated_leaf_schedule_keeps_full_reply_coverage(self):
        # First 32 actual plies of recursive_global_v3_formal/game_000002.
        moves = [(8,8),(9,9),(7,10),(9,8),(7,9),(7,11),(8,10),(9,11),
                 (9,10),(6,10),(8,9),(8,11),(10,11),(7,8),(11,10),(10,10),
                 (11,9),(11,11),(12,9),(6,11),(5,11),(9,12),(13,8),(14,7),
                 (13,9),(14,9),(13,10),(13,11),(11,8),(12,12),(13,13),(10,7)]
        board = np.zeros((16,16), dtype=np.uint8)
        for i, move in enumerate(moves):
            board = apply_board_move(board, move, 1+i%2)
            if i+1 not in (30, 32):
                continue
            # A separate 1us/check diagnostic exhausts this same 5000-node
            # budget. The decorator explicitly fixes discovery scheduling;
            # every accepted continuation is still independently replayed.
            result = solve_threat(board, 1, max_nodes=5000, time_limit=5,
                                  vcf_max_nodes=500, vcf_time_limit=.005, max_quiet_plies=2)
            self.assertEqual(result['move'], (13,13) if i+1 == 30 else (9,7))
            self.assertEqual(result['winning_candidate']['certified_replies'], 255-(i+1))
            self.verify_all_replies(board, 1, result)
            self.assertLessEqual(result['nodes'], 5000)

    @simulated_leaf_schedule
    def test_browser_last_quiet_setup_retains_all_counter_four_defenses(self):
        fixture = json.loads((Path(__file__).parent / 'browser/fixtures/layout_sequence.json').read_text())
        board = np.zeros((16, 16), dtype=np.uint8)
        for ply, move in enumerate(fixture['history'][:22]):
            board = apply_board_move(board, move, 1 + ply % 2)
        from board_threat_search import _ordered_moves
        original_order = _ordered_moves
        def root_only(position, side, lines):
            if side == 1 and np.array_equal(position, board):
                return [(11, 8)]
            return original_order(position, side, lines)
        with patch('board_threat_search._ordered_moves', side_effect=root_only):
            result = solve_threat(board, 1, time_limit=3, max_nodes=30000,
                                  max_quiet_plies=1, max_total_plies=19)
        self.assertEqual(result['proven_value'], 1)
        self.assertEqual(result['quiet_plies_used'], 1)
        self.assertEqual(result['winning_candidate']['certified_replies'], 233)
        self.verify_all_replies(board, 1, result)
        mandatory_zero = []
        def walk(proof):
            attempt = proof.get('winning_candidate')
            if not attempt:
                return
            if attempt['mandatory_defense'] and proof['quiet_remaining'] == 0:
                mandatory_zero.append(proof)
            for reply in attempt['reply_proofs']:
                if reply.get('proof_source') == 'recursive':
                    walk(reply['continuation'])
        walk(result)
        self.assertTrue(mandatory_zero)
        self.assertTrue(all(p['winning_candidate']['quiet_cost'] == 0 for p in mandatory_zero))

    @simulated_leaf_schedule
    def test_browser_forcing_attack_can_precede_quiet_setup_with_complete_replies(self):
        fixture = json.loads((Path(__file__).parent / 'browser/fixtures/layout_sequence.json').read_text())
        from board_threat_search import _ordered_moves
        original_order = _ordered_moves
        for swap, rotate in ((False, 0), (True, 1)):
            with self.subTest(swap=swap, rotate=rotate):
                board = np.zeros((16, 16), dtype=np.uint8)
                for ply, move in enumerate(fixture['history'][:20]):
                    board = apply_board_move(board, move, 1 + ply % 2)
                quiet_board = apply_board_move(board, (8, 5), 1)
                quiet_board = apply_board_move(quiet_board, (8, 4), 2)
                if swap:
                    colors = np.array([0, 2, 1, 3], dtype=np.uint8)
                    board, quiet_board = colors[board], colors[quiet_board]
                board = np.rot90(board, rotate).copy()
                quiet_board = np.rot90(quiet_board, rotate).copy()
                side = 2 if swap else 1
                original = board.copy()
                root_move, quiet_move = (8, 5), (11, 8)
                for _ in range(rotate):
                    root_move = (15-root_move[1], root_move[0])
                    quiet_move = (15-quiet_move[1], quiet_move[0])
                def actual_attacks(position, actor, lines):
                    # Isolate forcing -> quiet along the actual attacking
                    # route, including its rotated/color-swapped equivalent.
                    # Every defender and other recursive attack is untouched.
                    if actor == side:
                        if np.array_equal(position, board):
                            return [root_move]
                        if np.array_equal(position, quiet_board):
                            return [quiet_move]
                    return original_order(position, actor, lines)
                with patch('board_threat_search._ordered_moves', new=actual_attacks):
                    result = solve_threat(board, side, time_limit=4, max_nodes=60000,
                                          max_quiet_plies=1, max_total_plies=21)
                self.assertEqual(result['proven_value'], 1, self.search_summary(result))
                self.assertEqual(result['move'], root_move)
                self.assertEqual(result['quiet_plies_used'], 1)
                self.assertTrue(result['winning_candidate']['forcing_attack'])
                self.assertEqual(result['winning_candidate']['quiet_cost'], 0)
                self.assertEqual(result['winning_candidate']['certified_replies'], 235)
                self.verify_all_replies(board, side, result)
                np.testing.assert_array_equal(board, original)

    def test_bad_parameters_fail_before_search(self):
        board = cross_position()
        for options in ({'max_nodes':True}, {'max_nodes':-1}, {'candidate_width':0},
                        {'candidate_width':1.5}, {'forcing_depth':-1}, {'time_limit':float('inf')},
                        {'time_limit':float('nan')}, {'vcf_time_limit':0}, {'vcf_max_nodes':0},
                        {'max_quiet_plies':True}, {'max_quiet_plies':-1},
                        {'max_total_plies':-1}, {'max_total_plies':257}, {'staged_vcf':1}, {'staged_vcf':'yes'}):
            with self.assertRaises(ValueError):
                solve_threat(board, 1, **options)
        with self.assertRaises(ValueError):
            solve_threat(board, True)
        with self.assertRaises(ValueError):
            solve_threat([[4]], 1)


if __name__ == '__main__':
    unittest.main()
