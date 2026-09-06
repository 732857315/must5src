"""Constructed-position reachability, exact tactical labels, and deduplication."""

import unittest
from unittest.mock import patch

import numpy as np

from board_rules import apply_board_move, board_winner, legal_cells, winning_cells
from global_data import canonical_board_key, physical_board_key
from global_tactics_data import build_tactical_positions


class GlobalTacticalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.report = build_tactical_positions(12, seed=62519, sizes=((6, 7), 10, 16))

    def test_records_are_constructed_cases_and_never_claim_completed_games(self):
        self.assertEqual(self.report['constructed_cases'], 12)
        self.assertEqual(self.report['requested_count'], 12)
        self.assertLessEqual(self.report['attempts'], self.report['max_attempts'])
        self.assertEqual(self.report['values'], {'1': 6, '-1': 6})
        self.assertEqual({record['side'] for record in self.records}, {1, 2})
        for record in self.records:
            self.assertEqual(record['group_kind'], 'constructed_position')
            self.assertFalse(record['game_terminal'])
            self.assertIsNone(record['game_winner'])
            self.assertIsNone(record['game_length'])
            self.assertIsNone(record['terminal_value'])
            self.assertEqual(record['game_termination'], 'constructed_not_played')
            self.assertEqual(record['value_source'], 'search_proof')
            self.assertTrue(record['value_valid'])

    def test_every_board_has_a_legal_monotone_history_from_black_center(self):
        for record in self.records:
            board = record['board']
            center = (board.shape[0] // 2, board.shape[1] // 2)
            self.assertEqual(board[center], 1)
            self.assertEqual(board_winner(board), 0)
            black, white = int(np.sum(board == 1)), int(np.sum(board == 2))
            self.assertEqual(black - white, int(record['side'] == 2))
            position = np.where(board == 3, 3, 0).astype(np.uint8)
            position = apply_board_move(position, center, 1)
            remaining = {side: [tuple(map(int, point)) for point in np.argwhere(board == side)
                                if tuple(point) != center] for side in (1, 2)}
            actor = 2
            for _ in range(black + white - 1):
                self.assertTrue(remaining[actor])
                position = apply_board_move(position, remaining[actor].pop(), actor)
                self.assertEqual(board_winner(position), 0)
                actor = 3 - actor
            np.testing.assert_array_equal(position, board)
            self.assertEqual(actor, record['side'])

    def test_principal_variations_replay_by_independent_rules_until_actual_five(self):
        for record in self.records:
            position = record['board'].copy()
            actor = record['side']
            line = record['search_principal_variation']
            self.assertTrue(line)
            self.assertEqual(tuple(line[0]['move']), tuple(record['action']))
            for step in line:
                self.assertEqual(step['side'], actor)
                position = apply_board_move(position, step['move'], actor)
                actor = 3 - actor
            expected_winner = record['side'] if record['value'] == 1 else 3 - record['side']
            self.assertEqual(board_winner(position), expected_winner)
            self.assertEqual(record['search_proven_value'], int(record['value']))

    def test_proved_losses_label_all_legal_actions_uniformly(self):
        for record in self.records:
            board, target, side = record['board'], record['target_policy'], record['side']
            self.assertTrue(np.isfinite(target).all())
            self.assertTrue(np.all(target[board != 0] == 0))
            self.assertAlmostEqual(float(target.sum()), 1, places=5)
            if record['value'] != -1:
                continue
            self.assertEqual(record['policy_source'], 'all_legal_proved_loss')
            legal = legal_cells(board)
            np.testing.assert_allclose(target[board == 0], 1 / len(legal), rtol=1e-6)
            enemy_wins = winning_cells(board, 3 - side)
            self.assertGreaterEqual(len(enemy_wins), 2)
            self.assertEqual(winning_cells(board, side), [])
            # Verify every labelled action loses to at least one real reply.
            for move in legal:
                child = apply_board_move(board, move, side)
                self.assertTrue(any(board_winner(apply_board_move(child, reply, 3 - side)) == 3 - side
                                    for reply in enemy_wins if child[reply] == 0))

    def test_keys_are_d4_and_simultaneous_color_side_invariant(self):
        keys = {record['physical_key'] for record in self.records}
        self.assertEqual(len(keys), len(self.records))
        for record in self.records[:4]:
            board, side = record['board'], record['side']
            for turns in range(4):
                for transformed in (np.rot90(board, turns), np.fliplr(np.rot90(board, turns))):
                    swapped = np.array([0, 2, 1, 3], dtype=np.uint8)[transformed]
                    self.assertEqual(physical_board_key(transformed), record['physical_key'])
                    self.assertEqual(physical_board_key(swapped), record['physical_key'])
                    self.assertEqual(canonical_board_key(swapped, 3 - side), record['board_key'])

    def test_existing_physical_keys_are_excluded_without_mutating_input(self):
        seen = {record['physical_key'] for record in self.records}
        original = seen.copy()
        new, report = build_tactical_positions(4, seed=62519, sizes=((6, 7), 10, 16), seen_physical=seen)
        self.assertTrue(seen.isdisjoint(record['physical_key'] for record in new))
        self.assertEqual(seen, original)
        self.assertEqual(report['excluded_physical_keys'], len(seen))

    def test_zero_count_and_attempt_budget_are_explicit(self):
        records, report = build_tactical_positions(np.int64(0), seed=np.int64(8), sizes=6, max_attempts=0)
        self.assertEqual(records, [])
        self.assertEqual(report['attempts'], 0)
        with self.assertRaisesRegex(RuntimeError, '0 of 1'):
            build_tactical_positions(1, max_attempts=0)
        with patch('global_tactics_data._candidate', return_value=None) as candidate:
            with self.assertRaisesRegex(RuntimeError, 'after 3 attempts'):
                build_tactical_positions(1, max_attempts=3)
            self.assertEqual(candidate.call_count, 3)

    def test_progress_and_mask_categories_describe_actual_output(self):
        events = []
        records, report = build_tactical_positions(2, sizes=(6,), seed=12, progress_callback=events.append)
        self.assertEqual(events[-1]['records'], 2)
        self.assertEqual(events[-1]['attempts'], report['attempts'])
        for record in self.records + records:
            self.assertEqual(record['mask_kind'] == 'none', not np.any(record['board'] == 3))

    def test_bad_counts_budgets_shapes_keys_and_callbacks_fail_early(self):
        for options in ({'count': -1}, {'count': True}, {'count': 1.2}, {'max_attempts': -1},
                        {'max_attempts': True}, {'max_attempts': .5}, {'seed': True}, {'seed': 1.2},
                        {'sizes': ()}, {'sizes': (5,)}, {'sizes': (6.2,)}, {'sizes': ((6, 7.2),)},
                        {'sizes': ((6, True),)}, {'sizes': None}, {'sizes': '16'},
                        {'seen_physical': None}, {'seen_physical': 'single-key'}, {'seen_physical': [3]},
                        {'progress_callback': 3}):
            kwargs = dict(count=0)
            kwargs.update(options)
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    build_tactical_positions(**kwargs)


if __name__ == '__main__':
    unittest.main()
