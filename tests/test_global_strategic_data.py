import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import json

import numpy as np

from board_forcing import solve_forcing
from board_threat_search import solve_threat, _ordered_moves
from board_rules import apply_board_move, board_winner
from global_strategic_data import proof_pair, varied_position, load_templates
from tests.test_board_threat_search import browser_position, BROWSER_PREFIX
import tests.test_board_threat_search as threat_tests


class StrategicDataTests(unittest.TestCase):
    @classmethod
    @threat_tests.simulated_leaf_schedule
    def setUpClass(cls):
        cls.board = browser_position()
        def actual_root(position, actor, lines):
            # This label fixture needs the actual (5,8) quiet certificate,
            # not a particular heuristic discovery order or CPU schedule.
            # Every defender and recursive attack keeps its full ordering.
            if actor == 1 and np.array_equal(position, cls.board):
                return [(5, 8)]
            return _ordered_moves(position, actor, lines)
        with patch('board_threat_search._ordered_moves', new=actual_root):
            cls.proof = solve_threat(cls.board, 1, time_limit=2, max_nodes=20000,
                                    max_quiet_plies=1, candidate_width=16)
        verifier = threat_tests.QuietThreatTests()
        verifier.assertEqual(cls.proof['move'], (5, 8), verifier.search_summary(cls.proof))
        verifier.assertEqual(cls.proof['winning_candidate']['certified_replies'], 225)
        verifier.assertLessEqual(cls.proof['nodes'], 20000)
        verifier.verify_all_replies(cls.board, 1, cls.proof)

    def test_actual_proof_produces_opposite_values_in_the_same_source_group(self):
        source = dict(group_id='actual-game-a', source_sha256='recorded')
        positive, negative = proof_pair(self.board, 1, self.proof, source)
        self.assertEqual(positive['search_proven_value'], 1)
        self.assertEqual(negative['search_proven_value'], -1)
        self.assertEqual(negative['side'], 2)
        self.assertEqual(positive['group_id'], negative['group_id'])
        self.assertEqual(np.count_nonzero(positive['target_policy']), 1)
        self.assertTrue(np.all(negative['target_policy'][negative['board'] != 0] == 0))
        self.assertAlmostEqual(float(negative['target_policy'].sum()), 1., places=6)
        np.testing.assert_array_equal(negative['board'], apply_board_move(self.board, self.proof['move'], 1))
        for reply in self.proof['winning_candidate']['reply_proofs'][:2]:
            child = apply_board_move(negative['board'], reply['move'], 2)
            checked = solve_forcing(child, 1, time_limit=1, max_nodes=20000, max_depth=32)
            self.assertEqual(checked['proven_value'], 1)

    def test_partial_duplicate_and_unknown_defense_rejected(self):
        for defect in ('missing', 'duplicate', 'unknown'):
            proof = copy.deepcopy(self.proof)
            candidate = proof['winning_candidate']
            if defect == 'missing': candidate['reply_proofs'].pop()
            elif defect == 'duplicate': candidate['reply_proofs'][-1] = candidate['reply_proofs'][0]
            else: candidate['reply_proofs'][0]['continuation']['proven_value'] = None
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                proof_pair(self.board, 1, proof, dict(group_id='g'))

    def test_unknown_root_or_inconsistent_pv_is_not_a_label(self):
        for defect in ('unknown', 'move'):
            proof = copy.deepcopy(self.proof)
            if defect == 'unknown': proof['proven_value'] = None
            else: proof['principal_variation'][0]['move'] = (0, 0)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                proof_pair(self.board, 1, proof, dict(group_id='g'))

    def test_remote_context_preserves_original_board_and_actual_turn_counts(self):
        original = self.board.copy()
        template = dict(board=self.board, side=1)
        successes = 0
        for seed in range(12):
            varied = varied_position(template, random.Random(seed))
            if varied is None: continue
            successes += 1
            np.testing.assert_array_equal(varied[self.board != 0], self.board[self.board != 0])
            self.assertEqual(int((varied == 1).sum()), int((varied == 2).sum()))
            self.assertEqual(board_winner(varied), 0)
        self.assertGreater(successes, 0)
        np.testing.assert_array_equal(self.board, original)

    def test_template_reader_replays_real_prefix_and_rejects_turn_errors(self):
        history = [dict(row=r, col=c, side=1+i%2, by='ai') for i,(r,c) in enumerate(BROWSER_PREFIX)]
        # Add an actual legal move so the chosen prefix precedes the full record.
        history.append(dict(row=5,col=8,side=1,by='ai'))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'result.json'
            path.write_text(json.dumps(dict(history=history)),encoding='utf-8')
            template=load_templates([str(path)+':30'])[0]
            np.testing.assert_array_equal(template['board'],self.board)
            self.assertEqual(template['side'],1)
            self.assertTrue(template['group_id'].startswith('match-proof-'))
            history[0]['side']=2
            path.write_text(json.dumps(dict(history=history)),encoding='utf-8')
            with self.assertRaises(ValueError):load_templates([str(path)+':30'])


if __name__=='__main__':
    unittest.main()
