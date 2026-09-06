import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from board_rules import apply_board_move, board_winner, legal_cells, winning_cells
from global_policy_constraints import (FORMAT, align_constraint, load_policy_constraints,
                                       verify_action_certificate)


def sample_position(*, white_to_move=True, shape=(6, 7)):
    moves = [(2, 1), (0, 0), (2, 2), (0, 2), (2, 3), (5, 6)]
    if white_to_move:
        moves.append((4, 4))
    board = np.zeros(shape, dtype=np.uint8)
    history = []
    for i, point in enumerate(moves):
        side = 1 + i % 2
        board = apply_board_move(board, point, side)
        history.append(dict(row=point[0], col=point[1], side=side))
    return board, 1 + len(history) % 2, history


def open_four_certificate(board, side, move, actor_value=-1):
    after = apply_board_move(board, move, side)
    winner = side if actor_value == 1 else 3 - side
    attack = move if actor_value == 1 else (2, 4)
    attacked = after if actor_value == 1 else apply_board_move(after, attack, winner)
    width = board.shape[1]
    replies = []
    for response in legal_cells(attacked):
        child = apply_board_move(attacked, response, 3 - winner)
        wins = winning_cells(child, winner)
        if not wins:
            raise AssertionError('small fixture must win after every real defense')
        replies.append(dict(move=response[0] * width + response[1], value=1, auditId=None,
                            line=[dict(side=winner, move=wins[0][0] * width + wins[0][1])]))
    point = attack[0] * width + attack[1]
    node = dict(board=attacked.reshape(-1).tolist(), side=winner, move=point,
                replies=replies, certifiedReplies=len(replies), mandatory=False)
    first = replies[0]
    line = [dict(side=winner, move=point), dict(side=3 - winner, move=first['move'])] + first['line']
    return dict(kind='complete_all_replies_certificate', result=dict(value=1, move=point,
                line=line, auditId=0, certifiedReplies=len(replies)), certificates=[node])


def nest_first_leaf(certificate, shape):
    cert = copy.deepcopy(certificate)
    root = cert['certificates'][0]
    first = root['replies'][0]
    child = apply_board_move(np.array(root['board']).reshape(shape), divmod(first['move'], shape[1]), 3 - root['side'])
    finish = first['line'][0]['move']
    terminal = apply_board_move(child, divmod(finish, shape[1]), root['side'])
    nested = dict(board=terminal.reshape(-1).tolist(), side=root['side'], move=finish,
                  replies=[], certifiedReplies=0, mandatory=False)
    first['auditId'] = 0
    cert['certificates'] = [nested, root]
    cert['result']['auditId'] = 1
    return cert


class CertificateTests(unittest.TestCase):
    def test_negative_action_is_a_successor_proof_with_every_defender_reply(self):
        board, side, _ = sample_position()
        original = board.copy()
        proof = open_four_certificate(board, side, (1, 0))
        checked = verify_action_certificate(board, side, (1, 0), -1, proof)
        expected = len(legal_cells(apply_board_move(apply_board_move(board, (1, 0), side), (2, 4), 1)))
        self.assertEqual(checked['defender_branches'], expected)
        self.assertEqual(checked['attacker_nodes'], 1)
        self.assertEqual(checked['forced_lines'], expected)
        np.testing.assert_array_equal(board, original)

    def test_positive_action_uses_the_root_after_that_exact_move(self):
        board, side, _ = sample_position(white_to_move=False)
        proof = open_four_certificate(board, side, (2, 4), 1)
        self.assertGreater(verify_action_certificate(board, side, (2, 4), 1, proof)['defender_branches'], 0)
        with self.assertRaisesRegex(ValueError, 'exactly the labeled action'):
            verify_action_certificate(board, side, (1, 0), 1, proof)

    def test_missing_duplicate_unknown_or_wrong_actor_branch_is_rejected(self):
        board, side, _ = sample_position()
        original = open_four_certificate(board, side, (1, 0))
        for defect in ('missing', 'duplicate', 'unknown', 'actor', 'boolean_count', 'wrong_start'):
            proof = copy.deepcopy(original)
            node = proof['certificates'][0]
            if defect == 'missing':
                node['replies'].pop()
                node['certifiedReplies'] -= 1
                proof['result']['certifiedReplies'] -= 1
            elif defect == 'duplicate': node['replies'][-1] = node['replies'][0]
            elif defect == 'unknown': node['replies'][0]['value'] = None
            elif defect == 'actor': node['side'] = 2
            elif defect == 'boolean_count': node['certifiedReplies'] = True
            else: node['board'][1 * board.shape[1] + 1] = 2
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                verify_action_certificate(board, side, (1, 0), -1, proof)

    def test_nested_certificate_links_are_replayed_and_cycles_or_detached_nodes_fail(self):
        board, side, _ = sample_position()
        proof = nest_first_leaf(open_four_certificate(board, side, (1, 0)), board.shape)
        self.assertEqual(verify_action_certificate(board, side, (1, 0), -1, proof)['attacker_nodes'], 2)
        for defect in ('cycle', 'board_identity', 'unreachable'):
            bad = copy.deepcopy(proof)
            if defect == 'cycle': bad['certificates'][1]['replies'][0]['auditId'] = 1
            elif defect == 'board_identity': bad['certificates'][0]['board'][0] = 0
            else: bad['certificates'].insert(0, copy.deepcopy(bad['certificates'][0]))
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                verify_action_certificate(board, side, (1, 0), -1, bad)

    def test_a_legal_winning_pv_with_a_free_defender_choice_is_not_a_proof(self):
        board, side, _ = sample_position()
        after = apply_board_move(board, (1, 0), side)
        attack = (4, 5)
        attacked = apply_board_move(after, attack, 1)
        width = board.shape[1]
        # This chosen defender allows a later open four; it was free to act
        # elsewhere after the first quiet black move. All root replies listed
        # is insufficient when even one continuation is only a cooperative PV.
        line = [dict(side=actor, move=r * width + c) for actor, r, c in
                [(1, 0, 4), (2, 1, 4), (1, 2, 4), (2, 2, 0), (1, 2, 5)]]
        replies = [dict(move=r * width + c, value=1, auditId=None, line=line)
                   for r, c in legal_cells(attacked)]
        root = dict(board=attacked.reshape(-1).tolist(), side=1, move=4 * width + 5,
                    replies=replies, certifiedReplies=len(replies))
        proof = dict(kind='complete_all_replies_certificate', verified=True, result=dict(
            value=1, auditId=0, move=root['move'], certifiedReplies=len(replies),
            line=[dict(side=1, move=root['move']), dict(side=2, move=replies[0]['move'])] + line), certificates=[root])
        with self.assertRaisesRegex(ValueError, 'free defender choice'):
            verify_action_certificate(board, side, (1, 0), -1, proof)
        with self.assertRaisesRegex(ValueError, 'complete all-replies'):
            verify_action_certificate(board, side, (1, 0), -1, dict(verified=True, value=1, line=line))

    def test_terminal_after_action_uses_rules_and_cannot_label_a_draw_or_loss(self):
        board, side, _ = sample_position(white_to_move=False)
        board[2, 4] = 1
        after = apply_board_move(board, (2, 5), side)
        proof = dict(kind='terminal_after_action', board=after.tolist(), winner=side)
        self.assertEqual(verify_action_certificate(board, side, (2, 5), 1, proof)['kind'], proof['kind'])
        for move, value in [((1, 0), 1), ((2, 5), -1)]:
            with self.subTest(move=move, value=value), self.assertRaises(ValueError):
                verify_action_certificate(board, side, move, value, proof)

    def test_boolean_cells_moves_sides_and_values_and_illegal_actions_fail(self):
        board, side, _ = sample_position()
        proof = open_four_certificate(board, side, (1, 0))
        cases = [(board, True, (1, 0), -1), (board, side, (True, 0), -1),
                 (board, side, (1, 0), True), (board, side, (0, 0), -1)]
        mixed = board.tolist(); mixed[1][1] = False
        cases.append((mixed, side, (1, 0), -1))
        forbidden = board.copy(); forbidden[1, 0] = 3
        cases.append((forbidden, side, (1, 0), -1))
        for args in cases:
            with self.subTest(side=args[1], move=args[2], value=args[3]), self.assertRaises(ValueError):
                verify_action_certificate(*args, proof)


class AlignmentTests(unittest.TestCase):
    def test_rectangular_d4_and_joint_color_swap_preserve_every_action(self):
        board, side, _ = sample_position()
        bad, good = np.zeros(board.shape, bool), np.zeros(board.shape, bool)
        bad[1, 0] = True; good[3, 5] = True
        row = dict(board=board, side=side, losing_mask=bad, winning_mask=good)
        for turns in range(4):
            for flip in (False, True):
                for swap in (False, True):
                    def transform(a):
                        a = np.rot90(a, turns)
                        return np.fliplr(a) if flip else a
                    target = transform(board)
                    if swap: target = np.array([0, 2, 1, 3], dtype=np.uint8)[target]
                    checked = align_constraint(row, target, 3 - side if swap else side)
                    with self.subTest(turns=turns, flip=flip, swap=swap):
                        np.testing.assert_array_equal(checked['losing_mask'], transform(bad))
                        np.testing.assert_array_equal(checked['winning_mask'], transform(good))
                        self.assertTrue(checked['evidence_transforms'])
        changed = board.copy(); changed[1, 1] = 3
        self.assertIsNone(align_constraint(row, changed, side))
        self.assertIsNone(align_constraint(row, board, 3 - side))

    def test_board_symmetries_union_certified_actions_instead_of_discarding_them(self):
        board = np.zeros((5, 7), np.uint8)
        bad = np.zeros(board.shape, bool); bad[0, 0] = True
        row = dict(board=board, side=1, losing_mask=bad, winning_mask=np.zeros(board.shape, bool))
        aligned = align_constraint(row, board, 1)
        self.assertEqual(set(map(tuple, np.argwhere(aligned['losing_mask']))), {(0, 0), (0, 6), (4, 0), (4, 6)})
        self.assertEqual(len(aligned['evidence_transforms']), 4)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.board, self.side, self.history = sample_position()
        self.source = dict(board=self.board.tolist(), history=self.history, plies=len(self.history), winner=0)
        self.source_sha = self.write_json('source.json', self.source)

    def write_json(self, name, value):
        raw = json.dumps(value, sort_keys=True).encode()
        (self.root / name).write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def row(self, move=(1, 0)):
        proof = open_four_certificate(self.board, self.side, move)
        filename = f'proof-{move[0]}-{move[1]}.json'
        digest = self.write_json(filename, proof)
        return dict(format=FORMAT, board=self.board.tolist(), side=self.side, game_id='original-visible-id',
                    source_file='source.json', source_sha256=self.source_sha, prefix_plies=len(self.history),
                    action_evidence=[dict(move=list(move), actor_value=-1, proof_actor=1,
                                          proof_file=filename, proof_file_sha256=digest)])

    def write_rows(self, rows, name='constraints.jsonl'):
        raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
        if name.endswith('.gz'): raw = gzip.compress(raw, mtime=0)
        path = self.root / name; path.write_bytes(raw)
        return path

    def test_partial_labels_remain_masked_unknown_and_metadata_has_stable_source_group(self):
        path = self.write_rows([self.row()])
        rows, report = load_policy_constraints(path)
        row = rows[0]
        self.assertEqual(row['game_id'], 'source-game-' + self.source_sha)
        self.assertEqual(row['source_game_id'], 'original-visible-id')
        self.assertEqual(row['source'], 'verified_action_constraints')
        self.assertEqual(row['policy_source'], 'verified_action_sets')
        self.assertEqual(row['group_kind'], 'constructed_position')
        self.assertEqual(row['value_source'], 'unknown')
        self.assertEqual(row['search_requested_depth'], 0)
        self.assertEqual(row['search_completed_depth'], 0)
        self.assertFalse(row['policy_mask']); self.assertFalse(row['value_valid'])
        self.assertFalse(row['terminal_value_valid']); self.assertFalse(row['game_terminal'])
        self.assertIsNone(row['search_proven_value']); self.assertEqual(row['value'], 0)
        self.assertEqual(row['target_policy'].dtype, np.float32)
        self.assertEqual(row['target_policy'].shape, self.board.shape)
        self.assertEqual(float(row['target_policy'].sum()), 0)
        self.assertEqual(set(map(tuple, np.argwhere(row['losing_mask']))), {(1, 0)})
        self.assertFalse(row['winning_mask'].any())
        self.assertEqual(report['whole_position_values'], 0)

    def test_jsonl_gzip_duplicates_union_different_moves_without_new_records_or_games(self):
        first = self.write_rows([self.row()], 'a.jsonl')
        second = self.write_rows([self.row((1, 1))], 'b.jsonl.gz')
        rows, report = load_policy_constraints([first, second])
        self.assertEqual(len(rows), 1)
        self.assertEqual(report['groups'], 1)
        self.assertEqual(report['merged_rows'], 1)
        self.assertEqual(report['action_evidence'], 2)
        self.assertEqual(set(map(tuple, np.argwhere(rows[0]['losing_mask']))), {(1, 0), (1, 1)})
        self.assertEqual(len(rows[0]['action_evidence']), 2)
        self.assertEqual(len(rows[0]['constraint_imports']), 2)

    def test_missing_or_wrong_hashes_and_tampered_source_prefix_are_rejected(self):
        baseline = self.row()
        for defect in ('source_hash', 'proof_hash', 'missing_source_hash', 'missing_proof_hash', 'prefix', 'side', 'board', 'actor'):
            row = copy.deepcopy(baseline)
            if defect == 'source_hash': row['source_sha256'] = '0' * 64
            elif defect == 'proof_hash': row['action_evidence'][0]['proof_file_sha256'] = '0' * 64
            elif defect == 'missing_source_hash': del row['source_sha256']
            elif defect == 'missing_proof_hash': del row['action_evidence'][0]['proof_file_sha256']
            elif defect == 'prefix': row['prefix_plies'] -= 1
            elif defect == 'side': row['side'] = 1
            elif defect == 'board': row['board'][1][1] = 3
            else: row['action_evidence'][0]['proof_actor'] = 2
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                load_policy_constraints(self.write_rows([row]))

    def test_source_must_replay_legally_and_cannot_continue_after_terminal(self):
        row = self.row()
        for defect in ('repeat', 'side', 'board', 'plies', 'winner', 'boolean'):
            source = copy.deepcopy(self.source)
            if defect == 'repeat': source['history'][1] = dict(row=2, col=1, side=2)
            elif defect == 'side': source['history'][1]['side'] = 1
            elif defect == 'board': source['board'][1][0] = 1
            elif defect == 'plies': source['plies'] += 1
            elif defect == 'winner': source['winner'] = 1
            else: source['history'][0]['row'] = True
            row['source_sha256'] = self.write_json('source.json', source)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                load_policy_constraints(self.write_rows([row]))
        source = dict(board=np.zeros((6, 7), dtype=int).tolist(), history=[])
        for i, point in enumerate([(2, 0), (0, 0), (2, 1), (0, 2), (2, 2), (5, 6), (2, 3), (5, 4), (2, 4), (5, 2)]):
            actor = 1 + i % 2
            source['board'][point[0]][point[1]] = actor
            source['history'].append(dict(row=point[0], col=point[1], side=actor))
        row['source_sha256'] = self.write_json('source.json', source)
        with self.assertRaisesRegex(ValueError, '已结束'):
            load_policy_constraints(self.write_rows([row]))

    def legacy_source(self):
        root = copy.deepcopy(self.source)
        board = root.pop('board')
        root['ai_side'] = 2
        root['final_state'] = dict(board=board, history=copy.deepcopy(root['history']),
                                   winner=root['winner'], move_count=root['plies'],
                                   finished=False, turn=self.side, ai_side=2, human_side=1)
        return root

    def test_legacy_python_final_state_preserves_the_original_source_identity(self):
        row = self.row()
        source = self.legacy_source()
        row['source_sha256'] = self.write_json('source.json', source)
        records, _ = load_policy_constraints(self.write_rows([row]))
        self.assertEqual(records[0]['source_sha256'], row['source_sha256'])
        self.assertEqual(records[0]['source_file'], str((self.root / 'source.json').resolve()))
        self.assertEqual(records[0]['game_id'], 'source-game-' + row['source_sha256'])
        self.assertTrue(records[0]['losing_mask'][1, 0])
        # The existing top-level-board format remains independent of this
        # compatibility branch, and explicit invalid board never falls back.
        row['source_sha256'] = self.write_json('source.json', self.source)
        self.assertEqual(len(load_policy_constraints(self.write_rows([row]))[0]), 1)
        source['board'] = None
        row['source_sha256'] = self.write_json('source.json', source)
        with self.assertRaises(ValueError):
            load_policy_constraints(self.write_rows([row]))

    def test_legacy_duplicate_identity_fields_must_agree_and_match_real_rules(self):
        row = self.row()
        for defect in ('history', 'count', 'winner', 'ai', 'human', 'finished', 'turn',
                       'bool_count', 'bool_winner', 'bool_ai', 'bool_turn', 'missing_finished'):
            source = self.legacy_source()
            state = source['final_state']
            if defect == 'history': state['history'][0]['col'] = 3
            elif defect == 'count': state['move_count'] += 1
            elif defect == 'winner': state['winner'] = 1
            elif defect == 'ai': state['ai_side'] = 1
            elif defect == 'human': state['human_side'] = 2
            elif defect == 'finished': state['finished'] = True
            elif defect == 'turn': state['turn'] = 0
            elif defect == 'bool_count': state['move_count'] = True
            elif defect == 'bool_winner': state['winner'] = False
            elif defect == 'bool_ai': state['ai_side'] = True
            elif defect == 'bool_turn': state['turn'] = False
            else: del state['finished']
            row['source_sha256'] = self.write_json('source.json', source)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                load_policy_constraints(self.write_rows([row]))

    def test_legacy_actual_terminal_requires_finished_true_and_turn_zero(self):
        row = self.row()
        source = self.legacy_source()
        final = np.array(source['final_state']['board'], dtype=np.uint8)
        for point in ((1, 2), (2, 4), (1, 4), (2, 5)):
            actor = 1 + len(source['history']) % 2
            final = apply_board_move(final, point, actor)
            source['history'].append(dict(row=point[0], col=point[1], side=actor))
        self.assertEqual(board_winner(final), 1)
        source['winner'] = 1
        source['plies'] = len(source['history'])
        state = source['final_state']
        state.update(board=final.tolist(), history=copy.deepcopy(source['history']),
                     winner=1, move_count=source['plies'], finished=True, turn=0)
        row['source_sha256'] = self.write_json('source.json', source)
        self.assertEqual(len(load_policy_constraints(self.write_rows([row]))[0]), 1)
        for defect in ('finished', 'turn'):
            bad = copy.deepcopy(source)
            bad['final_state'][defect] = False if defect == 'finished' else 2
            row['source_sha256'] = self.write_json('source.json', bad)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                load_policy_constraints(self.write_rows([row]))

    def test_rehashed_pv_or_missing_defense_is_not_rescued_by_verified_flags(self):
        row = self.row()
        proof = open_four_certificate(self.board, self.side, (1, 0))
        proof['certificates'][0]['replies'].pop()
        proof['certificates'][0]['certifiedReplies'] -= 1
        proof['result']['certifiedReplies'] -= 1
        proof['verified'] = True
        row['action_evidence'][0]['verified'] = True
        row['action_evidence'][0]['proof_file_sha256'] = self.write_json('proof-1-0.json', proof)
        with self.assertRaisesRegex(ValueError, 'missing, duplicate or illegal'):
            load_policy_constraints(self.write_rows([row]))

    def test_identical_physical_board_from_two_sources_requires_owner_reconciliation(self):
        first = self.row()
        second = copy.deepcopy(first)
        source = dict(self.source, run_id='a different source identity')
        second['source_file'] = 'other.json'
        second['source_sha256'] = self.write_json('other.json', source)
        with self.assertRaisesRegex(ValueError, 'crosses source groups'):
            load_policy_constraints(self.write_rows([first, second]))

    def test_loader_always_verifies_each_action_before_merging(self):
        rows = [self.row(), self.row((1, 1))]
        import global_policy_constraints as module
        with patch.object(module, 'verify_action_certificate', wraps=module.verify_action_certificate) as verify:
            loaded, _ = load_policy_constraints(self.write_rows(rows))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(verify.call_count, 2)


if __name__ == '__main__':
    unittest.main()
