"""Native positive proofs, independently replayed against all legal replies.

No model is loaded. Real trajectory fixtures are solved and verified once each;
the recorded unknown trajectory is deliberately not a positive assertion.
Ample wall time and fixed node caps test certificate semantics, not a promise
that the same proof is discovered under every browser/device time schedule.
"""
from copy import deepcopy
import ctypes
from functools import lru_cache
from itertools import count
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import native_threat as native
from board_rules import apply_board_move, board_winner, legal_cells, winning_cells
from global_policy_constraints import verify_action_certificate


FIXTURES = Path(__file__).parent / 'browser' / 'fixtures' / 'native_threat_regressions.json'


def cross_board():
    """Two open threes after the center move, plus a remote legal defense."""
    board = np.full((7, 9), 3, dtype=np.uint8)
    board[3, :7] = 0
    board[:, 3] = 0
    board[3, 1:3] = 1
    board[1:3, 3] = 1
    board[6, 8] = 0
    return board


def open_three_board():
    board = np.full((7, 9), 3, dtype=np.uint8)
    board[3, :7] = 0
    board[3, 2:5] = 1
    board[6, 8] = 0
    return board


def immediate_board():
    board = np.full((6, 9), 3, dtype=np.uint8)
    board[2, 1:5] = 1
    board[2, 5] = 0
    return board


def transform(board, turns=0, reflected=False, swap=False):
    result = np.rot90(board, turns)
    if reflected:
        result = np.fliplr(result)
    if swap:
        result = np.array([0, 2, 1, 3], dtype=np.uint8)[result]
    return result.copy()


def exact_small(board, side):
    """Unpruned rule minimax for the two-empty false-fork fixture only."""
    winner = board_winner(board)
    if winner:
        return 1 if winner == side else -1
    return max((-exact_small(apply_board_move(board, move, side), 3 - side)
                for move in legal_cells(board)), default=0)


@lru_cache(maxsize=1)
def small_cross_result():
    # Consumers must deepcopy before tampering; only this tiny fixture is reused.
    return native.solve_native_threat(cross_board(), 1, milliseconds=5000,
        max_nodes=20000, quiet=1, total=11, width=16)


class NativeThreatInputTests(unittest.TestCase):
    """These checks do not load a DLL or invoke a search."""

    def test_invalid_boards_are_rejected_before_native_load(self):
        mixed_bool = np.zeros((5, 5), dtype=object)
        mixed_bool[1, 1] = True
        invalid = [None, [], np.zeros(25, dtype=int), [[0] * 5, [0] * 6],
                   np.zeros((4, 5), dtype=int), np.zeros((5, 33), dtype=int),
                   np.zeros((5, 5), dtype=float), np.zeros((5, 5), dtype=bool),
                   mixed_bool, np.full((5, 5), 4), np.full((5, 5), -1)]
        with patch.object(native, 'native_threat_library') as library:
            for board in invalid:
                with self.subTest(board_type=type(board).__name__, shape=np.shape(board) if isinstance(board, np.ndarray) else None):
                    with self.assertRaises(ValueError):
                        native.solve_native_threat(board, 1)
            library.assert_not_called()

    def test_integer_ranges_bool_and_nonfinite_budget_are_strict(self):
        invalid = {
            'side': [0, 3, True, 1.0],
            'milliseconds': [-1, True, float('nan'), float('inf'), 10**400, '10'],
            'max_nodes': [-1, 2**31, True, 1.0], 'quiet': [-1, 5, True, 1.0],
            'total': [0, 65, True, 1.0], 'width': [0, 65, True, 1.0],
            'node_capacity': [-1, 2049, True, 1.0],
            'edge_capacity': [-1, 65537, True, 1.0],
            'line_capacity': [-1, 262145, True, 1.0],
            'certificate': [0, 1, None, np.bool_(True)],
            'minimum_quiet': [-1, 0, 3, True, 1.0],
        }
        with patch.object(native, 'native_threat_library') as library:
            for name, values in invalid.items():
                for value in values:
                    with self.subTest(name=name, value=value):
                        options = dict(side=1)
                        options[name] = value
                        with self.assertRaises(ValueError):
                            native.solve_native_threat(immediate_board(), **options)
            library.assert_not_called()

    def test_context_pointers_cannot_escape_or_be_misaligned(self):
        context = ctypes.create_string_buffer(64)
        pointer = ctypes.cast(ctypes.addressof(context), ctypes.POINTER(ctypes.c_int))
        self.assertIs(native._checked_pointer(pointer, context, ctypes.c_int, 16), pointer)
        bad = [ctypes.POINTER(ctypes.c_int)(),
               ctypes.cast(ctypes.addressof(context) - 4, ctypes.POINTER(ctypes.c_int)),
               ctypes.cast(ctypes.addressof(context) + 1, ctypes.POINTER(ctypes.c_int))]
        for value in bad:
            with self.assertRaises(RuntimeError):
                native._checked_pointer(value, context, ctypes.c_int)
        with self.assertRaises(RuntimeError):
            native._checked_pointer(pointer, context, ctypes.c_int, 17)

    def test_merely_legal_free_reply_pv_is_not_a_forcing_certificate(self):
        board = np.full((7, 9), 3, dtype=np.uint8)
        board[3, :7] = 0
        board[3, 1:3] = 1
        board[0, 0] = 0
        line = [dict(side=1 + i % 2, move=p) for i, p in
                enumerate([30, 0, 31, 27, 32])]
        position = board.copy()
        for step in line:
            position = apply_board_move(position, divmod(step['move'], 9), step['side'])
        self.assertEqual(board_winner(position), 1)  # A legal winning PV alone is insufficient.
        proof = dict(kind='forcing_line_certificate', board=board.tolist(), side=1,
                     result=dict(value=1, move=30, line=line, certifiedReplies=0))
        with self.assertRaisesRegex(ValueError, 'free defender choice'):
            native.verify_native_threat_result(board, 1,
                dict(value=1, move=30, line=line, proof=proof))


    def test_line_decode_uses_cell_count_not_board_height_as_its_bound(self):
        board = np.zeros((5, 9), dtype=np.uint8)
        board[2, 1] = 1
        points = [20, 36, 21, 38, 22, 40, 23]
        before = board.copy()
        line = native._decode_line(points, board, 1, 7)
        self.assertEqual([item['move'] for item in line], points)
        self.assertGreater(len(line), board.shape[0])
        self.assertEqual([item['side'] for item in line], [1, 2, 1, 2, 1, 2, 1])
        np.testing.assert_array_equal(board, before)
        with self.assertRaises(RuntimeError):
            native._decode_line(points, board, 1, 6)


class _StorageFixture(ctypes.Structure):
    _fields_ = [('nodes', native.NativeThreatNode * 2), ('scratch', native.NativeThreatEdge),
                ('lines', ctypes.c_int * 2), ('pv', ctypes.c_int * 3),
                ('board', ctypes.c_ubyte * 1024), ('iteration', native.NativeThreatIteration)]


class _StorageLibrary:
    """A complete four-reply proof with two physical leaf groups; no DLL."""
    context_size = ctypes.sizeof(_StorageFixture)
    source_sha256 = {'mock': 'not_a_compiled_or_searched_result'}
    binary_sha256 = 'mock'

    def __init__(self):
        self.combined_sha256 = 'storage-mock-' + str(id(self))
        self.calls = []
        self.edges = [(27, -1, 0, 1), (32, -1, 1, 1),
                      (33, -1, 1, 1), (62, -1, 1, 1)]
        self.groups = 2
        self.version = 2
        self.corrupt = lambda output, context: None

    def native_threat_storage_version(self):
        return self.version

    def _pointer(self, context, field, ctype):
        return ctypes.cast(ctypes.addressof(context) + getattr(_StorageFixture, field).offset,
                           ctypes.POINTER(ctype))

    def _solve(self, arguments, minimum):
        context, board, rows, cols, side, ms, cap, quiet, total, width, nc, ec, lc, now, out = arguments
        self.calls.append((quiet, minimum))
        self.capacity = ec
        storage = _StorageFixture.from_buffer(context)
        storage.board[:rows*cols] = board[:rows*cols]
        storage.nodes[0].move, storage.nodes[0].side = 28, side
        storage.nodes[0].first_edge, storage.nodes[0].edge_count = 0, len(self.edges)
        storage.nodes[0].board[:rows*cols] = board[:rows*cols]
        storage.nodes[0].board[28] = side
        storage.lines[:] = [32, 27]
        storage.pv[:] = [28, 27, 32]
        storage.iteration = native.NativeThreatIteration(0, minimum, 5, 5, 1)
        output = out._obj
        for name, value in zip((field[0] for field in native.NativeThreatOutput._fields_),
                              [1, 28, 5, 0, 0, 1, len(self.edges), 3, len(self.edges), minimum, 1, 1]):
            setattr(output, name, value)
        self.corrupt(output, storage)
        return 0

    def native_threat_solve(self, *arguments):
        return self._solve(arguments, 0 if arguments[7] == 0 else 1)

    def native_threat_solve_range(self, *arguments):
        return self._solve(arguments[:-1], arguments[-1])

    def native_threat_board(self, context):
        return self._pointer(context, 'board', ctypes.c_ubyte)

    def native_threat_pv(self, context):
        return self._pointer(context, 'pv', ctypes.c_int)

    def native_threat_lines(self, context):
        return self._pointer(context, 'lines', ctypes.c_int)

    def native_threat_line_count(self, context):
        return 2

    def native_threat_group_count(self, context):
        return self.groups

    def native_threat_group_capacity(self, context):
        return self.capacity

    def native_threat_iteration(self, context, index):
        return self._pointer(context, 'iteration', native.NativeThreatIteration)

    def native_threat_node(self, context, index):
        return ctypes.cast(ctypes.addressof(context) + _StorageFixture.nodes.offset +
            index * ctypes.sizeof(native.NativeThreatNode), ctypes.POINTER(native.NativeThreatNode))

    def native_threat_edge(self, context, index):
        # All logical indices deliberately alias the exact same scratch struct.
        storage = _StorageFixture.from_buffer(context)
        storage.scratch = native.NativeThreatEdge(*self.edges[index])
        return self._pointer(context, 'scratch', native.NativeThreatEdge)


class NativeThreatStorageMockTests(unittest.TestCase):
    """ABI/decode checks are synthetic, not C compression or search evidence."""

    def run_fixture(self, library, **options):
        with patch.object(native, 'native_threat_library', return_value=library):
            return native.solve_native_threat(open_three_board(), 1, edge_capacity=2, **options)

    def test_shared_scratch_and_leaf_offset_keep_every_reply_independently_validated(self):
        library = _StorageLibrary()
        with patch.object(native, '_decode_line', wraps=native._decode_line) as decode:
            result = self.run_fixture(library)
        self.assertEqual(decode.call_count, 5)  # Root PV plus all four real replies.
        self.assertEqual(result['certificate_edges'], 4)
        self.assertEqual(result['certificate_groups'], 2)
        self.assertGreater(result['certificate_edges'], result['group_capacity'])
        root = result['proof']['certificates'][0]
        self.assertEqual([edge['move'] for edge in root['replies']], [27, 32, 33, 62])
        self.assertEqual([edge['line'][0]['move'] for edge in root['replies']], [32, 27, 27, 27])
        counts = native.verify_native_threat_result(open_three_board(), 1, result)
        self.assertEqual(counts['defender_branches'], 4)
        # A later getter changes scratch memory, never the decoded certificate.
        before = deepcopy(result)
        library.native_threat_edge(native._LOCAL.context, 0)
        self.assertEqual(result, before)

    def test_same_offset_is_not_a_license_to_skip_a_different_defenders_board(self):
        library = _StorageLibrary()
        library.edges[0] = (27, -1, 1, 1)  # Shared PV now tries the occupied reply 27.
        with self.assertRaisesRegex(RuntimeError, 'illegal continuation'):
            self.run_fixture(library)

    def test_missing_duplicate_or_detached_logical_records_are_rejected(self):
        for kind in ('missing', 'duplicate', 'detached_node', 'detached_edge'):
            library = _StorageLibrary()
            if kind == 'missing':
                library.edges.pop()
            elif kind == 'duplicate':
                library.edges[3] = library.edges[2]
            elif kind == 'detached_node':
                library.corrupt = lambda output, context: setattr(output, 'certificate_nodes', 2)
            else:
                library.corrupt = lambda output, context: setattr(output, 'certificate_edges', 5)
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                self.run_fixture(library)

    def test_physical_group_counts_and_logical_bound_are_separate(self):
        for kind in ('group_capacity', 'group_count', 'missing_group', 'logical_count', 'storage_version'):
            library = _StorageLibrary()
            if kind == 'group_capacity':
                library.native_threat_group_capacity = lambda context: 3
            elif kind == 'group_count':
                library.groups = 3
            elif kind == 'missing_group':
                library.groups = 0
            elif kind == 'logical_count':
                library.corrupt = lambda output, context: setattr(output, 'certificate_edges', 64)
            else:
                library.version = 1
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                self.run_fixture(library)

    def test_abi_logical_count_above_old_65536_cap_is_not_mistaken_for_group_capacity(self):
        # Count-bound test only: this synthetic large arena is deliberately NOT
        # decoded, independently verified, or claimed to be a real C proof.
        library = _StorageLibrary()
        def counts(output, context):
            output.certificate_nodes = 1200
            output.certificate_edges = 70000
        library.corrupt = counts
        result = self.run_fixture(library, certificate=False)
        self.assertEqual(result['certificate_edges'], 70000)
        self.assertEqual(result['certificate_groups'], 2)
        self.assertIsNone(result['proof'])
        self.assertFalse(result['certificateDecoded'])
        with self.assertRaises(ValueError):
            native.verify_native_threat_result(open_three_board(), 1, result)
        library.corrupt = lambda output, context: (setattr(output, 'certificate_nodes', 1200),
            setattr(output, 'certificate_edges', 1200 * open_three_board().size + 1))
        with self.assertRaisesRegex(RuntimeError, 'out of bounds'):
            self.run_fixture(library, certificate=False)

    def test_range_dispatch_records_only_the_requested_stage(self):
        for quiet, minimum in ((0, 0), (3, 3), (4, 4), (4, 2)):
            library = _StorageLibrary()
            result = self.run_fixture(library, quiet=quiet, minimum_quiet=minimum)
            self.assertEqual(library.calls, [(quiet, minimum)])
            self.assertEqual(result['minimumQuiet'], minimum)
            self.assertEqual([it['quiet'] for it in result['iterations']], [minimum])
        library = _StorageLibrary()
        result = self.run_fixture(library)
        self.assertEqual(library.calls, [(2, 1)])
        self.assertEqual(result['minimumQuiet'], 1)

    def test_range_rejects_out_of_range_request_or_returned_iteration(self):
        with patch.object(native, 'native_threat_library') as load:
            for quiet, minimum in ((0, 1), (1, 0), (3, 4), (4, True), (4, 2.0)):
                with self.subTest(quiet=quiet, minimum=minimum), self.assertRaises(ValueError):
                    native.solve_native_threat(open_three_board(), 1, quiet=quiet, minimum_quiet=minimum)
            load.assert_not_called()
        library = _StorageLibrary()
        library.corrupt = lambda output, context: setattr(context.iteration, 'quiet', 2)
        with self.assertRaisesRegex(RuntimeError, 'iteration fields'):
            self.run_fixture(library, quiet=4, minimum_quiet=3)
        library.corrupt = lambda output, context: setattr(output, 'completed_quiet', 1)
        with self.assertRaisesRegex(RuntimeError, 'below the requested minimum'):
            self.run_fixture(library, quiet=4, minimum_quiet=3)

    def test_unknown_cannot_expose_leftover_complete_or_partial_storage(self):
        library = _StorageLibrary()
        def incomplete(output, context):
            output.value, output.move, output.root_id = 2, -1, -1
            output.pv_length, output.certified_replies = 0, 0
            output.status, output.exhausted = 3, 1
        library.corrupt = incomplete
        with self.assertRaisesRegex(RuntimeError, 'uncompleted root proof'):
            self.run_fixture(library)


class NativeThreatProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            native.native_threat_library()
        except RuntimeError as exc:
            if 'requires clang and lld-link' in str(exc):
                raise unittest.SkipTest(str(exc))
            raise  # A failed build or malformed ABI is a regression, not a skip.

    def assert_unknown(self, result):
        self.assertIsNone(result['value'], result)
        self.assertIsNone(result['move'])
        self.assertIsNone(result['coordinate'])
        self.assertEqual(result['line'], [])
        self.assertIsNone(result['proof'])
        self.assertEqual(result['root_id'], -1)
        self.assertEqual(result['certifiedReplies'], 0)
        for key in ('certificate_nodes', 'certificate_edges', 'certificate_groups', 'certificate_lines'):
            self.assertEqual(result[key], 0, key)

    def check_positive(self, board, side, result, *, max_nodes, full_graph=False):
        self.assertEqual(result['value'], 1, {k: result.get(k) for k in
            ('value', 'move', 'nodes', 'status', 'exhausted', 'iterations')})
        self.assertEqual(tuple(result['coordinate']), divmod(result['move'], board.shape[1]))
        self.assertEqual(board.flat[result['move']], 0)
        self.assertEqual(result['storage_version'], 2)
        self.assertLessEqual(result['certificate_groups'], result['group_capacity'])
        self.assertLessEqual(result['certificate_edges'], result['certificate_nodes'] * board.size)
        self.assertGreater(result['nodes'], 0)
        self.assertLessEqual(result['nodes'], max_nodes)
        for key in ('elapsedMs', 'decodeMs', 'setupMs'):
            self.assertTrue(math.isfinite(result[key]) and result[key] >= 0, key)
        if result['iterations']:
            self.assertEqual(sum(item['nodes'] for item in result['iterations']), result['nodes'])
        proof = result['proof']
        if full_graph:
            self.assertEqual(proof['kind'], 'complete_all_replies_certificate')
        if proof['kind'] == 'complete_all_replies_certificate':
            # The existing verifier is independent of native code and checks
            # every reachable defender edge, board identity, and forced leaf.
            counts = verify_action_certificate(board, side, result['coordinate'], 1, proof)
            root = proof['certificates'][proof['result']['auditId']]
            after = apply_board_move(board, result['coordinate'], side)
            expected = {r * board.shape[1] + c for r, c in legal_cells(after)}
            self.assertEqual({item['move'] for item in root['replies']}, expected)
            self.assertEqual(len(root['replies']), len(expected))
            self.assertEqual(counts['attacker_nodes'], len(proof['certificates']))
            self.assertEqual(counts['defender_branches'], sum(len(node['replies']) for node in proof['certificates']))
        else:
            self.assertEqual(proof['kind'], 'forcing_line_certificate')
            self.assertEqual(result['root_id'], -1)
            counts = native.verify_native_threat_result(board, side, result)
        self.assertGreater(counts['moves_replayed'], 0)
        return counts

    def test_gray_rectangular_cross_covers_remote_defense(self):
        board = cross_board()
        before = board.copy()
        result = small_cross_result()
        self.check_positive(board, 1, result, max_nodes=20000, full_graph=True)
        root = result['proof']['certificates'][result['proof']['result']['auditId']]
        self.assertIn(6 * 9 + 8, {edge['move'] for edge in root['replies']})
        np.testing.assert_array_equal(board, before)

    def test_physical_group_capacity_preserves_more_logical_edges_and_fails_closed_one_short(self):
        # Freeze the host clock so this tests arena capacity, not machine speed.
        board = cross_board()
        options = dict(milliseconds=5000, max_nodes=20000, quiet=1, total=11)
        with patch.object(native, '_clock_ms', return_value=100.0):
            baseline = native.solve_native_threat(board, 1, **options)
            groups = baseline['certificate_groups']
            self.assertGreater(groups, 0)
            self.assertGreater(baseline['certificate_edges'], groups)
            exact = native.solve_native_threat(board, 1, edge_capacity=groups, **options)
            self.check_positive(board, 1, exact, max_nodes=20000, full_graph=True)
            self.assertEqual(exact['certificate_groups'], groups)
            self.assertEqual(exact['proof'], baseline['proof'])
            failed = native.solve_native_threat(board, 1, edge_capacity=groups-1, **options)
            self.assert_unknown(failed)
            self.assertEqual(failed['status'], 3)
            self.assertTrue(failed['exhausted'])
            # The failed arena cannot contaminate the next complete proof.
            recovered = native.solve_native_threat(board, 1, edge_capacity=groups, **options)
            self.assertEqual(recovered['proof'], exact['proof'])

    def test_later_quiet_range_does_not_rerun_completed_earlier_stages(self):
        # Two isolated empties have no forcing line and make an inexpensive unknown.
        board = np.full((5, 7), 3, dtype=np.uint8)
        board[0, 0] = board[4, 6] = 0
        with patch.object(native, '_clock_ms', return_value=100.0):
            for quiet, minimum in ((3, 3), (4, 4), (4, 3)):
                result = native.solve_native_threat(board, 1, quiet=quiet, minimum_quiet=minimum,
                    total=9, max_nodes=1000, milliseconds=1000)
                self.assert_unknown(result)
                self.assertTrue(result['iterations'])
                self.assertTrue(all(minimum <= row['quiet'] <= quiet for row in result['iterations']))
                self.assertEqual(sum(row['nodes'] for row in result['iterations']), result['nodes'])
                self.assertEqual(result['minimumQuiet'], minimum)

    def test_cross_d4_rectangle_and_synchronous_color_swap(self):
        # Two representative transforms, not a costly 16-way real-game sweep.
        remote = np.zeros((7, 9), dtype=np.uint8)
        remote[6, 8] = 1
        for turns, reflected, swap in ((1, False, True), (2, True, False)):
            with self.subTest(turns=turns, reflected=reflected, swap=swap):
                board = transform(cross_board(), turns, reflected, swap)
                marker = transform(remote, turns, reflected)
                side = 2 if swap else 1
                before = board.copy()
                result = native.solve_native_threat(board, side, milliseconds=5000,
                    max_nodes=20000, quiet=1, total=11, width=16)
                self.check_positive(board, side, result, max_nodes=20000, full_graph=True)
                root = result['proof']['certificates'][result['proof']['result']['auditId']]
                self.assertIn(int(np.flatnonzero(marker)[0]), {edge['move'] for edge in root['replies']})
                np.testing.assert_array_equal(board, before)

    def test_immediate_win_has_no_arena_requirement(self):
        board = immediate_board()
        result = native.solve_native_threat(board, 1, milliseconds=1000, max_nodes=100,
            node_capacity=0, edge_capacity=0, line_capacity=0)
        self.check_positive(board, 1, result, max_nodes=100)
        self.assertEqual(result['move'], 2 * 9 + 5)
        self.assertEqual(len(result['line']), 1)
        self.assertEqual(result['certificate_nodes'], 0)

    def test_strict_forcing_open_four_is_not_a_fake_full_graph(self):
        board = open_three_board()
        result = native.solve_native_threat(board, 1, milliseconds=1000,
            max_nodes=1000, quiet=0, total=3)
        counts = self.check_positive(board, 1, result, max_nodes=1000)
        self.assertEqual(result['proof']['kind'], 'forcing_line_certificate')
        self.assertEqual(counts['defender_branches'], 0)
        self.assertGreater(counts['double_win_facts'], 0)
        self.assertEqual(len(result['line']), 3)

    def test_two_independent_fours_have_real_distinct_completion_points(self):
        board = np.full((5, 5), 3, dtype=np.uint8)
        board[2, :] = 0
        board[:, 2] = 0
        for move in ((2, 0), (2, 1), (2, 3), (0, 2), (1, 2), (3, 2)):
            board[move] = 1
        result = native.solve_native_threat(board, 1, milliseconds=1000,
            max_nodes=1000, quiet=0, total=3)
        self.check_positive(board, 1, result, max_nodes=1000)
        self.assertEqual(result['coordinate'], (2, 2))
        self.assertEqual(set(winning_cells(apply_board_move(board, (2, 2), 1), 1)), {(2, 4), (4, 2)})

    def test_shared_completion_two_fours_are_not_an_unstoppable_fork(self):
        board = np.full((5, 5), 3, dtype=np.uint8)
        board[2, :] = 1
        board[:, 2] = 1
        board[2, 2] = board[4, 4] = 0
        self.assertEqual(winning_cells(board, 1), [(2, 2)])
        self.assertEqual(exact_small(board, 2), 0)
        result = native.solve_native_threat(board, 2, milliseconds=1000,
            max_nodes=1000, quiet=2, total=5)
        self.assert_unknown(result)
        after = apply_board_move(board, (2, 2), 2)
        self.assertEqual(winning_cells(after, 1), [])
        self.assert_unknown(native.solve_native_threat(after, 1,
            milliseconds=1000, max_nodes=1000, quiet=2, total=5))

    def test_opponent_immediate_counter_win_preempts_own_three(self):
        board = np.full((7, 9), 3, dtype=np.uint8)
        board[1, :6] = 0
        board[1, 1:5] = 2
        board[5, :7] = 0
        board[5, 2:5] = 1
        self.assertEqual(len(winning_cells(board, 2)), 2)
        self.assert_unknown(native.solve_native_threat(board, 1,
            milliseconds=1000, max_nodes=1000))
        board[5, 1] = 1  # An actual own immediate five wins before the opponent moves.
        result = native.solve_native_threat(board, 1, milliseconds=1000, max_nodes=1000)
        self.check_positive(board, 1, result, max_nodes=1000)
        self.assertEqual(board_winner(apply_board_move(board, result['coordinate'], 1)), 1)

    def test_gray_break_and_rectangular_row_boundary_do_not_make_five(self):
        boards = []
        gray = np.full((5, 8), 3, dtype=np.uint8)
        gray[2, :6] = [1, 1, 3, 1, 1, 0]
        boards.append(gray)
        wrapped = np.full((5, 8), 3, dtype=np.uint8)
        wrapped[1, 6:] = 1
        wrapped[2, :3] = [1, 1, 0]
        boards.append(wrapped)
        for board in boards:
            with self.subTest(board=board.tolist()):
                self.assertEqual(board_winner(board), 0)
                self.assertEqual(winning_cells(board, 1), [])
                self.assert_unknown(native.solve_native_threat(board, 1,
                    milliseconds=1000, max_nodes=100, quiet=1, total=5))

    def test_zero_quiet_still_follows_the_only_mandatory_block(self):
        case = json.loads(FIXTURES.read_text(encoding='utf-8'))['zero_quiet_mandatory']
        board = np.array([[int(x) for x in row] for row in case['board_rows']], dtype=np.uint8)
        expected = tuple(case['unique_block'])
        self.assertEqual(winning_cells(board, 3 - case['side']), [expected])
        before = board.copy()
        result = native.solve_native_threat(board, case['side'], milliseconds=10000,
            max_nodes=100000, quiet=0, total=case['total'], width=16)
        self.check_positive(board, case['side'], result, max_nodes=100000)
        self.assertEqual(result['coordinate'], expected)
        np.testing.assert_array_equal(board, before)

    def test_real_saved_prefixes_each_have_one_independent_full_replay(self):
        fixtures = json.loads(FIXTURES.read_text(encoding='utf-8'))
        for case in fixtures['positive_cases']:
            with self.subTest(case=case['name']):
                source = fixtures['sources'][case['source']]
                board = np.zeros((source['n'], source['n']), dtype=np.uint8)
                for ply, move in enumerate(source['history'][:case['prefix_plies']]):
                    board = apply_board_move(board, tuple(move), 1 + ply % 2)
                self.assertEqual(board_winner(board), 0)
                before = board.copy()
                result = native.solve_native_threat(board, case['side'], milliseconds=10000,
                    max_nodes=case['max_nodes'], quiet=case['quiet'], total=case['total'], width=16)
                self.check_positive(board, case['side'], result,
                    max_nodes=case['max_nodes'], full_graph=True)
                np.testing.assert_array_equal(board, before)
        self.assertTrue(all(case['status'] == 'unknown_not_a_positive_test'
                            for case in fixtures['diagnostic_only']))

    def test_zero_time_or_nodes_do_not_publish_even_an_immediate_win(self):
        for options in (dict(milliseconds=0, max_nodes=100),
                        dict(milliseconds=1000, max_nodes=0)):
            with self.subTest(options=options):
                result = native.solve_native_threat(immediate_board(), 1, **options)
                self.assert_unknown(result)
                self.assertEqual(result['nodes'], 0)
                self.assertTrue(result['exhausted'])
                self.assertEqual(result['status'], 2)

    def test_small_node_caps_cannot_publish_an_incomplete_cross(self):
        for cap in (1, 2, 3):
            with self.subTest(cap=cap):
                result = native.solve_native_threat(cross_board(), 1,
                    milliseconds=1000, max_nodes=cap, quiet=1, total=11)
                self.assert_unknown(result)
                self.assertLessEqual(result['nodes'], cap)
                self.assertTrue(result['exhausted'])

    def test_host_deadline_interrupts_without_turning_unknown_into_loss(self):
        ticks = count(0.0, 10.0)
        # A deliberately expired host deadline tests interruption, not speed.
        with patch.object(native, '_clock_ms', side_effect=lambda: next(ticks)):
            result = native.solve_native_threat(cross_board(), 1,
                milliseconds=1, max_nodes=20000, quiet=1, total=11)
        self.assert_unknown(result)
        self.assertTrue(result['exhausted'])
        self.assertEqual(result['status'], 2)
        self.assertLessEqual(result['nodes'], 20000)

    def test_exhaustion_of_each_certificate_arena_fails_closed(self):
        for name in ('node_capacity', 'edge_capacity', 'line_capacity'):
            with self.subTest(capacity=name):
                result = native.solve_native_threat(cross_board(), 1,
                    milliseconds=5000, max_nodes=20000, quiet=1, total=11,
                    **{name: 0})
                self.assert_unknown(result)
                self.assertTrue(result['exhausted'])
                self.assertEqual(result['status'], 3)

    def test_context_reuse_does_not_retain_previous_proof_or_board(self):
        first = native.solve_native_threat(immediate_board(), 1,
            milliseconds=1000, max_nodes=100)
        self.check_positive(immediate_board(), 1, first, max_nodes=100)
        saved = deepcopy(first)
        unknown = native.solve_native_threat(cross_board(), 1, max_nodes=0)
        self.assert_unknown(unknown)
        board = transform(immediate_board(), 1, True, True)
        before = board.copy()
        last = native.solve_native_threat(board, 2, milliseconds=1000, max_nodes=100)
        self.check_positive(board, 2, last, max_nodes=100)
        self.assertEqual(first, saved)  # Decoded results do not alias arena memory.
        np.testing.assert_array_equal(board, before)

    def test_terminal_state_is_proved_without_inventing_an_action(self):
        board = apply_board_move(immediate_board(), (2, 5), 1)
        result = native.solve_native_threat(board, 1, milliseconds=1000, max_nodes=100)
        self.assertEqual(result['value'], 1)
        self.assertIsNone(result['move'])
        self.assertIsNone(result['coordinate'])
        self.assertEqual(result['line'], [])
        self.assertEqual(result['proof']['kind'], 'terminal_state')
        counts = native.verify_native_threat_result(board, 1, result)
        self.assertEqual(counts['moves_replayed'], 0)
        self.assert_unknown(native.solve_native_threat(board, 2,
            milliseconds=1000, max_nodes=100))
        self.assert_unknown(native.solve_native_threat(np.full((5, 7), 3, dtype=np.uint8), 1,
            milliseconds=1000, max_nodes=100))
        self.assert_unknown(native.solve_native_threat(board, 1, max_nodes=0))

    def test_certificate_false_does_not_expose_an_auditable_proof(self):
        result = native.solve_native_threat(immediate_board(), 1,
            milliseconds=1000, max_nodes=100, certificate=False)
        self.assertEqual(result['value'], 1)
        self.assertIsNone(result['proof'])
        self.assertFalse(result['certificateDecoded'])
        with self.assertRaises(ValueError):
            native.verify_native_threat_result(immediate_board(), 1, result)

    def test_clock_exception_and_nonfinite_clock_discard_result_then_recover(self):
        for replacement in (dict(side_effect=OSError('test clock failure')),
                            dict(return_value=float('nan'))):
            with self.subTest(replacement=replacement):
                with patch.object(native, '_clock_ms', **replacement):
                    with self.assertRaisesRegex(RuntimeError, 'clock callback failed'):
                        native.solve_native_threat(immediate_board(), 1,
                            milliseconds=1000, max_nodes=100)
                self.assertEqual(native.solve_native_threat(immediate_board(), 1,
                    milliseconds=1000, max_nodes=100)['value'], 1)

    def test_bad_ffi_return_code_and_pointer_do_not_escape_as_proofs(self):
        library = native.native_threat_library()
        with patch.object(library, 'native_threat_solve', return_value=7):
            with self.assertRaisesRegex(RuntimeError, 'rejected request'):
                native.solve_native_threat(immediate_board(), 1)
        with patch.object(library, 'native_threat_board',
                          return_value=ctypes.POINTER(ctypes.c_ubyte)()):
            with self.assertRaisesRegex(RuntimeError, 'invalid context pointer'):
                native.solve_native_threat(immediate_board(), 1)
        self.assertEqual(native.solve_native_threat(immediate_board(), 1)['value'], 1)

    def test_certificate_tampering_cannot_hide_a_legal_reply(self):
        original = small_cross_result()
        self.assertEqual(original['value'], 1)
        for corruption in ('missing_reply', 'duplicate_reply', 'wrong_board', 'wrong_actor',
                           'false_count', 'cycle', 'detached_node'):
            with self.subTest(corruption=corruption):
                result = deepcopy(original)
                proof = result['proof']
                root = proof['certificates'][proof['result']['auditId']]
                if corruption == 'missing_reply':
                    root['replies'].pop()
                    root['certifiedReplies'] -= 1
                    proof['result']['certifiedReplies'] -= 1
                elif corruption == 'duplicate_reply':
                    root['replies'][1] = deepcopy(root['replies'][0])
                elif corruption == 'wrong_board':
                    root['board'][root['move']] = 0
                elif corruption == 'wrong_actor':
                    root['side'] = 2
                elif corruption == 'false_count':
                    root['certifiedReplies'] += 1
                elif corruption == 'cycle':
                    root['replies'][0]['auditId'] = proof['result']['auditId']
                else:
                    proof['certificates'].append(deepcopy(root))
                with self.assertRaises(ValueError):
                    native.verify_native_threat_result(cross_board(), 1, result)

    def test_forcing_certificate_identity_and_line_tampering_are_rejected(self):
        board = immediate_board()
        original = native.solve_native_threat(board, 1, milliseconds=1000, max_nodes=100)
        for corruption in ('board', 'side', 'move', 'line'):
            with self.subTest(corruption=corruption):
                result = deepcopy(original)
                if corruption == 'board':
                    result['proof']['board'][0][0] = 0
                elif corruption == 'side':
                    result['proof']['side'] = 2
                elif corruption == 'move':
                    result['move'] = 0
                else:
                    result['line'] = []
                with self.assertRaises(ValueError):
                    native.verify_native_threat_result(board, 1, result)


if __name__ == '__main__':
    unittest.main()
