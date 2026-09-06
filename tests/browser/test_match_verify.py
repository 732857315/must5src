"""Browser evidence validation without opening a browser or loading a model."""
import copy
from pathlib import Path
import unittest

from tools.browser.match_verify import (ROOT, verify_browser_state, verify_extension,
                                       verify_terminal_game, winner, trajectory_key)

N = 16


def raw(indices=(), ai_side=2, **changes):
    indices = list(indices)
    board = [0] * 256
    for index, point in enumerate(indices):
        board[point] = 1 + index % 2
    result = dict(n=16, board=board, history=indices, human=3-ai_side, seconds=1.0,
                  started=True, ready=True, busy=False, revision=len(indices)+1, analysis=None)
    result.update(changes)
    return result


def state(indices=(), ai_side=2):
    return verify_browser_state(raw(indices, ai_side), ai_side)


def interleave(black, white):
    moves = []
    for index, move in enumerate(black):
        moves.append(move)
        if index < len(white):
            moves.append(white[index])
    return moves


BLACK_WIN = interleave([0, 1, 2, 3, 4], [32, 34, 36, 38])
WHITE_WIN = interleave([0, 2, 4, 6, 8], [48, 49, 50, 51, 52])


def draw_history():
    black, white = [], []
    for row in range(N):
        for col in range(N):
            (black if (row + 2*col) % 4 < 2 else white).append(row*N + col)
    return interleave(black, white)


class BrowserStateTests(unittest.TestCase):
    def test_root_and_fresh_normalized_history(self):
        self.assertEqual(ROOT, Path(__file__).resolve().parents[2])
        original = raw([136, 137], ai_side=2)
        result = verify_browser_state(original, 2, require_human_turn=True)
        self.assertEqual(result["history"], [dict(row=8, col=8, side=1, by="human"),
                                            dict(row=8, col=9, side=2, by="ai")])
        self.assertEqual((result["plies"], result["winner"], result["terminal"]), (2, 0, False))
        result["board"][8][8] = 0
        result["history"][0]["row"] = 0
        self.assertEqual(original["board"][136], 1)
        self.assertEqual(original["history"], [136, 137])

    def test_legacy_and_explicit_square_columns_have_identical_results(self):
        legacy = verify_browser_state(raw([136, 137]), 2)
        explicit = verify_browser_state(raw([136, 137], cols=16), 2)
        self.assertEqual(explicit, legacy)
        self.assertNotIn("cols", explicit)
        self.assertNotIn("n", explicit)
        old_terminal = verify_terminal_game(verify_browser_state(raw(BLACK_WIN), 2))
        new_terminal = verify_terminal_game(verify_browser_state(raw(BLACK_WIN, cols=16), 2))
        self.assertEqual(old_terminal, new_terminal)

    def test_formal_columns_reject_rectangles_and_noninteger_metadata(self):
        for cols in (True, False, 16.0, None, "16", 0, 12, 17):
            # A forged 256-cell buffer must not hide incompatible geometry.
            with self.subTest(cols=cols), self.assertRaises(ValueError):
                verify_browser_state(raw(cols=cols), 2)
        for rows, cols in ((8, 12), (16, 12), (12, 16)):
            with self.subTest(rows=rows, cols=cols), self.assertRaises(ValueError):
                verify_browser_state(raw(n=rows, cols=cols, board=[0]*(rows*cols)), 2)

    def test_black_and_white_ai_human_turns_include_incomplete_reply(self):
        verify_browser_state(raw([], 2), 2, require_human_turn=True)
        verify_browser_state(raw([136], 1), 1, require_human_turn=True)
        verify_browser_state(raw([136], 2, busy=True), 2)
        for snapshot, side in ((raw([136], 2), 2), (raw([], 1), 1),
                               (raw([], 2, busy=True), 2), (raw([], 2, ready=False), 2)):
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                verify_browser_state(snapshot, side, require_human_turn=True)

    def test_storage_failure_or_pending_commit_cannot_be_accepted(self):
        baseline = verify_browser_state(raw([136, 137]), 2)
        for changes in ({}, dict(storageBlocked=False, storageError=None, pendingCommit=None),
                        dict(storageError="")):
            self.assertEqual(verify_browser_state(raw([136, 137], **changes), 2), baseline)
        for changes in (dict(storageBlocked=True), dict(storageBlocked=1),
                        dict(storageBlocked=None), dict(storageError="QuotaExceeded"),
                        dict(storageError=False), dict(storageError=[]),
                        dict(pendingCommit={}), dict(pendingCommit={"move": 138}),
                        dict(pendingCommit=False), dict(pendingCommit=[])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                verify_browser_state(raw([136, 137], **changes), 2)
        with self.assertRaises(ValueError):
            verify_browser_state(raw(BLACK_WIN, storageBlocked=True), 2)

    def test_prestart_empty_snapshot_allowed_but_cannot_claim_played_moves(self):
        verify_browser_state(raw([], started=False, ready=False), 2)
        with self.assertRaises(ValueError):
            verify_browser_state(raw([0], started=False), 2)
        with self.assertRaises(ValueError):
            verify_browser_state(raw([], started=False), 2, require_human_turn=True)

    def test_boolean_or_noninteger_wire_numbers_are_rejected(self):
        bad = [
            dict(n=True), dict(n=16.0), dict(n=15), dict(human=True), dict(human=1.0),
            dict(revision=True), dict(revision=-1), dict(revision=1.5),
            dict(seconds=True), dict(seconds=0), dict(seconds=float("nan")), dict(seconds=float("inf")),
            dict(started=1), dict(ready=0), dict(busy="false"), dict(analysis=[]),
        ]
        for changes in bad:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                verify_browser_state(raw(**changes), 2)
        for side in (True, 2.0, 0, 3):
            with self.subTest(side=side), self.assertRaises(ValueError):
                verify_browser_state(raw(), side)
        for expected in (True, 0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                verify_browser_state(raw(), 2, seconds=expected)
        with self.assertRaises(ValueError):
            verify_browser_state(raw(), 2, require_human_turn=1)

    def test_board_shape_cells_and_history_encoding_are_strict(self):
        for board in ([0]*255, [[0]*16 for _ in range(16)], [False]+[0]*255,
                      [0.0]+[0]*255, [3]+[0]*255, [-1]+[0]*255):
            with self.subTest(board=board[:2]), self.assertRaises(ValueError):
                verify_browser_state(raw(board=board), 2)
        for history in ([True], [1.0], [-1], [256], "0", None):
            with self.subTest(history=history), self.assertRaises(ValueError):
                verify_browser_state(raw(history=history), 2)
        for value in (None, [], "snapshot"):
            with self.assertRaises(ValueError):
                verify_browser_state(value, 2)
        incomplete = raw()
        del incomplete["analysis"]
        with self.assertRaises(ValueError):
            verify_browser_state(incomplete, 2)

    def test_budget_identity_and_board_history_tampering_are_rejected(self):
        verify_browser_state(raw(seconds=.5), 2, seconds=.5)
        for snapshot in (raw(seconds=.5), raw(human=2), raw([0, 0]),
                         raw([0, 1], board=[0]*256), raw([1, 0], board=raw([0, 1])["board"])):
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                verify_browser_state(snapshot, 2)

    def test_terminal_after_nine_moves_and_no_post_terminal_extension(self):
        result = state(BLACK_WIN)
        self.assertEqual((result["winner"], result["terminal"], result["plies"]), (1, True, 9))
        with self.assertRaisesRegex(ValueError, "after the game ended"):
            state(BLACK_WIN + [100])
        verify_browser_state(raw(BLACK_WIN, busy=True), 2, require_human_turn=True)

    def test_six_in_row_and_diagonal_are_wins_without_edge_wrapping(self):
        six = interleave([0, 1, 2, 4, 5, 3], [32, 34, 36, 38, 40])
        diagonal = interleave([4, 19, 34, 49, 64], [128, 130, 132, 134])
        wrapped = interleave([14, 15, 16, 17, 18], [96, 98, 100, 102])
        self.assertEqual(state(six)["winner"], 1)
        self.assertEqual(state(diagonal)["winner"], 1)
        self.assertFalse(state(wrapped)["terminal"])


class ExtensionTests(unittest.TestCase):
    def test_human_click_may_have_pending_or_completed_ai_reply(self):
        before = state([], ai_side=2)
        pending = state([136], ai_side=2)
        complete = state([136, 137], ai_side=2)
        self.assertEqual(verify_extension(before, pending, (8, 8)), pending)
        self.assertEqual(verify_extension(before, complete, [8, 8]), complete)
        self.assertEqual(verify_extension(pending, complete), complete)
        self.assertEqual(verify_extension(complete, complete), complete)

    def test_white_human_click_and_black_ai_initial_move(self):
        empty = state([], ai_side=1)
        opening = state([136], ai_side=1)
        after = state([136, 137, 120], ai_side=1)
        verify_extension(empty, opening)
        verify_extension(opening, after, (8, 9))
        with self.assertRaises(ValueError):
            verify_extension(empty, opening, (8, 8))

    def test_prefix_rollback_replacement_and_second_round_are_rejected(self):
        for before, after, click in ((state([0, 1]), state([]), None),
                                      (state([0, 1]), state([2, 3]), None),
                                      (state([]), state([0, 1, 2]), (0, 0)),
                                      (state([]), state([0, 1, 2]), None),
                                      (state([0]), state([0, 1, 2]), None)):
            with self.subTest(click=click), self.assertRaises(ValueError):
                verify_extension(before, after, click)
        with self.assertRaises(ValueError):
            verify_extension(state([], 1), state([0], 2))

    def test_click_requires_first_human_move_with_matching_coordinates(self):
        before, after = state([]), state([0, 1])
        for move in ((0, 1), (True, 0), (0.0, 0), (-1, 0), (0, 16), [0], "0,0"):
            with self.subTest(move=move), self.assertRaises(ValueError):
                verify_extension(before, after, move)
        with self.assertRaises(ValueError):
            verify_extension(before, before, (0, 0))

    def test_terminal_human_move_needs_no_ai_reply(self):
        before, after = state(BLACK_WIN[:-1]), state(BLACK_WIN)
        verify_extension(before, after, (0, 4))
        verify_extension(after, after)
        with self.assertRaises(ValueError):
            verify_extension(after, after, (4, 4))

    def test_normalized_inputs_are_revalidated_instead_of_trusted(self):
        before = state([])
        valid = state([0, 1])
        bad_rows = []
        for field, value in (("winner", True), ("terminal", 1), ("plies", True),
                             ("winner", 1), ("terminal", True), ("plies", 4)):
            changed = copy.deepcopy(valid)
            changed[field] = value
            bad_rows.append(changed)
        for field, value in (("side", 2), ("side", True), ("by", "ai"), ("row", 1)):
            changed = copy.deepcopy(valid)
            changed["history"][0][field] = value
            bad_rows.append(changed)
        for changed in bad_rows:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                verify_extension(before, changed)


class TerminalTests(unittest.TestCase):
    def test_both_colors_win_and_lose_based_on_actual_winner(self):
        for history, won in ((BLACK_WIN, 1), (WHITE_WIN, 2)):
            for ai_side in (1, 2):
                with self.subTest(winner=won, ai_side=ai_side):
                    verified = state(history, ai_side)
                    result = verify_terminal_game(verified)
                    self.assertEqual(result["winner"], won)
                    self.assertEqual(result["outcome"], "win" if ai_side == won else "loss")
                    self.assertEqual(result["termination"], "five_or_more")
                    self.assertEqual(result["canonical_trajectory"], trajectory_key(verified["history"], ai_side))

    def test_full_board_without_five_is_draw_and_one_empty_is_not(self):
        history = draw_history()
        self.assertEqual(len(history), 256)
        full = state(history)
        self.assertEqual(winner(full["board"]), 0)
        result = verify_terminal_game(full)
        self.assertEqual((result["outcome"], result["termination"], result["plies"]), ("draw", "board_full", 256))
        partial = state(history[:-1])
        self.assertFalse(partial["terminal"])
        with self.assertRaises(ValueError):
            verify_terminal_game(partial)

    def test_search_claims_cannot_turn_a_live_board_into_a_result(self):
        snapshot = raw([0, 1], analysis={"search": {"proven_value": 1}, "value": 1.0, "terminal": True})
        verified = verify_browser_state(snapshot, 2)
        self.assertFalse(verified["terminal"])
        with self.assertRaises(ValueError):
            verify_terminal_game(verified)
        forged = dict(verified, winner=2, terminal=True)
        with self.assertRaises(ValueError):
            verify_terminal_game(forged)

    def test_rotated_complete_game_has_same_canonical_identity(self):
        original = verify_terminal_game(state(BLACK_WIN, 1))
        rotated = [col*16 + 15-row for row, col in (divmod(point, 16) for point in BLACK_WIN)]
        self.assertEqual(original["canonical_trajectory"],
                         verify_terminal_game(state(rotated, 1))["canonical_trajectory"])


if __name__ == "__main__":
    unittest.main()
