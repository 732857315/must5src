"""Boundary/forbidden two-bit rules; no model inference, training, or file output."""

import random
import unittest

from game import (
    BLACK, WHITE, EMPTY, FORBIDDEN, CELL_COUNT, LINE_INDICES, Board,
    apply_move, empty_count, forbidden_moves, legal_moves, move_rejection_reason,
    player_at, stone_count, validate_state, winner,
)


def pack_cells(cells):
    return sum(cell << (2 * i) for i, cell in enumerate(cells))


class BoundaryRulesTests(unittest.TestCase):
    def test_forbidden_is_a_valid_two_bit_value_and_roundtrips(self):
        rng = random.Random(61)
        for _ in range(40):
            cells = [rng.randrange(4) for _ in range(CELL_COUNT)]
            state = pack_cells(cells)
            validate_state(state)
            board = Board(cells)
            self.assertEqual(board.pack(), state)
            self.assertEqual(list(Board.unpack(state).grid), cells)
        validate_state((1 << 50) - 1)
        for bad_state in (-1, 1 << 50, True):
            with self.assertRaises(ValueError):
                validate_state(bad_state)

    def test_forbidden_interrupts_every_winning_line_for_either_player(self):
        for line in LINE_INDICES:
            for side in (BLACK, WHITE):
                complete = sum(side << (2 * i) for i in line)
                self.assertEqual(winner(complete), side)
                for blocked in line:
                    state = complete | (FORBIDDEN << (2 * blocked))
                    self.assertEqual(winner(state), EMPTY)
                    self.assertEqual(Board.unpack(state).winner(), EMPTY)
            all_forbidden = sum(FORBIDDEN << (2 * i) for i in line)
            self.assertEqual(winner(all_forbidden), EMPTY)

    def test_forbidden_outside_a_winning_line_does_not_cancel_the_win(self):
        for side in (BLACK, WHITE):
            state = sum(side << (2 * i) for i in range(5)) | (FORBIDDEN << 48)
            self.assertEqual(winner(state), side)

    def test_counts_and_turn_ignore_forbidden_cells(self):
        rng = random.Random(91)
        for _ in range(60):
            cells = [rng.randrange(4) for _ in range(CELL_COUNT)]
            state = pack_cells(cells)
            stones = sum(cell in (BLACK, WHITE) for cell in cells)
            self.assertEqual(stone_count(state), stones)
            self.assertEqual(empty_count(state), cells.count(EMPTY))
            self.assertEqual(player_at(state), BLACK if stones % 2 == 0 else WHITE)
            self.assertEqual(forbidden_moves(state), [i for i, cell in enumerate(cells) if cell == FORBIDDEN])
            self.assertEqual(legal_moves(state), [i for i, cell in enumerate(cells) if cell == EMPTY])
        boundary_only = FORBIDDEN | (FORBIDDEN << 48)
        self.assertEqual(player_at(boundary_only), BLACK)
        self.assertEqual(player_at(boundary_only | (BLACK << 24)), WHITE)

    def test_cannot_play_on_forbidden_cells(self):
        state = (FORBIDDEN << 6) | (BLACK << 24)
        self.assertNotIn(3, legal_moves(state))
        self.assertNotIn(3, Board.unpack(state).legal_moves())
        reason = move_rejection_reason(state, 3)
        self.assertIn("边界或禁下格", reason)
        for side in (BLACK, WHITE):
            with self.assertRaisesRegex(ValueError, "边界或禁下格"):
                apply_move(state, 3, side)
            with self.assertRaisesRegex(ValueError, "边界或禁下格"):
                Board.unpack(state).apply(3, side)
        self.assertIsNone(move_rejection_reason(state, 4))
        moved = apply_move(state, 4, WHITE)
        self.assertEqual(forbidden_moves(moved), [3])
        self.assertEqual(stone_count(moved), 2)

    def test_move_reasons_distinguish_boundary_occupied_terminal_and_outside(self):
        for move in (-1, 25, True, 2.5):
            self.assertIn("越界", move_rejection_reason(0, move))
        state = BLACK << 24
        self.assertIn("已有棋子", move_rejection_reason(state, 12))
        won = sum(WHITE << (2 * i) for i in range(5))
        self.assertIn("棋局已结束", move_rejection_reason(won, 5))
        with self.assertRaisesRegex(ValueError, "棋局已结束"):
            apply_move(won, 5, BLACK)

    def test_fully_forbidden_board_is_terminal_without_a_winner(self):
        state = (1 << 50) - 1
        self.assertEqual(stone_count(state), 0)
        self.assertEqual(empty_count(state), 0)
        self.assertEqual(legal_moves(state), [])
        self.assertEqual(winner(state), EMPTY)
        self.assertTrue(Board.unpack(state).is_terminal())

    def test_board_set_and_display_boundary_options_do_not_change_encoding(self):
        board = Board()
        board.set(0, 0, FORBIDDEN)
        board.set(2, 2, BLACK)
        state = board.pack()
        plain = board.show().splitlines()
        self.assertEqual(len(plain), 5)
        self.assertTrue(all(len(row) == 5 for row in plain))
        self.assertEqual(plain[0], "#....")
        framed = board.show(show_boundary=True).splitlines()
        self.assertEqual(len(framed), 7)
        self.assertEqual(framed[0], "#######")
        self.assertEqual(framed[-1], "#######")
        self.assertEqual(framed[1], "##....#")
        self.assertIn("# 边界/禁下", board.show(show_legend=True))
        self.assertEqual(board.pack(), state)
        self.assertEqual(len(board.grid), CELL_COUNT)
        with self.assertRaises(ValueError):
            board.set(0, 0, 4)


if __name__ == "__main__":
    unittest.main()
