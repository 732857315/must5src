"""Search and evaluation regressions; no network files or training needed."""

import contextlib
from functools import lru_cache
import io
import random
import unittest
from unittest.mock import patch

import numpy as np

import arena
from game import BLACK, WHITE, EMPTY, FORBIDDEN, CENTER_INDEX, initial_state, legal_moves, winner


def pack(black=(), white=(), forbidden=()):
    return (sum(BLACK << (i * 2) for i in black)
            | sum(WHITE << (i * 2) for i in white)
            | sum(FORBIDDEN << (i * 2) for i in forbidden))


def reference_winner(cells):
    lines = [cells[r * 5:r * 5 + 5] for r in range(5)]
    lines += [cells[c::5] for c in range(5)]
    lines += [tuple(cells[i * 6] for i in range(5)), tuple(cells[4 + i * 4] for i in range(5))]
    for line in lines:
        if line[0] in (BLACK, WHITE) and len(set(line)) == 1:
            return line[0]
    return EMPTY


@lru_cache(maxsize=None)
def reference_value(cells, me):
    result = reference_winner(cells)
    if result:
        return 1 if result == me else -1
    moves = [i for i, cell in enumerate(cells) if cell == EMPTY]
    if not moves:
        return 0
    return max(-reference_value(cells[:i] + (me,) + cells[i + 1:], 3 - me) for i in moves)


def late_positions(count=30):
    rng = random.Random(763)
    positions = []
    while len(positions) < count:
        cells = [EMPTY] * 25
        cells[CENTER_INDEX] = BLACK
        me = WHITE
        for _ in range(19):
            moves = [i for i, cell in enumerate(cells) if not cell]
            cells[rng.choice(moves)] = me
            me = 3 - me
            if reference_winner(tuple(cells)):
                break
        else:
            cells = tuple(cells)
            state = sum(cell << (i * 2) for i, cell in enumerate(cells))
            positions.append((state, cells, me))
    return positions


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.player = arena.PlayerAB(None, "test", depth=5)

    def test_pattern_score_counts_each_line_once_and_is_bounded(self):
        state = pack(black=(0, 1, 2, 3))
        self.assertAlmostEqual(self.player._score(state, BLACK), 10 / 121)
        self.assertAlmostEqual(self.player._score(state, WHITE), -10 / 121)
        for state, _, me in late_positions():
            self.assertLess(abs(self.player._score(state, me)), 1)
            self.assertEqual(self.player._score(state, me), -self.player._score(state, 3 - me))

    def test_forbidden_cells_cancel_the_whole_pattern_line(self):
        for owner in (BLACK, WHITE):
            state = pack(black=(0, 1, 2, 3), forbidden=(4,)) if owner == BLACK else pack(
                white=(0, 1, 2, 3), forbidden=(4,))
            self.assertEqual(winner(state), EMPTY)
            self.assertEqual(self.player._score(state, BLACK), 0)
            self.assertEqual(self.player._score(state, WHITE), 0)
        self.assertEqual(self.player._score(pack(forbidden=range(25)), BLACK), 0)

    def test_forbidden_threat_is_ignored_and_real_threat_is_blocked(self):
        state = pack(white=(0, 1, 2, 3, 20, 21, 22, 23), forbidden=(4,))
        wins, blocks, rest = self.player._ordered(state, BLACK)
        self.assertEqual(wins, [])
        self.assertEqual(blocks, [24])
        self.assertNotIn(4, rest)
        self.assertEqual(self.player.move(state, BLACK), 24)

    def test_completely_blocked_board_is_a_draw_without_a_move(self):
        state = pack(forbidden=range(25))
        self.assertEqual(self.player._ab(state, BLACK, 3), 0)
        self.assertEqual(self.player._ab(state, WHITE, 3), 0)
        self.assertIsNone(self.player.move(state, BLACK))

    def test_model_and_search_never_choose_forbidden_cells(self):
        state = pack(black=(CENTER_INDEX,), forbidden=(24,))
        policy = np.arange(25, dtype=float)
        with patch("arena.ncnn_infer", return_value=(policy, np.array([0.0]))):
            self.assertEqual(arena.Player(None, "bit", "raw").move(state, WHITE), 23)
        self.player.depth = 1
        with patch.object(self.player, "_cnn_logits", return_value=policy):
            self.assertIn(self.player.move(state, WHITE), legal_moves(state))

    def test_terminal_result_uses_side_to_move(self):
        state = pack(black=range(5))
        self.assertEqual(self.player._ab(state, BLACK, 0), 1)
        self.assertEqual(self.player._ab(state, WHITE, 0), -1)
        self.assertIsNone(self.player.move(state, WHITE))
        self.assertIsNone(arena.Player(None, "bit", "raw").move(state, WHITE))

    def test_immediate_win_precedes_defense(self):
        state = pack(black=(0, 1, 2, 3), white=(20, 21, 22, 23))
        self.assertEqual(self.player.move(state, BLACK), 4)
        self.assertEqual(arena.Player(None, "two", "tactical", tactical=True).move(state, BLACK), 4)

    def test_single_block_is_mandatory_even_at_depth_one(self):
        state = pack(black=(0, 1, 2), white=(20, 21, 22, 23))
        self.player.depth = 1
        self.assertEqual(self.player.move(state, BLACK), 24)
        self.assertEqual(arena.Player(None, "bit", "tactical", tactical=True).move(state, BLACK), 24)

    def test_two_distinct_threats_are_a_proven_loss(self):
        state = pack(black=(12,), white=(0, 1, 2, 3, 5, 6, 7, 8))
        self.assertEqual(self.player._ab(state, BLACK, 0), -1)

    def test_full_search_matches_independent_exhaustive_solver(self):
        with patch.object(self.player, "_cnn_logits", return_value=np.arange(25, dtype=float)):
            for state, cells, me in late_positions():
                with self.subTest(state=state):
                    expected = reference_value(cells, me)
                    self.player.cache.clear()
                    self.assertEqual(self.player._ab(state, me, 5), expected)
                    move = self.player.move(state, me)
                    self.assertEqual(cells[move], EMPTY)
                    child = cells[:move] + (me,) + cells[move + 1:]
                    self.assertEqual(-reference_value(child, 3 - me), expected)

    def test_cached_bounds_do_not_become_exact_values(self):
        saw_bound = False
        for state, cells, me in late_positions():
            self.player.cache.clear()
            expected = reference_value(cells, me)
            self.player._ab(state, me, 5, -1.0, -0.5)
            saw_bound |= self.player.cache[(state, me, 5)][1] != "exact"
            self.assertEqual(self.player._ab(state, me, 5), expected)
            self.player.cache.clear()
            self.player._ab(state, me, 5, 0.5, 1.0)
            self.assertEqual(self.player._ab(state, me, 5), expected)
        self.assertTrue(saw_bound)

    def test_policy_can_only_select_empty_cells(self):
        player = arena.Player(None, "bit", "raw")
        policy = np.arange(25, dtype=float)
        policy[CENTER_INDEX] = 1e9
        with patch("arena.ncnn_infer", return_value=(policy, np.array([0.0]))):
            self.assertEqual(player.move(initial_state(), WHITE), 24)

    def test_depth_must_be_positive(self):
        for depth in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                arena.PlayerAB(None, "test", depth)


class FirstLegalPlayer:
    def __init__(self, name, reverse=False):
        self.name = name
        self.reverse = reverse

    def move(self, state, me):
        return legal_moves(state)[-1 if self.reverse else 0]


class EvaluationTests(unittest.TestCase):
    def test_openings_are_reproducible_legal_and_symmetry_unique(self):
        openings = arena.generate_openings(50, seed=41)
        self.assertEqual(openings, arena.generate_openings(50, seed=41))
        self.assertNotEqual(openings, arena.generate_openings(50, seed=42))
        states = []
        for opening in openings:
            state, me, transcript = arena._opening_position(opening)
            self.assertEqual(len(opening), 4)
            self.assertEqual(me, WHITE)
            self.assertEqual(len(transcript), 4)
            self.assertEqual(winner(state), EMPTY)
            self.assertEqual(len(legal_moves(state)), 20)
            states.append(arena._canonical_state(state))
        self.assertEqual(len(set(states)), 50)

    def test_each_opening_is_paired_with_models_swapping_colors(self):
        mine, repo = FirstLegalPlayer("mine"), FirstLegalPlayer("repo", True)
        openings = arena.generate_openings(3, seed=7)
        with patch("arena.play_game", return_value=(BLACK, [(WHITE, 0)])) as play:
            stats = arena.evaluate(mine, repo, openings)
        self.assertEqual((stats["games"], stats["pairs"], stats["unique_openings"]), (6, 3, 3))
        self.assertEqual((stats["W"], stats["L"], stats["Wb"], stats["Ww"]), (3, 3, 3, 0))
        for i, opening in enumerate(openings):
            self.assertEqual(play.call_args_list[2 * i].args, (mine, repo, False, opening))
            self.assertEqual(play.call_args_list[2 * i + 1].args, (repo, mine, False, opening))
        self.assertEqual(stats["unique_transcripts"], 1)

    def test_repeated_openings_are_reported_as_one_independent_position(self):
        mine, repo = FirstLegalPlayer("mine"), FirstLegalPlayer("repo")
        stats = arena.evaluate(mine, repo, [(), ()])
        self.assertEqual(stats["games"], 4)
        self.assertEqual(stats["unique_openings"], 1)
        self.assertEqual(stats["unique_transcripts"], 1)

    def test_illegal_player_moves_fail_before_corrupting_the_board(self):
        player = FirstLegalPlayer("bad-player")
        for move in (None, CENTER_INDEX, -1, 25, True, 1.5):
            with self.subTest(move=move), patch.object(player, "move", return_value=move):
                with self.assertRaisesRegex(ValueError, "bad-player returned illegal move"):
                    arena.play_game(player, player)

    def test_illegal_move_reports_the_shared_specific_reason(self):
        state = pack(black=(12,), forbidden=(4,))
        for move, reason in ((4, "边界或禁下格"), (12, "已有棋子"), (25, "越界")):
            with self.subTest(move=move), self.assertRaisesRegex(ValueError, reason):
                arena._checked_move(state, move, WHITE, "test-player")

    def test_terminal_opening_does_not_ask_a_player_to_move(self):
        # White completes the top row; black placements stay scattered.
        opening = (0, 5, 1, 7, 2, 9, 3, 16, 4)
        player = FirstLegalPlayer("unused")
        with patch.object(player, "move", side_effect=AssertionError("terminal move")):
            result, transcript = arena.play_game(player, player, opening=opening)
        self.assertEqual(result, WHITE)
        self.assertEqual(len(transcript), 9)
        with self.assertRaisesRegex(ValueError, "continues after"):
            arena.play_game(player, player, opening=opening + (20,))

    def test_cli_keeps_positionals_and_applies_assistance_symmetrically(self):
        base = ["mine.param", "mine.bin", "repo.param", "repo.bin", "2"]
        stats = dict(games=2, pairs=1, unique_openings=1, unique_transcripts=2,
                     W=0, L=0, D=2, Wb=0, Ww=0)
        for flags in ([], ["--raw"], ["--hint"], ["--ab", "--depth", "2"]):
            with self.subTest(flags=flags), patch("arena.load_net", return_value=None), \
                 patch("arena.evaluate", return_value=stats) as evaluate, \
                 contextlib.redirect_stdout(io.StringIO()):
                arena.main(base + flags)
                mine, repo = evaluate.call_args.args[:2]
                self.assertEqual(mine.tactical, repo.tactical)
                self.assertEqual(mine.hint, repo.hint)
                self.assertEqual((mine.enc_type, repo.enc_type), ("bit", "two"))
                if "--ab" in flags:
                    self.assertIsInstance(mine, arena.PlayerAB)
                    self.assertIsInstance(repo, arena.PlayerAB)
                    self.assertEqual((mine.depth, repo.depth), (2, 2))
                if "--raw" in flags:
                    self.assertFalse(mine.tactical)
                    self.assertFalse(mine.hint)

    def test_unpaired_cli_and_impossible_opening_counts_fail_clearly(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arena.main(["a", "b", "c", "d", "3"])
        self.assertEqual(arena.generate_openings(1, plies=0), [()])
        with self.assertRaisesRegex(ValueError, "only one"):
            arena.generate_openings(2, plies=0)


if __name__ == "__main__":
    unittest.main()
