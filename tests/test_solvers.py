"""Small, independent endgame checks; no training or dataset files are written."""

from functools import lru_cache
import random
import unittest
from unittest.mock import patch

import exact_solver
import game
import solve_sampler
import solver
import solver_guard


def packed(rows):
    cells = "".join(rows.split())
    if len(cells) != 25:
        raise ValueError("Fixture must have 25 cells")
    values = {".": 0, "X": 1, "O": 2, "#": 3}
    return sum(values[cell] << (2 * index) for index, cell in enumerate(cells))


WIN_CHOICE = packed("OXO.O XXX.X XOXOX .XXO. .OOOO")
WON_WITH_EMPTY = 747827249894950
DRAW = packed("OXXXX XOOXO XOXXO OOXOO XXXOO")
THREE_EMPTY = DRAW & ~sum(3 << (2 * index) for index in (0, 1, 2))

# This reference deliberately avoids the production board and win helpers.
REFERENCE_LINES = (
    [tuple(r * 5 + c for c in range(5)) for r in range(5)]
    + [tuple(r * 5 + c for r in range(5)) for c in range(5)]
    + [(0, 6, 12, 18, 24), (20, 16, 12, 8, 4)]
)


def reference_moves(state):
    cells = tuple((state >> (2 * i)) & 3 for i in range(25))
    if any(cells[line[0]] in (1, 2) and all(cells[i] == cells[line[0]] for i in line)
           for line in REFERENCE_LINES):
        return cells, None
    return cells, [12] if state == 0 else [i for i, cell in enumerate(cells) if not cell]


@lru_cache(None)
def reference_value(state):
    cells, moves = reference_moves(state)
    if moves is None:
        return -1
    if not moves:
        return 0
    me = 1 if sum(cell in (1, 2) for cell in cells) % 2 == 0 else 2
    return max(-reference_value(state | (me << (2 * move))) for move in moves)


def reference_analysis(state):
    cells, moves = reference_moves(state)
    value = reference_value(state)
    if not moves:
        return value, []
    me = 1 if sum(cell in (1, 2) for cell in cells) % 2 == 0 else 2
    return value, [move for move in moves
                   if -reference_value(state | (me << (2 * move))) == value]


class ExactSolverTests(unittest.TestCase):
    def setUp(self):
        solver.TT.clear()
        solver_guard.TT.clear()
        solve_sampler.TT.clear()
        solver_guard.NODES = 0
        solve_sampler.NODES = 0

    def test_all_legacy_solvers_match_independent_endgame_values(self):
        states = [WIN_CHOICE, WON_WITH_EMPTY, DRAW, THREE_EMPTY]
        rng = random.Random(761)
        for _ in range(12):
            states.extend(state for state in solve_sampler.play_random_game(rng)
                          if game.empty_count(state) <= 4)
        for state in states:
            with self.subTest(state=state):
                expected = reference_value(state)
                self.assertEqual(solver.negamax(game.Board.unpack(state), -1, 1), expected)
                self.assertEqual(solver_guard.negamax(state), expected)
                self.assertEqual(solve_sampler.solve(state), expected)

    def test_labels_include_every_optimal_move_after_a_winning_cutoff(self):
        engine = exact_solver.ExactSolver()
        self.assertEqual(engine.solve(WIN_CHOICE), 1)
        self.assertEqual(engine.analyze(WIN_CHOICE), reference_analysis(WIN_CHOICE))
        labels = solve_sampler.label_samples([WIN_CHOICE])
        self.assertEqual(labels, [(WIN_CHOICE, *reference_analysis(WIN_CHOICE))])

    def test_won_position_with_empty_cells_has_no_label_moves(self):
        self.assertTrue(game.legal_moves(WON_WITH_EMPTY))
        self.assertNotEqual(game.winner(WON_WITH_EMPTY), game.EMPTY)
        self.assertEqual(exact_solver.ExactSolver().analyze(WON_WITH_EMPTY), (-1, []))
        self.assertEqual(solve_sampler.label_samples([WON_WITH_EMPTY]),
                         [(WON_WITH_EMPTY, -1, [])])

    def test_draw_is_an_exact_terminal_value(self):
        self.assertEqual(game.winner(DRAW), game.EMPTY)
        self.assertEqual(exact_solver.ExactSolver().analyze(DRAW), (0, []))

    def test_fully_blocked_board_and_blocked_full_board_are_terminal_draws(self):
        states = [packed("##### ##### ##### ##### #####"),
                  packed("##### ##### ##X## ##### ####O")]
        for state in states:
            with self.subTest(state=state):
                self.assertEqual(game.winner(state), game.EMPTY)
                self.assertEqual(exact_solver.ExactSolver().analyze(state), (0, []))
                self.assertEqual(solver.negamax(game.Board.unpack(state)), 0)
                self.assertEqual(solver_guard.negamax(state), 0)
                self.assertEqual(solve_sampler.solve(state), 0)

    def test_masked_board_without_stones_does_not_force_center(self):
        state = packed("..### #.### ##### ##### #####")
        self.assertEqual(game.stone_count(state), 0)
        self.assertEqual(exact_solver.search_moves(state), [0, 1, 6])
        self.assertEqual(exact_solver.ExactSolver().analyze(state), (0, [0, 1, 6]))

    def test_forbidden_endgames_match_independent_search_and_never_play_boundary(self):
        states = [packed("XXXX# ####. ##### ##### ####."),
                  packed("OOOO. ####X ##### ##### ####."),
                  packed("XOOOX OXXXO #OX#X O#X.O X.O#.")]
        for state in states:
            with self.subTest(state=state):
                expected = reference_analysis(state)
                value, moves = exact_solver.ExactSolver().analyze(state)
                self.assertEqual((value, moves), expected)
                self.assertTrue(all(((state >> (2 * move)) & 3) == 0 for move in moves))
                self.assertEqual(solver.negamax(game.Board.unpack(state)), expected[0])
                self.assertEqual(solver_guard.negamax(state), expected[0])
                self.assertEqual(solve_sampler.label_samples([state]), [(state, *expected)])

    def test_forced_center_opening_is_shared_by_all_solvers(self):
        # Supply a child-value oracle so this checks opening expansion without
        # attempting to solve the 24-empty opening position.
        center = game.initial_state()
        engine = exact_solver.ExactSolver({center: 0})
        self.assertEqual(engine.analyze(0, budget=exact_solver.SearchBudget(1, None)),
                         (0, [game.CENTER_INDEX]))
        self.assertEqual(set(engine.table), {0, center})
        for table in (solver.TT, solver_guard.TT, solve_sampler.TT):
            table[center] = 0
        self.assertEqual(solver.negamax(game.Board(), max_nodes=1), 0)
        self.assertEqual(solver_guard.negamax(0, max_nodes=1), 0)
        self.assertEqual(solve_sampler.solve(0, max_nodes=1), 0)

    def test_complete_endgame_collection_includes_correct_children_and_labels(self):
        store = {}
        count, _ = solver.solve_and_collect(store, start_state=WIN_CHOICE,
                                            max_nodes=20_000, time_limit=5)
        self.assertEqual(count, len(store))
        self.assertIn(WIN_CHOICE, store)
        for state, label in store.items():
            self.assertEqual(label, reference_analysis(state))
            cells, moves = reference_moves(state)
            me = 1 if sum(cell in (1, 2) for cell in cells) % 2 == 0 else 2
            for move in moves or []:
                self.assertIn(state | (me << (2 * move)), store)

    def test_node_exhaustion_does_not_cache_an_unfinished_value(self):
        engine = exact_solver.ExactSolver()
        with self.assertRaises(exact_solver.SearchLimitExceeded):
            engine.solve(THREE_EMPTY, budget=exact_solver.SearchBudget(1, None))
        self.assertNotIn(THREE_EMPTY, engine.table)
        self.assertEqual(engine.analyze(THREE_EMPTY), reference_analysis(THREE_EMPTY))

    def test_interrupted_labeling_keeps_only_proved_cache_values(self):
        with self.assertRaises(exact_solver.SearchLimitExceeded):
            solve_sampler.label_samples([WIN_CHOICE], max_nodes=3, time_limit=None)
        self.assertTrue(solve_sampler.TT)
        for state, value in solve_sampler.TT.items():
            self.assertEqual(value, reference_value(state))

    def test_time_exhaustion_does_not_return_a_draw(self):
        engine = exact_solver.ExactSolver()
        with self.assertRaises(exact_solver.SearchLimitExceeded):
            engine.solve(THREE_EMPTY, budget=exact_solver.SearchBudget(None, 0))
        self.assertEqual(engine.table, {})

    def test_label_batch_shares_one_node_budget(self):
        with self.assertRaises(exact_solver.SearchLimitExceeded):
            solve_sampler.label_samples([DRAW, WON_WITH_EMPTY], max_nodes=1, time_limit=None)
        self.assertEqual(solve_sampler.TT, {DRAW: 0})

    def test_collection_itself_also_consumes_budget(self):
        with self.assertRaises(exact_solver.SearchLimitExceeded):
            solver.solve_and_collect({}, start_state=DRAW, max_nodes=1, time_limit=None)
        store = {}
        count, _ = solver.solve_and_collect(store, start_state=DRAW,
                                            max_nodes=2, time_limit=None)
        self.assertEqual(count, 1)
        self.assertEqual(store, {DRAW: (0, [])})

    def test_sampling_defaults_to_endgames_and_deduplicates_symmetries(self):
        samples = solve_sampler.collect_samples(random.Random(12345), 20, 15)
        self.assertEqual(len(samples), 15)
        self.assertTrue(all(game.empty_count(state) <= solve_sampler.DEFAULT_MAX_EMPTY
                            for state in samples))
        self.assertNotIn(game.initial_state(), samples)
        self.assertEqual(len({solve_sampler.canon(state) for state in samples}), len(samples))

    def test_zero_requested_samples_returns_no_positions(self):
        self.assertEqual(solve_sampler.collect_samples(random.Random(1), 1, 0), [])

    def test_sampling_counts_actual_empty_cells_on_masked_boards(self):
        early = packed("##### ##### ..... ..... .....")
        late = packed("##### ##### ##### ##### ##..#")
        with patch("solve_sampler.play_random_game", return_value=[early, late]):
            samples = solve_sampler.collect_samples(random.Random(1), 1, 10, max_empty=6)
        self.assertEqual(samples, [late])

    def test_invalid_search_limits_are_rejected(self):
        for max_nodes in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                exact_solver.SearchBudget(max_nodes, None)
        for time_limit in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                exact_solver.SearchBudget(None, time_limit)


if __name__ == "__main__":
    unittest.main()
