import unittest
from unittest.mock import patch

import numpy as np

import evaluate_global as evaluate
from board_rules import board_winner
from board_threats import analyze_threats


def fake_local(board, side, opponent, play, tactical=True):
    legal = board == 0
    terminal = bool(board_winner(board) or not legal.any())
    probability = legal.astype(float)
    if terminal:
        probability.fill(0)
    else:
        probability /= probability.sum()
    return dict(opponent_policy=probability.copy(), play_policy=probability.copy(),
                raw_play_policy=probability.copy(), coverage=np.zeros_like(board) if terminal else legal.astype(int),
                window_count=0 if terminal else 1,
                threats={"self": analyze_threats(board, side), "opponent": analyze_threats(board, 3-side)})


def fake_global(board, side, analysis, model):
    terminal = bool(board_winner(board) or not np.any(board == 0))
    return dict(global_policy=analysis["raw_play_policy"].copy(),
                combined_policy=analysis["raw_play_policy"].copy(),
                value=None if terminal else 0.25,
                value_source="terminal_no_inference" if terminal else "network_estimate")


class GlobalEvaluationTests(unittest.TestCase):
    def test_openings_are_fixed_paired_deduplicated_and_have_five_stones(self):
        prefixes = evaluate.generate_paired_openings(16, 16, 2, 17)
        self.assertEqual(prefixes, evaluate.generate_paired_openings(16, 16, 2, 17))
        keys = []
        for prefix in prefixes:
            board, side, transcript = evaluate._start_board(16, 16, prefix)
            self.assertEqual(len(prefix), 4)
            self.assertEqual(board[8, 8], 1)
            self.assertEqual((len(transcript), np.count_nonzero(board), side), (5, 5, 2))
            keys.append(evaluate.canonical_opening(board))
        self.assertEqual(len(set(keys)), 2)

    def test_legal_contract_and_probability_summary(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        board[0, 0] = 3
        analysis = fake_local(board, 1, None, None)
        self.assertTrue(evaluate.policy_contract(board, analysis)["passed"])
        summary = evaluate.policy_summary(analysis["raw_play_policy"], board, (1, 1))
        self.assertEqual(summary["chosen_rank"], 1)
        self.assertEqual(summary["chosen_tie_count"], 41)
        self.assertAlmostEqual(summary["chosen_probability"], 1/41)
        analysis["raw_play_policy"][0, 0] = 0.1
        self.assertFalse(evaluate.policy_contract(board, analysis)["passed"])

    @patch("evaluate_global.analyze_global", side_effect=fake_global)
    @patch("evaluate_global.analyze_board", side_effect=fake_local)
    def test_paired_one_ply_caps_do_not_turn_estimates_or_certificates_into_results(self, *_):
        calls = []
        def chosen(board, side, priors, **kwargs):
            calls.append((board.copy(), side, priors.copy(), kwargs))
            return dict(move=tuple(map(int, np.argwhere(board == 0)[0])), reason="test certificate",
                        nodes=0, completed_depth=0, proven_value=1, budget_exhausted=False,
                        elapsed_seconds=0, value_evaluations=0)
        with patch("evaluate_global.select_move", side_effect=chosen):
            result = evaluate.paired_games(None, None, None, games=2, max_plies=1, search_seconds=0, max_nodes=0)
        self.assertEqual(result["outcomes_for_global"], {"win": 0, "loss": 0, "draw": 0, "truncated": 2})
        self.assertEqual(result["terminal_games"], 0)
        first, second = result["records"]
        self.assertEqual((first["global_side"], second["global_side"]), (1, 2))
        self.assertEqual(first["initial_board"], second["initial_board"])
        self.assertEqual(first["opening_prefix"], second["opening_prefix"])
        self.assertEqual(calls[0][1], calls[1][1])
        for call in calls:
            self.assertEqual(call[3]["depth"], 5)
            self.assertEqual(call[3]["candidate_width"], 12)
            self.assertEqual(call[3]["max_nodes"], 0)
            self.assertEqual(call[3]["time_limit"], 0)
        self.assertNotIn("value_evaluator", calls[0][3])
        self.assertTrue(callable(calls[1][3]["value_evaluator"]))
        self.assertEqual(first["transcript"][-1]["prior_used"], "raw_play_policy")
        self.assertEqual(second["transcript"][-1]["prior_used"], "combined_policy")
        self.assertIn("threats_before", second["transcript"][-1])
        self.assertIn("threats_after", second["transcript"][-1])
        self.assertIsNotNone(second["transcript"][-1]["global_value"])

    @patch("evaluate_global.analyze_global", side_effect=fake_global)
    @patch("evaluate_global.analyze_board", side_effect=fake_local)
    def test_no_empty_cells_is_a_real_draw(self, *_):
        board = np.full((6, 6), 3, dtype=np.uint8)
        board[0, 0] = 0
        with patch("evaluate_global._start_board", return_value=(board, 1, [])):
            result = evaluate.play_one_game(None, None, None, 6, 6, (), 1,
                                            search_seconds=0, max_nodes=0, max_plies=1)
        self.assertTrue(result["terminal"])
        self.assertFalse(result["truncated"])
        self.assertEqual(result["outcome_for_global"], "draw")
        self.assertEqual(result["stop_reason"], "board_full")

    @patch("evaluate_global.analyze_global", side_effect=fake_global)
    @patch("evaluate_global.analyze_board", side_effect=fake_local)
    def test_known_probes_use_real_search_without_neural_weights_or_game_rollouts(self, *_):
        result = evaluate.compliance_probes(None, None, None, search_seconds=0, max_nodes=0)
        self.assertTrue(result["passed"], [r["name"] for r in result["records"] if not r["passed"]])
        records = [r for r in result["records"] if r["name"] == "cross_window_independent_enemy_kills"]
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record["search"]["proven_value"], -1)
            self.assertEqual(record["global_value"], 0.25 if record["mode"] == "global" else None)

    def test_invalid_pair_and_budget_configuration_rejected_before_models(self):
        for kwargs in ({"games": 3}, {"games": -2}, {"max_plies": 0},
                       {"rows": 5}, {"search_seconds": float("nan")}, {"max_nodes": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate.paired_games(None, None, None, **kwargs)


if __name__ == "__main__":
    unittest.main()
