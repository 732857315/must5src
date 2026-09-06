import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from global_inference import (CHECKPOINT_FORMAT, GLOBAL_WEIGHT, analyze_global,
                              evaluate_global_value, load_global_model)
from global_model import GlobalBoardNet


class FixedGlobal(nn.Module):
    def __init__(self, value=0.25, bad_logits=False, bad_shape=False):
        super().__init__()
        self.value = value
        self.bad_logits = bad_logits
        self.bad_shape = bad_shape
        self.last_input = None

    def forward(self, image):
        self.last_input = image.detach().clone()
        logits = torch.zeros_like(image[:, :1])
        logits[:, :, 0, 0] = 100.0  # Illegal stone: masking must discard it.
        logits[:, :, 1, 1] = 3.0
        if self.bad_logits:
            logits[:, :, 2, 2] = float("nan")
        if self.bad_shape:
            logits = logits[:, :, :-1]
        return logits, torch.tensor([self.value], dtype=image.dtype, device=image.device)


class GlobalInferenceTests(unittest.TestCase):
    def setUp(self):
        self.board = np.zeros((6, 9), dtype=np.uint8)
        self.board[0, 0] = 1
        self.board[3, 4] = 3
        policy = (self.board == 0).astype(np.float64)
        policy /= policy.sum()
        self.local = {"my_side": 2, "opponent_policy": policy.copy(),
                      "raw_play_policy": policy.copy()}

    def test_fusion_is_continuous_normalized_and_legal(self):
        model = FixedGlobal()
        result = analyze_global(self.board, 2, self.local, model)
        self.assertEqual(tuple(model.last_input.shape), (1, 9, 6, 9))
        self.assertEqual(result["value"], 0.25)
        self.assertEqual(result["value_source"], "network_estimate")
        for name in ("global_policy", "combined_policy"):
            self.assertAlmostEqual(float(result[name].sum()), 1.0, places=7)
            self.assertTrue(np.all(result[name][self.board != 0] == 0))
            self.assertEqual(np.unravel_index(result[name].argmax(), self.board.shape), (1, 1))
        np.testing.assert_allclose(result["combined_policy"],
                                   GLOBAL_WEIGHT * result["global_policy"] +
                                   (1 - GLOBAL_WEIGHT) * self.local["raw_play_policy"])
        np.testing.assert_allclose(model.last_input[0, 3].numpy(), self.local["opponent_policy"])
        np.testing.assert_allclose(model.last_input[0, 4].numpy(), self.local["raw_play_policy"])

    def test_value_evaluator_uses_zero_auxiliary_and_explicit_actor(self):
        model = FixedGlobal(value=-0.4)
        self.assertAlmostEqual(evaluate_global_value(self.board, 2, model), -0.4, places=6)
        self.assertTrue(torch.all(model.last_input[0, 3:5] == 0))
        self.assertTrue(torch.all(model.last_input[0, 6] == 1))
        evaluate_global_value(self.board, 1, model)
        self.assertTrue(torch.all(model.last_input[0, 6] == -1))

    def test_terminal_never_calls_network_and_has_no_policy(self):
        self.board[1, :5] = 1
        self.local["opponent_policy"].fill(0)
        self.local["raw_play_policy"].fill(0)
        model = FixedGlobal()
        result = analyze_global(self.board, 2, self.local, model)
        self.assertIsNone(model.last_input)
        self.assertIsNone(result["value"])
        self.assertEqual(result["combined_policy"].sum(), 0)
        self.assertEqual(result["global_policy"].sum(), 0)
        self.assertEqual(evaluate_global_value(self.board, 2, model), -1)
        self.assertIsNone(model.last_input)

    def test_rejects_wrong_actor_or_bad_input_probabilities(self):
        with self.assertRaises(ValueError):
            analyze_global(self.board, 1, self.local, FixedGlobal())
        for side in (True, 0, 3):
            with self.subTest(side=side), self.assertRaises(ValueError):
                evaluate_global_value(self.board, side, FixedGlobal())
        for location, value in (((0, 0), 0.1), ((2, 2), float("nan")), ((2, 2), -0.1)):
            broken = {**self.local, "raw_play_policy": self.local["raw_play_policy"].copy()}
            broken["raw_play_policy"][location] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                analyze_global(self.board, 2, broken, FixedGlobal())

    def test_rejects_invalid_network_outputs(self):
        for model in (FixedGlobal(value=float("nan")), FixedGlobal(value=2),
                      FixedGlobal(bad_logits=True), FixedGlobal(bad_shape=True)):
            with self.subTest(model=model.__dict__), self.assertRaises(ValueError):
                analyze_global(self.board, 2, self.local, model)

    @staticmethod
    def payload():
        model = GlobalBoardNet(base_channels=8, token_dim=16, attention_heads=4)
        return dict(format=CHECKPOINT_FORMAT, role="global", base_channels=8, token_dim=16,
                    attention_heads=4, trained=False, state_dict=model.state_dict())

    def test_checkpoint_roundtrip_is_weights_only_and_separate_format(self):
        payload = self.payload()
        stream = io.BytesIO()
        torch.save(payload, stream)
        real_load = torch.load

        def read(_path, **kwargs):
            self.assertEqual(kwargs, {"map_location": "cpu", "weights_only": True})
            stream.seek(0)
            return real_load(stream, **kwargs)

        with patch("global_inference.torch.load", side_effect=read) as mocked:
            model, metadata = load_global_model("in-memory-global.pt")
        mocked.assert_called_once_with(Path("in-memory-global.pt"), map_location="cpu", weights_only=True)
        self.assertFalse(model.training)
        self.assertFalse(metadata["trained"])
        self.assertNotIn("state_dict", metadata)
        for key, tensor in payload["state_dict"].items():
            self.assertTrue(torch.equal(tensor, model.state_dict()[key]))

    def test_checkpoint_rejects_format_configuration_and_nonfinite_weights(self):
        payload = self.payload()
        for change in ({"format": "gomoku_rgb_unet_v1"}, {"role": "opponent"},
                       {"token_dim": 15}, {"attention_heads": True}, {"trained": "yes"}):
            with self.subTest(change=change), patch("global_inference.torch.load", return_value={**payload, **change}):
                with self.assertRaises(ValueError):
                    load_global_model("unused.pt")
        bad = {key: value.clone() for key, value in payload["state_dict"].items()}
        next(iter(bad.values())).reshape(-1)[0] = float("nan")
        with patch("global_inference.torch.load", return_value={**payload, "state_dict": bad}):
            with self.assertRaises(ValueError):
                load_global_model("unused.pt")

    def test_composed_analyzer_keeps_forced_tactic_and_uses_global_otherwise(self):
        from play_unet import build_analyzer
        forced_policy = np.zeros_like(self.board, dtype=float)
        forced_policy[0, 1] = 1.0
        local = {**self.local, "play_policy": forced_policy,
                 "terminal": False, "tactical_applied": True, "move": (0, 1), "reason": "forced block"}
        with patch("unet_board.analyze_board", return_value=local):
            analyzer = build_analyzer(FixedGlobal())
            forced = analyzer(self.board, 2, None, None)
            self.assertEqual(forced["move"], (0, 1))
            self.assertEqual(forced["reason"], "forced block")
            np.testing.assert_array_equal(forced["play_policy"], forced_policy)
            self.assertNotEqual(float(forced["combined_policy"][0, 1]), 1.0)
        quiet = {**local, "tactical_applied": False}
        with patch("unet_board.analyze_board", return_value=quiet):
            analyzer = build_analyzer(FixedGlobal())
            result = analyzer(self.board, 2, None, None)
            self.assertEqual(result["move"], (1, 1))
            np.testing.assert_array_equal(result["play_policy"], result["combined_policy"])

    def test_actual_search_uses_global_value_and_keeps_board_unchanged(self):
        from play_unet import build_move_selector
        model = FixedGlobal()
        analysis = {**self.local, **analyze_global(self.board, 2, self.local, model)}
        selector = build_move_selector(model, search_seconds=0.5, search_nodes=60, search_depth=1, search_width=4)
        original = self.board.copy()
        result = selector(self.board, 2, analysis)
        self.assertEqual(self.board[result["move"]], 0)
        np.testing.assert_array_equal(self.board, original)
        self.assertGreater(result["value_evaluations"], 0)
        self.assertIsNone(result["proven_value"])
        self.assertTrue(torch.all(model.last_input[0, 3:5] == 0))
        self.assertIn("大局估计", result["reason"])

    def test_move_reason_uses_observed_patterns_and_preserves_tactical_reason(self):
        from play_unet import describe_ai_move
        board = np.zeros((6, 9), dtype=np.uint8)
        board[2, 1:3] = 1
        self.assertIn("增加活三", describe_ai_move(board, 1, (2, 3), {"reason": "search"}))
        board[2, 3] = 1
        self.assertIn("升四路径", describe_ai_move(board, 2, (2, 4), {"reason": "search"}))
        self.assertIn("活四", describe_ai_move(board, 1, (2, 4), {"reason": "search"}))
        board[2, 0] = 1
        reason = describe_ai_move(board, 2, (2, 4), {"reason": "唯一立即防点"})
        self.assertEqual(reason, "唯一立即防点")
        reason = describe_ai_move(board, 1, (2, 4), {"reason": "全盘立即成五", "proven_value": 1})
        self.assertEqual(reason, "全盘立即成五")

    def test_global_session_uses_ai_actor_then_refreshes_current_human_policies(self):
        from play_unet import GameSession
        records = []
        priors_seen = []
        class ActorModel(FixedGlobal):
            def forward(self, image):
                logits, _ = super().forward(image)
                return logits, image[:, 6].mean(dim=(1, 2)) * 0.5
        model = ActorModel()
        def analyzer(board, side, opponent, play, tactical=True):
            probability = (board == 0).astype(float)
            probability /= probability.sum()
            result = dict(side=side, opponent_policy=probability, play_policy=probability,
                          raw_play_policy=probability, coverage=(board == 0).astype(int))
            result.update(analyze_global(board, side, result, model))
            result["move"] = tuple(map(int, np.unravel_index(result["combined_policy"].argmax(), board.shape)))
            records.append((board.copy(), side, result["combined_policy"].copy()))
            return result
        def selector(board, side, analysis):
            self.assertEqual((side, analysis["side"]), (1, 1))
            self.assertEqual(analysis["value"], -0.5)
            np.testing.assert_array_equal(analysis["combined_policy"], records[-1][2])
            priors_seen.append(np.array(analysis["combined_policy"]))
            return {"move": (0, 0), "reason": "test chosen move"}
        game = GameSession(None, None, analyzer=analyzer, move_selector=selector,
                           strategy_source="in-memory", strategy_trained=False)
        result = game.move({"row": 1, "col": 1, "revision": game.revision})
        self.assertEqual((result["analysis"]["side"], result["analysis"]["value"]), (2, 0.5))
        self.assertEqual(result["board"][0][0], 1)
        self.assertEqual(result["analysis"]["combined_policy"][0][0], 0)
        self.assertGreater(priors_seen[0][0, 0], 0.69)
        np.testing.assert_array_equal(records[-1][0], result["board"])
        np.testing.assert_array_equal(records[-1][2], result["analysis"]["combined_policy"])

    def test_server_carries_all_global_fields_and_model_status(self):
        from play_unet import GameSession
        def analyzer(board, side, opponent, play, tactical=True):
            probabilities = (board == 0).astype(float)
            probabilities /= probabilities.sum()
            local = dict(opponent_policy=probabilities, play_policy=probabilities,
                         raw_play_policy=probabilities, move=(0, 0), reason="test",
                         coverage=(board == 0).astype(int))
            local.update(analyze_global(board, side, local, FixedGlobal(value=0.5)))
            return local
        game = GameSession(None, None, analyzer=analyzer, strategy_source="test.pt", strategy_trained=False)
        state = game.snapshot()
        self.assertEqual(state["analysis"]["value"], 0.5)
        self.assertTrue(state["models"]["global"]["loaded"])
        self.assertFalse(state["models"]["global"]["trained"])
        self.assertIn("global_policy", state["analysis"])
        self.assertIn("combined_policy", state["analysis"])


if __name__ == "__main__":
    unittest.main()
