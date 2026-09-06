"""Role-aware checkpoint and prediction contracts, without training or model files."""

import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from game import BLACK, WHITE, FORBIDDEN, apply_move, legal_moves
from unet_codec import encode_rgb, policy_rgb
from unet_models import OpponentUNet5x5, PlayUNet5x5
import unet_pipeline as pipeline


STATE = (BLACK << 24) | (WHITE << 2) | (FORBIDDEN << 8)


class RecordingModel:
    def __init__(self):
        self.calls = []

    def __call__(self, rgb, side):
        self.calls.append((rgb.clone(), side.clone()))
        return torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5) / 10


def payload_for(model):
    return {"format": pipeline.CHECKPOINT_FORMAT, "role": model.role,
            "base_channels": model.base_channels,
            "state_dict": {key: value.detach().clone() for key, value in model.state_dict().items()},
            "trained": False, "test_fixture": True}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_both_role_checkpoints_roundtrip_without_writing_weight_files(self):
        for model_class in (OpponentUNet5x5, PlayUNet5x5):
            model = model_class(base_channels=2).eval()
            buffer = io.BytesIO()
            torch.save(payload_for(model), buffer)
            buffer.seek(0)
            payload = torch.load(buffer, map_location="cpu", weights_only=True)
            with patch.object(pipeline.torch, "load", return_value=payload) as loader:
                restored, metadata = pipeline.load_model("unused-test-path.pt", model.role)
            self.assertTrue(loader.call_args.kwargs["weights_only"])
            self.assertEqual(metadata["format"], pipeline.CHECKPOINT_FORMAT)
            self.assertEqual(metadata["role"], model.role)
            self.assertNotIn("state_dict", metadata)
            self.assertFalse(restored.training)
            image = torch.from_numpy(encode_rgb(STATE))[None]
            with torch.no_grad():
                expected = model(image, torch.tensor([WHITE]))
                actual = restored(image, torch.tensor([WHITE]))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_old_format_wrong_role_and_incompatible_weights_are_rejected(self):
        payload = payload_for(OpponentUNet5x5(base_channels=2))
        for invalid in ({"state_dict": {}}, {**payload, "format": "old_format"},
                        {**payload, "role": "play"}):
            with patch.object(pipeline.torch, "load", return_value=invalid):
                with self.assertRaises(ValueError):
                    pipeline.load_model("unused.pt", "opponent")
        with patch.object(pipeline.torch, "load", return_value={**payload, "state_dict": {}}):
            with self.assertRaises(RuntimeError):
                pipeline.load_model("unused.pt", "opponent")
        with self.assertRaises(ValueError):
            pipeline.load_model("unused.pt", "unknown")

    def test_nonfinite_checkpoint_parameters_are_rejected(self):
        payload = payload_for(PlayUNet5x5(base_channels=2))
        next(iter(payload["state_dict"].values())).flatten()[0] = float("nan")
        with patch.object(pipeline.torch, "load", return_value=payload):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                pipeline.load_model("unused.pt", "play")

    def test_play_uses_continuous_current_position_red_and_explicit_side(self):
        for my_side in (BLACK, WHITE):
            opponent, play = RecordingModel(), RecordingModel()
            result = pipeline.analyze_position(STATE, my_side, opponent, play)
            np.testing.assert_array_equal(opponent.calls[0][0][0].numpy(), encode_rgb(STATE))
            self.assertEqual(opponent.calls[0][1].item(), 3 - my_side)
            self.assertEqual(play.calls[0][1].item(), my_side)
            expected_red = policy_rgb(STATE, result["opponent_policy"], "red", levels=None)
            np.testing.assert_array_equal(play.calls[0][0][0].numpy(), expected_red)
            self.assertIn(result["move"], legal_moves(STATE))
            self.assertEqual(result["state"], STATE)
            self.assertEqual(result["opponent_assumption"], "opponent_to_move_on_current_position")
            for name in ("opponent_policy", "play_policy"):
                probabilities = result[name].reshape(-1)
                self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
                self.assertTrue(np.all(probabilities[[1, 4, 12]] == 0))

    def test_true_reply_applies_own_move_before_predicting_opponent(self):
        for my_side in (BLACK, WHITE):
            opponent = RecordingModel()
            next_state, probabilities = pipeline.predict_reply(STATE, my_side, 8, opponent)
            self.assertEqual(next_state, apply_move(STATE, 8, my_side))
            np.testing.assert_array_equal(opponent.calls[0][0][0].numpy(), encode_rgb(next_state))
            self.assertEqual(opponent.calls[0][1].item(), 3 - my_side)
            self.assertEqual(probabilities.reshape(-1)[8], 0)
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)

    def test_illegal_reply_moves_are_rejected_before_network_call(self):
        opponent = RecordingModel()
        for move in (1, 4, 12, -1, 25):
            with self.assertRaises(ValueError):
                pipeline.predict_reply(STATE, BLACK, move, opponent)
        self.assertEqual(opponent.calls, [])

    def test_terminal_positions_and_winning_own_move_have_no_recommendation(self):
        won = sum(BLACK << (2 * i) for i in range(5))
        result = pipeline.analyze_position(won, WHITE, RecordingModel(), RecordingModel())
        self.assertIsNone(result["move"])
        self.assertTrue(np.all(result["play_policy"] == 0))
        self.assertTrue(np.all(result["opponent_policy"] == 0))
        before = sum(BLACK << (2 * i) for i in range(4))
        next_state, reply = pipeline.predict_reply(before, BLACK, 4, RecordingModel())
        self.assertEqual(next_state, won)
        self.assertTrue(np.all(reply == 0))


if __name__ == "__main__":
    unittest.main()
