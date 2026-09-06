"""Input-mode compatibility across whole-board encoding, inference and training."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from global_inference import analyze_global, evaluate_global_value, load_global_model
from global_model import GlobalBoardNet, encode_global
from train_global import augment_record, batches, evaluate
from unet_codec import BLACK_RGB, WHITE_RGB, FORBIDDEN_RGB


class CaptureGlobal(nn.Module):
    def __init__(self, input_mode):
        super().__init__()
        self.input_mode = input_mode
        self.last_input = None

    def forward(self, inputs):
        self.last_input = inputs.detach().clone()
        return inputs[:, 4:5] * 20, torch.zeros(len(inputs), dtype=inputs.dtype)


def fixture(side=2):
    board = np.zeros((6, 7), dtype=np.uint8)
    board[0, 1], board[2, 4], board[5, 6] = 1, 2, 3
    opponent, local = np.zeros(board.shape, dtype=np.float32), np.zeros(board.shape, dtype=np.float32)
    opponent[3, 3], local[4, 5] = 1, 1
    return dict(board=board, side=side, opponent_policy=opponent, local_policy=local,
                target_policy=local.copy(), value=0, value_valid=True, policy_source="searched_move")


class GlobalInputModeTests(unittest.TestCase):
    def test_legacy_encoder_remains_absolute_and_default(self):
        row = fixture()
        original = encode_global(row["board"], row["side"], row["opponent_policy"], row["local_policy"])
        explicit = encode_global(row["board"], row["side"], row["opponent_policy"], row["local_policy"],
                                 input_mode="absolute_rgb")
        np.testing.assert_array_equal(original, explicit)
        np.testing.assert_allclose(original[:3, 0, 1], np.array(BLACK_RGB)/255, atol=1e-7)
        self.assertTrue(np.all(original[6] == 1))
        self.assertEqual(GlobalBoardNet().input_mode, "absolute_rgb")

    def test_relative_rgb_is_exactly_joint_color_actor_invariant(self):
        row = fixture()
        before = row["board"].copy()
        relative = encode_global(row["board"], 2, row["opponent_policy"], row["local_policy"],
                                 input_mode="relative_rgb")
        swapped_board = np.array([0, 2, 1, 3], dtype=np.uint8)[row["board"]]
        swapped = encode_global(swapped_board, 1, row["opponent_policy"], row["local_policy"],
                                input_mode="relative_rgb")
        np.testing.assert_array_equal(relative, swapped)
        np.testing.assert_array_equal(row["board"], before)
        for cell, color in (((0,1), WHITE_RGB), ((2,4), BLACK_RGB), ((5,6), FORBIDDEN_RGB)):
            np.testing.assert_allclose(relative[:3, cell[0], cell[1]], np.array(color)/255, atol=1e-7)
        np.testing.assert_array_equal(relative[3], row["opponent_policy"])
        np.testing.assert_array_equal(relative[4], row["local_policy"])
        np.testing.assert_array_equal(relative[5], row["board"] == 0)
        self.assertTrue(np.all(relative[6] == -1))

    def test_checkpoint_mode_is_optional_metadata_and_does_not_change_weights(self):
        original = GlobalBoardNet(base_channels=4, token_dim=8, attention_heads=2)
        payload = dict(format="gomoku_global_v1", role="global", trained=False,
                       base_channels=4, token_dim=8, attention_heads=2, state_dict=original.state_dict())
        for mode in (None, "absolute_rgb", "relative_rgb"):
            candidate = payload if mode is None else dict(payload, input_mode=mode)
            with patch("global_inference.torch.load", return_value=candidate):
                loaded, metadata = load_global_model("in-memory.pt")
            self.assertEqual(loaded.input_mode, mode or "absolute_rgb")
            self.assertFalse(loaded.training)
            for name, tensor in original.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]))
            if mode is not None:
                self.assertEqual(metadata["input_mode"], mode)
        with patch("global_inference.torch.load", return_value=dict(payload, input_mode="role_planes")):
            with self.assertRaisesRegex(ValueError, "input_mode"):
                load_global_model("in-memory.pt")

    def test_invalid_input_modes_are_rejected(self):
        row = fixture()
        for mode in (None, False, 1, "relative", "role_planes"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "input_mode"):
                    GlobalBoardNet(input_mode=mode)
                with self.assertRaisesRegex(ValueError, "input_mode"):
                    encode_global(row["board"], 2, row["opponent_policy"], row["local_policy"],
                                  input_mode=mode)

    def test_root_and_leaf_inference_follow_model_mode_without_changing_board(self):
        row = fixture()
        before = row["board"].copy()
        model = CaptureGlobal("relative_rgb")
        local = dict(my_side=2, opponent_policy=row["opponent_policy"], raw_play_policy=row["local_policy"])
        analyze_global(row["board"], 2, local, model)
        expected = encode_global(row["board"], 2, row["opponent_policy"], row["local_policy"],
                                 input_mode="relative_rgb")
        np.testing.assert_array_equal(model.last_input[0].numpy(), expected)
        evaluate_global_value(row["board"], 2, model)
        self.assertTrue(torch.all(model.last_input[:, 3:5] == 0))
        self.assertTrue(torch.all(model.last_input[:, 6] == -1))
        np.testing.assert_array_equal(row["board"], before)
        swapped = np.array([0, 2, 1, 3], dtype=np.uint8)[row["board"]]
        previous = model.last_input.clone()
        evaluate_global_value(swapped, 1, model)
        self.assertTrue(torch.equal(previous, model.last_input))

    def test_training_transform_and_validation_forward_relative_mode(self):
        row = fixture()
        for turns in range(4):
            for reflect in (False, True):
                original = augment_record(row, turns, reflect, False, input_mode="relative_rgb")
                swapped = augment_record(row, turns, reflect, True, input_mode="relative_rgb")
                np.testing.assert_array_equal(original[0], swapped[0])
                np.testing.assert_array_equal(original[2], swapped[2])
                self.assertEqual(original[3:], swapped[3:])
                self.assertFalse(np.array_equal(original[1], swapped[1]))
                np.testing.assert_array_equal(original[1] == 0, swapped[1] == 0)
        (inputs, _, _, _, _), _ = next(batches([row], 1, input_mode="relative_rgb"))
        self.assertTrue(torch.all(inputs[:, 6] == -1))
        model = CaptureGlobal("relative_rgb")
        metrics = evaluate(model, [row], 1)
        self.assertEqual(metrics["policy"]["top1"], 1)
        self.assertTrue(torch.equal(model.last_input, inputs))


if __name__ == "__main__":
    unittest.main()
