"""RGB/U-Net checks using forward and loss evaluation only, without training."""

import math
import unittest

import numpy as np
import torch
from torch import nn

from game import BLACK, WHITE, FORBIDDEN
from unet_codec import (
    EMPTY_RGB, BLACK_RGB, WHITE_RGB, FORBIDDEN_RGB, RED_RGB, GREEN_RGB,
    normalize_grid, encode_rgb, policy_rgb,
)
from unet_models import OpponentUNet5x5, PlayUNet5x5, masked_probabilities, policy_loss


MIXED_STATE = BLACK | (WHITE << 2) | (FORBIDDEN << 4)
WON_STATE = sum(BLACK << (2 * i) for i in range(5))
FORBIDDEN_STATE = (1 << 50) - 1


class RGBCodecTests(unittest.TestCase):
    def test_absolute_colors_and_layout(self):
        image = encode_rgb(MIXED_STATE)
        self.assertEqual(image.shape, (3, 5, 5))
        self.assertEqual(image.dtype, np.float32)
        self.assertTrue(image.flags.c_contiguous)
        for index, color in enumerate((BLACK_RGB, WHITE_RGB, FORBIDDEN_RGB, EMPTY_RGB)):
            np.testing.assert_allclose(image.reshape(3, 25)[:, index],
                                       np.array(color, dtype=np.float32) / 255)

    def test_grid_normalization_copies_and_rejects_invalid_values(self):
        grid = normalize_grid(MIXED_STATE)
        self.assertEqual(grid.dtype, np.uint8)
        self.assertEqual(grid.shape, (5, 5))
        np.testing.assert_array_equal(normalize_grid(grid.reshape(25)), grid)
        copy = normalize_grid(grid)
        copy[0, 0] = 0
        self.assertEqual(grid[0, 0], BLACK)
        for invalid in (-1, 1 << 50, True, [[0]], [0] * 24, [4] * 25, [0.0] * 25):
            with self.subTest(invalid=str(invalid)[:25]):
                with self.assertRaises(ValueError):
                    normalize_grid(invalid)

    def test_continuous_overlay_preserves_stones_and_forbidden(self):
        values = np.ones(25, dtype=np.float32)
        values[3] = 0.123456
        for color, rgb in (("red", RED_RGB), ("green", GREEN_RGB)):
            image = policy_rgb(MIXED_STATE, values, color)
            original = encode_rgb(MIXED_STATE)
            np.testing.assert_array_equal(image.reshape(3, 25)[:, :3], original.reshape(3, 25)[:, :3])
            expected = (np.array(EMPTY_RGB) * (1 - float(values[3])) + np.array(rgb) * float(values[3])) / 255
            np.testing.assert_allclose(image.reshape(3, 25)[:, 3], expected, atol=1e-7)
            np.testing.assert_allclose(image.reshape(3, 25)[:, 4], np.array(rgb) / 255)
            self.assertEqual(image.dtype, np.float32)

    def test_display_quantization_is_opt_in(self):
        values = np.zeros(25, dtype=np.float32)
        values[10] = 0.001
        continuous = policy_rgb(0, values, "red")
        display = policy_rgb(0, values, "red", levels=25)
        self.assertFalse(np.array_equal(continuous, display))
        expected = (np.array(EMPTY_RGB) * 0.96 + np.array(RED_RGB) * 0.04) / 255
        np.testing.assert_allclose(display.reshape(3, 25)[:, 10], expected, atol=1e-7)
        np.testing.assert_array_equal(display.reshape(3, 25)[:, 0], encode_rgb(0).reshape(3, 25)[:, 0])

    def test_overlay_rejects_invalid_probabilities_and_options(self):
        for values in ([0] * 24, [float("nan")] * 25, [float("inf")] * 25, [-0.1] * 25, [1.1] * 25):
            with self.assertRaises(ValueError):
                policy_rgb(0, values, "red")
        with self.assertRaises(ValueError):
            policy_rgb(0, [0] * 25, "blue")
        for levels in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                policy_rgb(0, [0] * 25, "red", levels=levels)


class UNetForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_two_resolution_downsampling_and_exact_skip_upsampling(self):
        net = OpponentUNet5x5(base_channels=4).eval()
        shapes = {}
        hooks = []
        for name in ("encoder1", "encoder2", "bottleneck", "decoder2", "decoder1"):
            def record(module, args, output, key=name):
                shapes[key] = (tuple(args[0].shape), tuple(output.shape))
            hooks.append(getattr(net, name).register_forward_hook(record))
        try:
            with torch.no_grad():
                output = net(torch.from_numpy(encode_rgb(MIXED_STATE))[None], torch.tensor([WHITE]))
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(output.shape, (1, 1, 5, 5))
        self.assertEqual(shapes["encoder1"][1][-2:], (5, 5))
        self.assertEqual(shapes["encoder2"][1][-2:], (3, 3))
        self.assertEqual(shapes["bottleneck"][1][-2:], (2, 2))
        self.assertEqual(shapes["decoder2"][0], (1, 24, 3, 3))
        self.assertEqual(shapes["decoder1"][0], (1, 12, 5, 5))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(any(isinstance(module, nn.GroupNorm) for module in net.modules()))
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in net.modules()))

    def test_models_have_independent_parameters_and_roles(self):
        opponent, play = OpponentUNet5x5(), PlayUNet5x5()
        self.assertEqual(opponent.role, "opponent")
        self.assertEqual(play.role, "play")
        self.assertEqual(opponent.base_channels, 16)
        self.assertEqual(play.base_channels, 16)
        opponent_storage = {parameter.data_ptr() for parameter in opponent.parameters()}
        self.assertTrue(opponent_storage.isdisjoint(parameter.data_ptr() for parameter in play.parameters()))
        image = torch.from_numpy(encode_rgb(MIXED_STATE))[None].repeat(2, 1, 1, 1)
        with torch.no_grad():
            for net in (opponent.eval(), play.eval()):
                logits = net(image, torch.tensor([BLACK, WHITE]))
                self.assertEqual(logits.shape, (2, 1, 5, 5))
                self.assertTrue(torch.isfinite(logits).all())

    def test_side_is_required_and_not_inferred_from_local_stones(self):
        net = OpponentUNet5x5(base_channels=4).eval()
        image = torch.from_numpy(encode_rgb(MIXED_STATE))[None]
        with self.assertRaises(TypeError):
            net(image)
        # The same local board can belong to either actual global turn.
        with torch.no_grad():
            black = net(image, torch.tensor([BLACK]))
            white = net(image, torch.tensor([WHITE]))
        self.assertFalse(torch.allclose(black, white))
        for bad_side in (torch.tensor([0]), torch.tensor([3]), torch.tensor([True]),
                         torch.tensor([1.0]), torch.tensor(1), torch.tensor([1, 2])):
            with self.assertRaises(ValueError):
                net(image, bad_side)

    def test_forward_rejects_invalid_rgb_shapes_ranges_and_nonfinite_values(self):
        net = PlayUNet5x5(base_channels=2)
        invalid = (torch.zeros(1, 1, 5, 5), torch.zeros(1, 3, 6, 5), torch.zeros(0, 3, 5, 5),
                   torch.zeros(1, 3, 5, 5, dtype=torch.long), torch.full((1, 3, 5, 5), 1.01),
                   torch.full((1, 3, 5, 5), float("nan")))
        for image in invalid:
            with self.assertRaises(ValueError):
                net(image, torch.tensor([BLACK]))


class MaskedPolicyTests(unittest.TestCase):
    def test_stones_and_forbidden_are_exactly_zero_despite_huge_logits(self):
        logits = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
        logits.reshape(-1)[:3] = 10000
        probabilities = masked_probabilities(logits, [MIXED_STATE])
        self.assertTrue(torch.all(probabilities.reshape(-1)[:3] == 0))
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertAlmostEqual(probabilities.sum().item(), 1.0, places=6)
        self.assertEqual(probabilities.argmax().item(), 24)

    def test_terminal_and_no_empty_boards_have_zero_policy(self):
        logits = torch.zeros(3, 1, 5, 5)
        probabilities = masked_probabilities(logits, [0, WON_STATE, FORBIDDEN_STATE])
        self.assertAlmostEqual(probabilities[0].sum().item(), 1.0, places=6)
        self.assertTrue(torch.all(probabilities[1:] == 0))
        self.assertTrue(torch.isfinite(probabilities).all())

    def test_sole_legal_cell_has_probability_one(self):
        state = FORBIDDEN_STATE & ~(3 << 24)
        probabilities = masked_probabilities(torch.randn(1, 1, 5, 5), torch.tensor([state]))
        self.assertEqual(probabilities.reshape(-1)[12].item(), 1)
        self.assertEqual(torch.count_nonzero(probabilities).item(), 1)

    def test_masked_loss_matches_legal_uniform_cross_entropy_without_training(self):
        logits = torch.zeros(1, 1, 5, 5, requires_grad=True)
        targets = torch.zeros(1, 1, 5, 5)
        targets.reshape(-1)[3] = 0.25
        targets.reshape(-1)[24] = 0.75
        loss = policy_loss(logits, targets, [MIXED_STATE])
        self.assertAlmostEqual(loss.item(), math.log(22), places=6)
        self.assertTrue(loss.requires_grad)
        self.assertIsNone(logits.grad)

    def test_invalid_terminal_and_empty_targets_are_rejected(self):
        logits = torch.zeros(1, 1, 5, 5)
        empty = torch.zeros(1, 25)
        occupied = empty.clone()
        occupied[0, 0] = 1
        forbidden = empty.clone()
        forbidden[0, 2] = 1
        for targets in (empty, occupied, forbidden, torch.full((1, 25), float("nan")),
                        torch.full((1, 25), -0.1)):
            with self.assertRaises(ValueError):
                policy_loss(logits, targets, [MIXED_STATE])
        for state in (WON_STATE, FORBIDDEN_STATE):
            with self.assertRaises(ValueError):
                policy_loss(logits, torch.ones(1, 25) / 25, [state])

    def test_bad_logits_and_mismatched_states_are_rejected(self):
        for logits in (torch.zeros(1, 25), torch.zeros(0, 1, 5, 5),
                       torch.full((1, 1, 5, 5), float("inf"))):
            with self.assertRaises(ValueError):
                masked_probabilities(logits, [0])
        for states in ([], [0, 0], [True], [1 << 50], [0.0]):
            with self.assertRaises(ValueError):
                masked_probabilities(torch.zeros(1, 1, 5, 5), states)


if __name__ == "__main__":
    unittest.main()
