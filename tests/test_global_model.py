"""Whole-board encoding, bounded attention, legal masking, and supervised loss."""

import math
import unittest

import numpy as np
import torch
from torch import nn

from global_model import GlobalBoardNet, encode_global, masked_global_policy, global_loss
from unet_codec import BLACK_RGB, WHITE_RGB, FORBIDDEN_RGB


class GlobalEncodingTests(unittest.TestCase):
    def test_channels_preserve_absolute_rgb_continuous_maps_and_explicit_side(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        board[0, :3] = [1, 2, 3]
        red = np.zeros(board.shape, dtype=np.float32)
        green = red.copy()
        red[1, 2] = 0.123456
        green[4, 5] = 0.654321
        encoded = encode_global(board, 2, red, green)
        self.assertEqual(encoded.shape, (9, 6, 7))
        self.assertEqual(encoded.dtype, np.float32)
        self.assertTrue(encoded.flags.c_contiguous)
        for column, color in enumerate((BLACK_RGB, WHITE_RGB, FORBIDDEN_RGB)):
            np.testing.assert_allclose(encoded[:3, 0, column], np.array(color) / 255, atol=1e-7)
        np.testing.assert_array_equal(encoded[3], red)
        np.testing.assert_array_equal(encoded[4], green)
        np.testing.assert_array_equal(encoded[5], board == 0)
        self.assertTrue(np.all(encoded[6] == 1))
        self.assertTrue(np.all(encode_global(board, 1, red, green)[6] == -1))

    def test_coordinates_are_rebuilt_after_rectangular_rotation(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        maps = np.zeros_like(board, dtype=np.float32)
        original = encode_global(board, 1, maps, maps)
        rotated = encode_global(np.rot90(board), 1, np.rot90(maps), np.rot90(maps))
        self.assertEqual(rotated.shape, (9, 7, 6))
        np.testing.assert_array_equal(rotated[7, :, 0], np.linspace(-1, 1, 7, dtype=np.float32))
        np.testing.assert_array_equal(rotated[8, 0], np.linspace(-1, 1, 6, dtype=np.float32))
        self.assertFalse(np.array_equal(rotated[7:], np.rot90(original[7:], axes=(-2, -1))))
        tiny = encode_global([[0]], 1, [[1]], [[1]])
        self.assertTrue(np.all(tiny[7:] == 0))

    def test_illegal_maps_side_and_terminal_moves_are_rejected(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        board[0, 0] = 3
        zero = np.zeros(board.shape)
        bad = zero.copy()
        bad[0, 0] = 0.1
        with self.assertRaises(ValueError):
            encode_global(board, 1, bad, zero)
        for side in (0, 3, True, 1.0):
            with self.assertRaises(ValueError):
                encode_global(board, side, zero, zero)
        for bad in (np.zeros((7, 6)), np.full(board.shape, np.nan), np.full(board.shape, -0.1)):
            with self.assertRaises(ValueError):
                encode_global(board, 1, bad, zero)
        board[1, :5] = 1
        self.assertEqual(encode_global(board, 2, zero, zero)[5].sum(), 0)
        bad = zero.copy()
        bad[5, 5] = 1
        with self.assertRaises(ValueError):
            encode_global(board, 2, bad, zero)


class GlobalNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_arbitrary_shapes_use_no_more_than_64_attention_tokens(self):
        model = GlobalBoardNet(base_channels=4, token_dim=8, attention_heads=2).eval()
        token_shapes = []
        hook = model.attention.register_forward_pre_hook(lambda module, args: token_shapes.append(tuple(args[0].shape)))
        try:
            for rows, cols in ((1, 1), (6, 7), (16, 16), (20, 24), (7, 35)):
                board = np.zeros((rows, cols), dtype=np.uint8)
                zero = np.zeros(board.shape)
                inputs = torch.from_numpy(encode_global(board, 1, zero, zero))[None]
                with torch.no_grad():
                    logits, value = model(inputs)
                self.assertEqual(logits.shape, (1, 1, rows, cols))
                self.assertEqual(value.shape, (1,))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(value).all())
                self.assertTrue(torch.all(value.abs() <= 1))
                self.assertEqual(token_shapes[-1], (1, min(8, rows) * min(8, cols), 8))
        finally:
            hook.remove()
        self.assertFalse(any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules()))
        self.assertEqual([layer.body[0].dilation for layer in model.spatial], [(1, 1), (2, 2), (4, 4)])

    def test_default_structure_independent_weights_and_same_shape_batch(self):
        first, second = GlobalBoardNet(), GlobalBoardNet()
        self.assertEqual((first.role, first.base_channels, first.token_dim, first.attention_heads), ("global", 16, 32, 4))
        self.assertLess(sum(parameter.numel() for parameter in first.parameters()), 100_000)
        storage = {parameter.data_ptr() for parameter in first.parameters()}
        self.assertTrue(storage.isdisjoint(parameter.data_ptr() for parameter in second.parameters()))
        board = np.zeros((6, 7), dtype=np.uint8)
        zero = np.zeros(board.shape)
        inputs = torch.from_numpy(np.stack([encode_global(board, side, zero, zero) for side in (1, 2)]))
        with torch.no_grad():
            logits, values = first.eval()(inputs)
        self.assertEqual(logits.shape, (2, 1, 6, 7))
        self.assertEqual(values.shape, (2,))
        self.assertFalse(torch.allclose(logits[0], logits[1]))

    def test_invalid_model_configuration_and_inputs(self):
        for config in (dict(base_channels=1), dict(token_dim=7, attention_heads=4), dict(attention_heads=0)):
            with self.assertRaises(ValueError):
                GlobalBoardNet(**config)
        model = GlobalBoardNet(base_channels=4, token_dim=8)
        for inputs in (torch.zeros(1, 8, 6, 7), torch.zeros(0, 9, 6, 7), torch.zeros(1, 9, 0, 7),
                       torch.zeros(1, 9, 6, 7, dtype=torch.int64), torch.full((1, 9, 6, 7), float("nan")),
                       torch.full((1, 9, 6, 7), 1.1)):
            with self.assertRaises(ValueError):
                model(inputs)


class GlobalMaskAndLossTests(unittest.TestCase):
    def test_global_softmax_hard_masks_illegal_and_ended_boards(self):
        boards = np.zeros((3, 6, 7), dtype=np.uint8)
        boards[0, 0, :3] = [1, 2, 3]
        boards[1, 2, :5] = 2
        boards[2] = 3
        logits = torch.zeros(3, 1, 6, 7)
        logits[0, 0, 0, :3] = 10000
        probabilities = masked_global_policy(logits, boards)
        self.assertTrue(torch.all(probabilities[0, 0, 0, :3] == 0))
        self.assertTrue(torch.all(probabilities[1:] == 0))
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertAlmostEqual(probabilities[0].sum().item(), 1, places=6)
        self.assertAlmostEqual(probabilities[0, 0, 1, 1].item(), 1 / 39, places=7)
        one = np.full((6, 7), 3, dtype=np.uint8)
        one[3, 4] = 0
        self.assertEqual(masked_global_policy(logits[:1], one)[0, 0, 3, 4].item(), 1)

    def test_supervised_loss_uses_only_known_values_and_legal_policies(self):
        boards = np.zeros((2, 6, 7), dtype=np.uint8)
        boards[:, 0, 0] = 3
        logits = torch.zeros(2, 1, 6, 7, requires_grad=True)
        values = torch.tensor([0.25, -0.5], requires_grad=True)
        target = torch.zeros_like(logits)
        target[:, 0, 1, 2] = 1
        result = global_loss(logits, values, target, [1, 1], [True, False], boards, return_components=True)
        self.assertAlmostEqual(result['policy_loss'].item(), math.log(41), places=6)
        self.assertAlmostEqual(result['value_loss'].item(), 0.75 ** 2, places=6)
        result['loss'].backward()
        self.assertEqual(values.grad[1].item(), 0)
        self.assertNotEqual(values.grad[0].item(), 0)
        self.assertTrue(torch.all(logits.grad[:, 0, 0, 0] == 0))

    def test_terminal_rows_can_train_value_without_policy(self):
        boards = np.zeros((2, 6, 7), dtype=np.uint8)
        boards[0, 0, :5] = 1
        boards[1] = 3
        logits = torch.zeros(2, 1, 6, 7, requires_grad=True)
        values = torch.tensor([0., 0.], requires_grad=True)
        result = global_loss(logits, values, torch.zeros_like(logits), [1., 0.], [True, True], boards,
                             return_components=True)
        self.assertEqual(result['policy_loss'].item(), 0)
        self.assertEqual(result['value_loss'].item(), 0.5)
        result['loss'].backward()
        self.assertTrue(torch.all(logits.grad == 0))
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_loss_backpropagates_to_both_heads_and_attention_without_optimizer(self):
        torch.set_num_threads(1)
        model = GlobalBoardNet(base_channels=4, token_dim=8, attention_heads=2)
        board = np.zeros((6, 7), dtype=np.uint8)
        zero = np.zeros(board.shape)
        logits, values = model(torch.from_numpy(encode_global(board, 1, zero, zero))[None])
        target = torch.zeros_like(logits)
        target[0, 0, 2, 3] = 1
        loss = global_loss(logits, values, target, [1], [True], board)
        loss.backward()
        for parameter in (model.policy_head.weight, model.value_head[-2].weight,
                          model.attention.self_attn.in_proj_weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_rejects_invalid_board_batch_and_target_support(self):
        board = np.zeros((6, 7), dtype=np.uint8)
        board[0, 0] = 3
        logits = torch.zeros(1, 1, 6, 7)
        target = torch.zeros_like(logits)
        target[0, 0, 2, 3] = 1
        for boards in (np.zeros((2, 6, 7), dtype=np.uint8), np.zeros((6, 7)), np.full((6, 7), 4)):
            with self.assertRaises(ValueError):
                masked_global_policy(logits, boards)
        bad = torch.zeros_like(target)
        bad[0, 0, 0, 0] = 1
        for targets in (bad, torch.zeros_like(target), torch.full_like(target, float('nan'))):
            with self.assertRaises(ValueError):
                global_loss(logits, [0], targets, [1], [True], board)
        for outcomes, mask in (([2], [True]), ([1], [2]), ([float('nan')], [False])):
            with self.assertRaises(ValueError):
                global_loss(logits, [0], target, outcomes, mask, board)


if __name__ == '__main__':
    unittest.main()
