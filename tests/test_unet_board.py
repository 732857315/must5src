"""Whole-board batch fusion/tactics checks using fake forwards, never training."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from board_rules import winning_cells
from game import BLACK, WHITE, FORBIDDEN, legal_moves
from unet_codec import FORBIDDEN_RGB, encode_rgb, policy_rgb
from windows import extract_window
import unet_board


class FixedModel(nn.Module):
    def __init__(self, logits=None, fail=False):
        super().__init__()
        self.logits = torch.zeros(25) if logits is None else torch.as_tensor(logits, dtype=torch.float32)
        self.fail = fail
        self.calls = []

    def forward(self, rgb, side):
        if self.fail:
            raise AssertionError("Terminal board called a model")
        self.calls.append((rgb.clone(), side.clone()))
        return self.logits.reshape(1, 1, 5, 5).repeat(len(rgb), 1, 1, 1)


def recorded_images(model):
    return np.concatenate([image.numpy() for image, _ in model.calls], axis=0)


class WholeBoardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_live_contract(self, result, board):
        self.assertFalse(result["terminal"])
        legal = board == 0
        for field in ("opponent_policy", "play_policy", "raw_play_policy"):
            probabilities = result[field]
            self.assertEqual(probabilities.shape, board.shape)
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
            self.assertTrue(np.all(probabilities[~legal] == 0))
            self.assertAlmostEqual(float(probabilities.sum()), 1, places=6)
        self.assertTrue(np.all(result["coverage"][legal] > 0))
        self.assertTrue(np.all(result["coverage"][~legal] == 0))
        self.assertTrue(np.all(result["coverage"] <= 25))
        self.assertEqual(board[result["move"]], 0)
        json.dumps({key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in result.items()}, allow_nan=False)

    def test_both_armies_are_primary_centers_and_every_empty_cell_is_covered(self):
        board = np.zeros((9, 11), dtype=np.uint8)
        board[0, 0], board[4, 4], board[4, 6], board[8, 10] = BLACK, BLACK, WHITE, WHITE
        board[0, 10], board[8, 0] = FORBIDDEN, FORBIDDEN
        original = board.copy()
        opponent, play = FixedModel(), FixedModel()
        result = unet_board.analyze_board(board, BLACK, opponent, play, tactical=False)
        expected_stones = [(0, 0), (4, 4), (4, 6), (8, 10)]
        self.assertEqual(result["stone_centers"], expected_stones)
        self.assertEqual(result["window_centers"][:4], expected_stones)
        self.assertTrue(set(expected_stones) <= set(result["window_centers"]))
        self.assertEqual(len(set(result["window_centers"])), result["window_count"])
        self.assert_live_contract(result, board)
        np.testing.assert_array_equal(board, original)
        self.assertEqual(len(opponent.calls), 1)
        self.assertEqual(len(play.calls), 1)
        self.assertEqual(opponent.calls[0][0].shape[0], result["window_count"])

    def test_corner_windows_pad_gray_without_coordinate_wraparound(self):
        board = np.zeros((8, 10), dtype=np.uint8)
        board[0, 0] = BLACK
        board[7, 9] = WHITE
        opponent, play = FixedModel(), FixedModel()
        result = unet_board.analyze_board(board, WHITE, opponent, play, tactical=False)
        images = recorded_images(opponent)
        for center in ((0, 0), (7, 9)):
            index = result["window_centers"].index(center)
            expected = encode_rgb(extract_window(board, *center).state)
            np.testing.assert_array_equal(images[index], expected)
        corner = images[result["window_centers"].index((0, 0))]
        gray = np.array(FORBIDDEN_RGB, dtype=np.float32) / 255
        np.testing.assert_allclose(corner[:, 0, 0], gray)
        np.testing.assert_allclose(corner[:, 1, 4], gray)

    def test_explicit_global_side_and_continuous_red_are_used_for_every_window(self):
        board = np.zeros((8, 9), dtype=np.uint8)
        board[1, 2], board[5, 6], board[3, 4] = BLACK, WHITE, FORBIDDEN
        opponent = FixedModel(torch.arange(25) / 7)
        play = FixedModel()
        result = unet_board.analyze_board(board, WHITE, opponent, play, tactical=False)
        self.assertTrue(all(torch.all(side == BLACK) for _, side in opponent.calls))
        self.assertTrue(all(torch.all(side == WHITE) for _, side in play.calls))
        red_images = recorded_images(play)
        for index, center in enumerate(result["window_centers"]):
            window = extract_window(board, *center)
            moves = legal_moves(window.state)
            probabilities = torch.zeros(25)
            probabilities[moves] = torch.softmax(opponent.logits[moves], dim=0)
            expected = policy_rgb(window.state, probabilities.numpy(), "red", levels=None)
            np.testing.assert_array_equal(red_images[index], expected)

    def test_uniform_windows_do_not_bias_corners_or_sparsely_playable_windows(self):
        board = np.zeros((13, 17), dtype=np.uint8)
        board[0, 0], board[6, 8], board[12, 16] = BLACK, WHITE, BLACK
        board[1:4, 0:2] = FORBIDDEN
        board[9:12, 13:16] = FORBIDDEN
        result = unet_board.analyze_board(board, BLACK, FixedModel(), FixedModel(), tactical=False)
        expected = np.zeros(board.shape)
        expected[board == 0] = 1 / np.count_nonzero(board == 0)
        np.testing.assert_allclose(result["opponent_policy"], expected, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(result["raw_play_policy"], expected, rtol=1e-6, atol=1e-10)
        self.assertEqual(result["fusion"]["method"], "mean_relative_to_local_uniform")
        self.assert_live_contract(result, board)

    def test_overlapping_fusion_matches_independent_relative_preference_average(self):
        board = np.zeros((7, 9), dtype=np.uint8)
        board[2, 2], board[4, 5], board[0, 8] = BLACK, WHITE, FORBIDDEN
        logits = np.linspace(-2, 2, 25).astype(np.float32)
        result = unet_board.analyze_board(board, BLACK, FixedModel(logits), FixedModel(logits), tactical=False)
        total, coverage = np.zeros(board.shape), np.zeros(board.shape, dtype=np.int32)
        for center in result["window_centers"]:
            window = extract_window(board, *center)
            moves = legal_moves(window.state)
            weights = np.exp(logits[moves].astype(np.float64) - float(logits[moves].max()))
            probabilities = weights / weights.sum()
            for move, probability in zip(moves, probabilities):
                coordinate = window.coordinates[move]
                total[coordinate] += probability * len(moves)
                coverage[coordinate] += 1
        expected = np.zeros(board.shape)
        expected[board == 0] = total[board == 0] / coverage[board == 0]
        expected /= expected.sum()
        np.testing.assert_array_equal(result["coverage"], coverage)
        np.testing.assert_allclose(result["raw_play_policy"], expected, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(result["opponent_policy"], expected, rtol=1e-6, atol=1e-9)

    def test_empty_board_starts_at_center_and_adds_deterministic_coverage(self):
        board = np.zeros((16, 16), dtype=np.uint8)
        first = unet_board.analyze_board(board, BLACK, FixedModel(), FixedModel(), tactical=False)
        second = unet_board.analyze_board(board.tolist(), BLACK, FixedModel(), FixedModel(), tactical=False)
        self.assertEqual(first["stone_centers"], [])
        self.assertEqual(first["window_centers"][0], (8, 8))
        self.assertTrue(first["extra_centers"])
        self.assertEqual(first["window_centers"], second["window_centers"])
        np.testing.assert_array_equal(first["coverage"], second["coverage"])
        np.testing.assert_array_equal(first["raw_play_policy"], second["raw_play_policy"])
        self.assert_live_contract(first, board)

    def test_rectangular_and_narrow_boards_have_valid_coverage(self):
        for shape in ((6, 19), (19, 6), (1, 7), (7, 1), (1, 1)):
            with self.subTest(shape=shape):
                board = np.zeros(shape, dtype=np.uint8)
                result = unet_board.analyze_board(board, BLACK, FixedModel(), FixedModel(), tactical=False)
                self.assert_live_contract(result, board)

    def test_window_forward_batches_are_bounded(self):
        board = np.zeros((16, 18), dtype=np.uint8)
        opponent, play = FixedModel(), FixedModel()
        with patch.object(unet_board, "WINDOW_BATCH_SIZE", 3):
            result = unet_board.analyze_board(board, BLACK, opponent, play, tactical=False)
        self.assertGreater(len(opponent.calls), 1)
        self.assertEqual(len(opponent.calls), len(play.calls))
        self.assertEqual(len(opponent.calls), result["model_batches"])
        self.assertTrue(all(len(images) <= 3 for images, _ in opponent.calls))
        self.assertEqual(sum(len(images) for images, _ in opponent.calls), result["window_count"])
        self.assert_live_contract(result, board)

    def test_all_forbidden_local_window_is_skipped(self):
        board = np.full((9, 9), FORBIDDEN, dtype=np.uint8)
        board[8, 8] = 0
        opponent, play = FixedModel(), FixedModel()
        result = unet_board.analyze_board(board, BLACK, opponent, play, tactical=False)
        self.assertIn((4, 4), result["skipped_window_centers"])
        self.assertEqual(result["window_count"], 1)
        self.assertEqual(result["move"], (8, 8))
        self.assertEqual(opponent.calls[0][0].shape[0], 1)
        self.assert_live_contract(result, board)

    def test_global_terminal_overline_or_no_empty_calls_no_model(self):
        won = np.zeros((10, 13), dtype=np.uint8)
        won[7, 4:10] = WHITE
        blocked = np.full((6, 8), FORBIDDEN, dtype=np.uint8)
        for board, expected_winner in ((won, WHITE), (blocked, 0)):
            result = unet_board.analyze_board(board, BLACK, FixedModel(fail=True), FixedModel(fail=True))
            self.assertTrue(result["terminal"])
            self.assertEqual(result["winner"], expected_winner)
            self.assertIsNone(result["move"])
            self.assertEqual(result["window_count"], 0)
            self.assertEqual(result["model_batches"], 0)
            for field in ("opponent_policy", "play_policy", "raw_play_policy", "coverage"):
                self.assertEqual(result[field].shape, board.shape)
                self.assertTrue(np.all(result[field] == 0))

    def test_gap_spanning_stone_windows_is_found_as_full_board_win(self):
        board = np.zeros((12, 13), dtype=np.uint8)
        board[6, [4, 5, 7, 8]] = BLACK
        self.assertEqual(winning_cells(board, BLACK), [(6, 6)])
        result = unet_board.analyze_board(board, BLACK, FixedModel(), FixedModel())
        self.assertEqual(result["move"], (6, 6))
        self.assertEqual(result["tactical_kind"], "immediate_win")
        self.assertTrue(result["tactical_applied"])
        self.assertEqual(result["play_policy"][6, 6], 1)
        self.assertLess(result["raw_play_policy"][6, 6], 1)
        self.assert_live_contract(result, board)

    def test_unique_full_board_defense_and_raw_mode_remain_distinct(self):
        board = np.zeros((12, 13), dtype=np.uint8)
        board[6, [4, 5, 7, 8]] = WHITE
        board[1, 1] = BLACK
        result = unet_board.analyze_board(board, BLACK, FixedModel(), FixedModel())
        self.assertEqual(result["move"], (6, 6))
        self.assertEqual(result["tactical_kind"], "unique_defense")
        self.assertEqual(result["play_policy"][6, 6], 1)
        raw = unet_board.analyze_board(board, BLACK, FixedModel(), FixedModel(), tactical=False)
        self.assertFalse(raw["tactical_applied"])
        self.assertIsNone(raw["tactical_kind"])
        np.testing.assert_array_equal(raw["play_policy"], raw["raw_play_policy"])
        np.testing.assert_array_equal(raw["raw_play_policy"], result["raw_play_policy"])

    def test_own_immediate_win_precedes_opponents_unique_threat(self):
        board = np.zeros((14, 14), dtype=np.uint8)
        board[6, [4, 5, 7, 8]] = BLACK
        board[10, [7, 8, 10, 11]] = WHITE
        result = unet_board.analyze_board(board, BLACK, FixedModel(), FixedModel())
        self.assertEqual(result["move"], (6, 6))
        self.assertEqual(result["tactical_kind"], "immediate_win")

    def test_forbidden_gap_breaks_tactics_and_multiple_blocks_are_not_called_unique(self):
        board = np.zeros((12, 13), dtype=np.uint8)
        board[6, [4, 5, 7, 8]] = WHITE
        board[6, 6] = FORBIDDEN
        result = unet_board.analyze_board(board, BLACK, FixedModel(), FixedModel())
        self.assertIsNone(result["tactical_kind"])
        board = np.zeros((12, 13), dtype=np.uint8)
        board[3, :4] = WHITE
        board[8, :4] = WHITE
        self.assertEqual(len(winning_cells(board, WHITE)), 2)
        result = unet_board.analyze_board(board, BLACK, FixedModel(), FixedModel())
        self.assertIsNone(result["tactical_kind"])
        self.assertIn("多个", result["reason"])
        np.testing.assert_array_equal(result["play_policy"], result["raw_play_policy"])

    def test_structural_double_four_with_shared_completion_remains_one_defense(self):
        board = np.zeros((13, 13), dtype=np.uint8)
        board[6, [4, 5, 7, 8]] = WHITE
        board[[4, 5, 7, 8], 6] = WHITE
        result = unet_board.analyze_board(board, BLACK, FixedModel(), FixedModel())
        threats = result["threats"]["opponent"]
        self.assertTrue(threats["structural_only"])
        self.assertTrue(threats["double_four"])
        self.assertFalse(threats["double_kill"])
        self.assertEqual(threats["winning_cells"], [(6, 6)])
        self.assertEqual(threats["shared_completion_cells"], [(6, 6)])
        self.assertEqual(result["tactical_kind"], "unique_defense")
        self.assertEqual(result["move"], (6, 6))
        self.assertEqual(board[6, 6], 0)

    def test_double_three_is_structural_information_not_forbidden_or_a_proof(self):
        board = np.zeros((13, 13), dtype=np.uint8)
        board[6, 5:8] = BLACK
        board[5:8, 6] = BLACK
        result = unet_board.analyze_board(board, BLACK, FixedModel(), FixedModel())
        threats = result["threats"]["self"]
        self.assertTrue(threats["double_three"])
        self.assertTrue(threats["structural_only"])
        self.assertFalse(threats["double_kill"])
        self.assertFalse(result["tactical_applied"])
        self.assertNotIn("proven_value", result)
        np.testing.assert_array_equal(result["raw_play_policy"], result["play_policy"])
        self.assert_live_contract(result, board)

    def test_terminal_threat_fields_are_present_and_empty(self):
        board = np.full((7, 9), FORBIDDEN, dtype=np.uint8)
        result = unet_board.analyze_board(board, BLACK, FixedModel(fail=True), FixedModel(fail=True))
        for role in ("self", "opponent"):
            threats = result["threats"][role]
            self.assertTrue(threats["terminal"])
            self.assertEqual(threats["winning_cells"], [])
            self.assertFalse(threats["double_kill"])

    def test_invalid_global_side_is_not_silently_inferred(self):
        board = np.zeros((8, 8), dtype=np.uint8)
        for side in (None, 0, 3, True, 1.0):
            with self.assertRaises(ValueError):
                unet_board.analyze_board(board, side, FixedModel(), FixedModel())
        with self.assertRaises(ValueError):
            unet_board.analyze_board(board, BLACK, FixedModel(), FixedModel(), tactical="yes")


if __name__ == "__main__":
    unittest.main()