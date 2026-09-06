import unittest

import numpy as np

from game import BLACK, WHITE, FORBIDDEN, PERMS, apply_move, legal_moves
from unet_codec import encode_rgb, policy_rgb
from unet_data import augment_sample, make_opponent_sample, make_play_sample


STATE = BLACK << 24 | WHITE << 2 | FORBIDDEN << 8


def uniform(state):
    p = np.zeros(25, dtype=np.float32)
    p[legal_moves(state)] = 1 / len(legal_moves(state))
    return p


class SampleTests(unittest.TestCase):
    def test_opponent_uses_the_position_before_the_observed_move(self):
        sample = make_opponent_sample(STATE, WHITE, 8)
        self.assertEqual((sample.state, sample.side, sample.role), (STATE, WHITE, "opponent"))
        np.testing.assert_array_equal(sample.rgb, encode_rgb(STATE))
        self.assertEqual(sample.target[8], 1)
        self.assertEqual(sample.target.sum(), 1)
        with self.assertRaisesRegex(ValueError, "已有棋子"):
            make_opponent_sample(apply_move(STATE, 8, WHITE), WHITE, 8)

    def test_explicit_color_is_not_inferred_from_local_counts(self):
        for side in (BLACK, WHITE):
            self.assertEqual(make_opponent_sample(STATE, side, 8).side, side)
        for side in (True, 0, 3, 1.0, None):
            with self.assertRaises(ValueError):
                make_opponent_sample(STATE, side, 8)

    def test_play_input_uses_prediction_without_target_leakage(self):
        prediction = uniform(STATE)
        first = make_play_sample(STATE, BLACK, 8, prediction)
        second = make_play_sample(STATE, BLACK, 9, prediction)
        np.testing.assert_array_equal(first.rgb, second.rgb)
        np.testing.assert_array_equal(first.rgb, policy_rgb(STATE, prediction, color="red", levels=None))
        self.assertFalse(np.array_equal(first.target, second.target))
        changed_prediction = np.zeros(25, dtype=np.float32)
        changed_prediction[10] = 1
        third = make_play_sample(STATE, BLACK, 8, changed_prediction)
        self.assertFalse(np.array_equal(first.rgb, third.rgb))
        np.testing.assert_array_equal(first.target, third.target)
        base = encode_rgb(STATE).reshape(3, 25)
        np.testing.assert_array_equal(first.rgb.reshape(3, 25)[:, ~first.legal_mask], base[:, ~first.legal_mask])

    def test_targets_and_predictions_reject_illegal_or_malformed_mass(self):
        good = uniform(STATE)
        bads = [np.ones((1, 25)), np.zeros(25), np.ones(25), np.full(25, np.nan),
                np.full(25, np.inf), np.full(25, -1), np.ones(25, dtype=complex)]
        for move in (1, 4, 12):
            bad = np.zeros(25)
            bad[move] = 1
            bads.append(bad)
            with self.assertRaises(ValueError):
                make_opponent_sample(STATE, WHITE, move)
        for bad in bads:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                make_play_sample(STATE, WHITE, bad, good)
            with self.assertRaises(ValueError):
                make_play_sample(STATE, WHITE, good, bad)
        for move in (-1, 25, True, 8.0):
            with self.assertRaises(ValueError):
                make_opponent_sample(STATE, WHITE, move)

    def test_terminal_positions_never_receive_action_labels(self):
        won = sum(BLACK << (2 * i) for i in range(5))
        full = sum(FORBIDDEN << (2 * i) for i in range(25))
        for state in (won, full):
            with self.assertRaisesRegex(ValueError, "terminal"):
                make_opponent_sample(state, WHITE, 8)
            with self.assertRaisesRegex(ValueError, "terminal"):
                make_play_sample(state, WHITE, 8, np.ones(25) / 25)

    def test_d4_transforms_keep_rgb_target_mask_and_board_aligned(self):
        sample = make_play_sample(STATE, WHITE, 8, uniform(STATE))
        for symmetry, permutation in enumerate(PERMS):
            transformed = augment_sample(sample, symmetry)
            self.assertEqual((transformed.side, transformed.role), (WHITE, "play"))
            self.assertEqual(transformed.target[permutation[8]], 1)
            self.assertEqual(transformed.target.sum(), 1)
            np.testing.assert_array_equal(transformed.legal_mask, np.array([
                i in legal_moves(transformed.state) for i in range(25)]))
            for old, new in enumerate(permutation):
                np.testing.assert_array_equal(transformed.rgb.reshape(3, 25)[:, new], sample.rgb.reshape(3, 25)[:, old])
            self.assertEqual(transformed.rgb.dtype, np.float32)
        with self.assertRaises(ValueError):
            augment_sample(sample, 8)


if __name__ == "__main__":
    unittest.main()
