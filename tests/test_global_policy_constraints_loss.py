"""Small, analytic policy-constraint regressions; no model training or optimizer."""
import math
import unittest

import numpy as np
import torch

from global_model import global_loss, masked_global_policy


class GlobalPolicyConstraintsLossTests(unittest.TestCase):
    moves = ((2, 1), (2, 2), (2, 3))

    def fixture(self, batch=1, shape=(5, 5)):
        boards = np.full((batch, *shape), 3, dtype=np.uint8)
        boards[:, 0, 0] = 1
        boards[:, 0, 1] = 2
        for row, col in self.moves:
            boards[:, row, col] = 0
        logits = torch.zeros((batch, 1, *shape), dtype=torch.float64, requires_grad=True)
        targets = torch.zeros((batch, *shape), dtype=torch.float64)
        return boards, logits, targets

    def call(self, boards, logits, targets, **options):
        batch = len(boards)
        args = dict(values=torch.zeros(batch, dtype=logits.dtype),
                    target_values=torch.zeros(batch, dtype=logits.dtype),
                    value_mask=torch.zeros(batch, dtype=torch.bool),
                    return_components=True)
        args.update(options)
        return global_loss(logits, target_policy=targets, boards=boards, **args)

    def constraint(self, boards, *moves, item=0):
        mask = torch.zeros(boards.shape, dtype=torch.bool)
        for row, col in moves:
            mask[item, row, col] = True
        return mask

    def gradients_at_moves(self, logits, item=0):
        return torch.stack([logits.grad[item, 0, r, c] for r, c in self.moves])

    def test_partial_losing_gradient_reduces_bad_mass_without_picking_unknown(self):
        boards, logits, targets = self.fixture()
        bad = self.constraint(boards, self.moves[0])
        result = self.call(boards, logits, targets, policy_mask=[False], losing_mask=bad)
        self.assertAlmostEqual(result["losing_loss"].item(), math.log(1.5))
        self.assertEqual(result["losing_count"].item(), 1)
        self.assertEqual(result["ce_count"].item(), 0)
        result["loss"].backward()
        torch.testing.assert_close(self.gradients_at_moves(logits),
                                   torch.tensor([1/3, -1/6, -1/6], dtype=logits.dtype))
        self.assertTrue(torch.all(logits.grad[:, 0][torch.from_numpy(boards != 0)] == 0))
        # Constraints train the distribution; they never hard-mask bad actions.
        policy = masked_global_policy(logits.detach(), boards)
        for row, col in self.moves:
            self.assertAlmostEqual(policy[0, 0, row, col].item(), 1/3)

    def test_multiple_winning_moves_remain_symmetric(self):
        boards, logits, targets = self.fixture()
        good = self.constraint(boards, *self.moves[:2])
        result = self.call(boards, logits, targets, policy_mask=[False], winning_mask=good)
        self.assertAlmostEqual(result["winning_loss"].item(), math.log(1.5))
        self.assertEqual(result["winning_count"].item(), 1)
        result["loss"].backward()
        torch.testing.assert_close(self.gradients_at_moves(logits),
                                   torch.tensor([-1/6, -1/6, 1/3], dtype=logits.dtype))

    def test_constraint_terms_have_independent_row_denominators(self):
        boards, logits, targets = self.fixture(batch=3)
        bad = self.constraint(boards, self.moves[0], item=0)
        good = self.constraint(boards, self.moves[2], item=1)
        result = self.call(boards, logits, targets, policy_mask=[False] * 3,
                           losing_mask=bad, winning_mask=good, policy_weights=[0.1] * 3)
        self.assertAlmostEqual(result["losing_loss"].item(), math.log(1.5))
        self.assertAlmostEqual(result["winning_loss"].item(), math.log(3))
        self.assertAlmostEqual(result["policy_loss"].item(), math.log(1.5) + math.log(3))
        self.assertEqual([result[k].item() for k in ("ce_count", "losing_count", "winning_count")],
                         [0, 1, 1])
        result["loss"].backward()
        self.assertTrue(torch.all(logits.grad[2] == 0))

    def test_uniform_ce_weight_scales_loss_and_gradient(self):
        boards, logits, targets = self.fixture(batch=3)
        targets[:2, 2, 1] = 1
        ordinary = self.call(boards, logits, targets, policy_mask=[True, True, False])
        ordinary["loss"].backward()
        ordinary_grad = logits.grad.clone()
        weighted_logits = logits.detach().clone().requires_grad_()
        weighted = self.call(boards, weighted_logits, targets, policy_mask=[True, True, False],
                             policy_weights=[0.1, 0.1, 0.1])
        self.assertAlmostEqual(weighted["ce_loss"].item(), 0.1 * math.log(3))
        self.assertEqual(weighted["ce_count"].item(), 2)
        weighted["loss"].backward()
        torch.testing.assert_close(weighted_logits.grad, 0.1 * ordinary_grad)
        self.assertTrue(torch.all(weighted_logits.grad[2] == 0))

    def test_zero_weight_row_stays_in_active_ce_denominator(self):
        boards, logits, targets = self.fixture(batch=2)
        targets[:, 2, 1] = 1
        result = self.call(boards, logits, targets, policy_weights=[0, 1])
        self.assertAlmostEqual(result["ce_loss"].item(), math.log(3)/2)
        self.assertEqual(result["ce_count"].item(), 2)
        result["loss"].backward()
        self.assertTrue(torch.all(logits.grad[0] == 0))

    def test_empty_constraints_and_disabled_ce_have_zero_finite_gradient(self):
        boards, logits, targets = self.fixture()
        empty = self.constraint(boards)
        result = self.call(boards, logits, targets, policy_mask=[False],
                           losing_mask=empty, winning_mask=empty)
        self.assertEqual(result["loss"].item(), 0)
        result["loss"].backward()
        self.assertTrue(torch.all(torch.isfinite(logits.grad)))
        self.assertTrue(torch.all(logits.grad == 0))

    def test_all_losing_disables_ce_even_when_requested_and_has_no_value_gradient_at_zero_weight(self):
        for policy_mask in (None, [True], [False]):
            with self.subTest(policy_mask=policy_mask):
                boards, logits, targets = self.fixture()
                values = torch.tensor([0.5], dtype=logits.dtype, requires_grad=True)
                result = self.call(boards, logits, targets, policy_mask=policy_mask,
                                   losing_mask=torch.from_numpy(boards == 0),
                                   values=values, target_values=[1.0], value_mask=[True], value_weight=0)
                self.assertEqual(result["policy_loss"].item(), 0)
                self.assertEqual(result["value_loss"].item(), 0.25)
                self.assertEqual(result["ce_count"].item(), 0)
                self.assertEqual(result["losing_count"].item(), 0)
                result["loss"].backward()
                self.assertTrue(torch.all(logits.grad == 0))
                self.assertTrue(torch.all(values.grad == 0))

    def test_all_losing_rejects_old_uniform_ce_target(self):
        boards, logits, targets = self.fixture()
        for row, col in self.moves:
            targets[0, row, col] = 1/3
        with self.assertRaisesRegex(ValueError, "inactive CE"):
            self.call(boards, logits, targets, losing_mask=torch.from_numpy(boards == 0))

    def test_all_winning_set_has_zero_loss_and_gradient(self):
        boards, logits, targets = self.fixture()
        result = self.call(boards, logits, targets, policy_mask=[False],
                           winning_mask=torch.from_numpy(boards == 0))
        self.assertEqual(result["winning_loss"].item(), 0)
        self.assertEqual(result["winning_count"].item(), 1)
        result["loss"].backward()
        self.assertTrue(torch.all(logits.grad == 0))

    def test_value_mse_and_numeric_value_mask_remain_compatible(self):
        boards, logits, targets = self.fixture(batch=2)
        result = self.call(boards, logits, targets, policy_mask=[False, False],
                           values=[0.5, -0.5], target_values=[1, 1], value_mask=[1, 0],
                           value_weight=0.3)
        self.assertEqual(result["value_loss"].item(), 0.25)
        self.assertAlmostEqual(result["loss"].item(), 0.075)

    def test_terminal_policy_is_zero_but_value_can_be_labelled(self):
        boards, logits, targets = self.fixture()
        boards[0, 0, :] = 1
        values = torch.tensor([0.25], dtype=logits.dtype, requires_grad=True)
        result = self.call(boards, logits, targets, values=values,
                           target_values=[1], value_mask=[True], policy_mask=[True])
        self.assertEqual(result["ce_count"].item(), 0)
        self.assertEqual(result["value_loss"].item(), 0.75 ** 2)
        result["loss"].backward()
        self.assertTrue(torch.all(logits.grad == 0))
        self.assertAlmostEqual(values.grad.item(), -1.5)
        with self.assertRaisesRegex(ValueError, "legal moves"):
            self.call(boards, logits, targets, policy_mask=[False],
                      losing_mask=self.constraint(boards, self.moves[0]))

    def test_full_gray_board_supports_value_only(self):
        boards, logits, targets = self.fixture()
        boards.fill(3)
        result = self.call(boards, logits, targets, values=[-0.5], target_values=[1],
                           value_mask=[True])
        self.assertEqual(result["policy_loss"].item(), 0)
        self.assertEqual(result["value_loss"].item(), 2.25)
        result["loss"].backward()
        self.assertTrue(torch.all(logits.grad == 0))

    def test_omitted_options_match_original_soft_ce_formula(self):
        boards, logits, targets = self.fixture()
        scores = [0, math.log(2), math.log(3)]
        with torch.no_grad():
            for score, (row, col) in zip(scores, self.moves):
                logits[0, 0, row, col] = score
        for weight, (row, col) in zip([0.2, 0.3, 0.5], self.moves):
            targets[0, row, col] = weight
        expected_ce = -sum(w * math.log(p) for w, p in zip([0.2, 0.3, 0.5], [1/6, 2/6, 3/6]))
        old = self.call(boards, logits, targets, values=[0.5], target_values=[-1],
                        value_mask=[True], value_weight=0.3)
        explicit = self.call(boards, logits, targets, values=[0.5], target_values=[-1],
                             value_mask=[True], value_weight=0.3, policy_mask=[True],
                             policy_weights=[1], losing_mask=self.constraint(boards),
                             winning_mask=self.constraint(boards))
        self.assertAlmostEqual(old["policy_loss"].item(), expected_ce)
        self.assertAlmostEqual(old["loss"].item(), expected_ce + 0.3 * 2.25)
        torch.testing.assert_close(old["loss"], explicit["loss"], rtol=0, atol=0)
        for key in ("loss", "policy_loss", "value_loss", "ce_loss", "losing_loss", "winning_loss"):
            self.assertEqual(old[key].ndim, 0)
        for key in ("ce_count", "losing_count", "winning_count"):
            self.assertEqual(old[key].dtype, torch.int64)

    def test_inactive_live_target_must_be_exactly_zero(self):
        boards, logits, targets = self.fixture()
        targets[0, 2, 1] = 1e-8
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            self.call(boards, logits, targets, policy_mask=[False])

    def test_active_target_must_be_normalized_and_legal(self):
        boards, logits, targets = self.fixture()
        with self.assertRaisesRegex(ValueError, "sum to one"):
            self.call(boards, logits, targets, policy_mask=[True])
        targets[0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "legal support"):
            self.call(boards, logits, targets)

    def test_new_masks_require_boolean_dtype_and_exact_shape(self):
        boards, logits, targets = self.fixture()
        cases = [
            {"policy_mask": [0]},
            {"policy_mask": [[False]]},
            {"policy_mask": []},
            {"losing_mask": np.zeros(boards.shape, dtype=np.int64)},
            {"winning_mask": np.zeros(boards.shape, dtype=np.float32)},
            {"losing_mask": torch.zeros_like(logits, dtype=torch.bool)},
            {"winning_mask": torch.zeros((5, 5), dtype=torch.bool)},
        ]
        for options in cases:
            with self.subTest(options=tuple(options)):
                args = dict(policy_mask=[False])
                args.update(options)
                with self.assertRaisesRegex(ValueError, "boolean tensor"):
                    self.call(boards, logits, targets, **args)

    def test_constraints_reject_occupied_gray_and_overlapping_support(self):
        boards, logits, targets = self.fixture()
        for position in ((0, 0), (0, 1), (0, 2)):
            for key in ("losing_mask", "winning_mask"):
                with self.subTest(position=position, key=key):
                    with self.assertRaisesRegex(ValueError, "legal moves"):
                        self.call(boards, logits, targets, policy_mask=[False],
                                  **{key: self.constraint(boards, position)})
        both = self.constraint(boards, self.moves[0])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.call(boards, logits, targets, policy_mask=[False],
                      losing_mask=both, winning_mask=both)

    def test_policy_weights_reject_invalid_values_and_shapes(self):
        boards, logits, targets = self.fixture()
        for weights in ([-1], [float("nan")], [float("inf")], [], [[0.1]], [True], [1+2j]):
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError, "policy_weights"):
                    self.call(boards, logits, targets, policy_mask=[False],
                              policy_weights=weights)

    def test_extreme_finite_logits_stay_stable(self):
        boards, logits, targets = self.fixture()
        with torch.no_grad():
            logits[0, 0, 2, 1] = 10000
            logits[0, 0, 2, 2] = -10000
            logits[0, 0, 2, 3] = -10000
        result = self.call(boards, logits, targets, policy_mask=[False],
                           losing_mask=self.constraint(boards, self.moves[0]),
                           winning_mask=self.constraint(boards, self.moves[1]))
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertAlmostEqual(result["losing_loss"].item(), 20000-math.log(2))
        self.assertAlmostEqual(result["winning_loss"].item(), 20000)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.all(logits.grad[:, 0][torch.from_numpy(boards != 0)] == 0))

    def test_rectangular_d4_and_color_swap_preserve_loss_and_gradients(self):
        boards, logits, targets = self.fixture(shape=(5, 7))
        bad = self.constraint(boards, self.moves[0])
        good = self.constraint(boards, self.moves[1])
        with torch.no_grad():
            logits[0, 0, 2, 1] = 0.7
            logits[0, 0, 2, 2] = -0.2
            logits[0, 0, 2, 3] = 0.1
        reference = self.call(boards, logits, targets, policy_mask=[False],
                              losing_mask=bad, winning_mask=good)
        reference["loss"].backward()
        for turns in range(4):
            for reflection in (False, True):
                for swap in (False, True):
                    with self.subTest(turns=turns, reflection=reflection, swap=swap):
                        def transformed_tensor(value):
                            out = torch.rot90(value, turns, (-2, -1))
                            return out.flip(-1) if reflection else out
                        rotated_boards = np.rot90(boards, turns, axes=(-2, -1)).copy()
                        if reflection:
                            rotated_boards = rotated_boards[..., ::-1].copy()
                        if swap:
                            rotated_boards = np.where(rotated_boards == 1, 2,
                                                      np.where(rotated_boards == 2, 1, rotated_boards))
                        rotated_logits = transformed_tensor(logits.detach()).clone().requires_grad_()
                        result = self.call(rotated_boards, rotated_logits, transformed_tensor(targets),
                                           policy_mask=[False], losing_mask=transformed_tensor(bad),
                                           winning_mask=transformed_tensor(good))
                        torch.testing.assert_close(result["loss"], reference["loss"])
                        result["loss"].backward()
                        torch.testing.assert_close(rotated_logits.grad,
                                                   transformed_tensor(logits.grad))


if __name__ == "__main__":
    unittest.main()
