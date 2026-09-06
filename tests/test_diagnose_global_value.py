"""Tiny diagnostic checks; no training or checkpoint writes."""
import unittest
import numpy as np

from diagnose_global_value import encode_diagnostic, select_balanced, value_statistics


def record(side=1, value=1, name="fixture"):
    board = np.zeros((6, 7), dtype=np.uint8)
    board[0, 1], board[2, 4], board[5, 6] = 1, 2, 3
    return dict(board=board, side=side, value=value, value_valid=True,
                search_proven_value=value, name=name)


class GlobalValueDiagnosticTests(unittest.TestCase):
    def test_role_encodings_are_invariant_to_joint_color_actor_exchange(self):
        row = record(side=2)
        for representation in ("relative_rgb", "role_planes"):
            plain = encode_diagnostic(row, representation)
            changed = encode_diagnostic(row, representation, swap_colors=True)
            np.testing.assert_array_equal(plain, changed)
            self.assertTrue(np.all(plain[3:5] == 0))
            self.assertTrue(np.all(plain[6] == -1))
        absolute = encode_diagnostic(row, "absolute_rgb")
        swapped = encode_diagnostic(row, "absolute_rgb", swap_colors=True)
        self.assertFalse(np.array_equal(absolute[:3], swapped[:3]))
        np.testing.assert_array_equal(absolute[6], -swapped[6])

    def test_role_planes_mark_current_actor_opponent_and_gray_explicitly(self):
        row = record(side=2)
        inputs = encode_diagnostic(row, "role_planes")
        np.testing.assert_array_equal(inputs[0], row["board"] == 2)
        np.testing.assert_array_equal(inputs[1], row["board"] == 1)
        np.testing.assert_array_equal(inputs[2], row["board"] == 3)
        np.testing.assert_array_equal(inputs[5], row["board"] == 0)
        with self.assertRaises(ValueError):
            encode_diagnostic(row, "unrecognized")

    def test_balancing_uses_existing_proof_and_value_without_relabeling(self):
        rows = [record(side, value, f"{side}:{value}:{n}")
                for side in (1, 2) for value in (-1, 1) for n in range(3)]
        extra = record(1, 1)
        extra["value"] = -1
        selected = select_balanced(rows + [extra], 8, seed=9)
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(row is not extra for row in selected))
        for side in (1, 2):
            for value in (-1, 1):
                self.assertEqual(sum(row["side"] == side and row["value"] == value for row in selected), 2)
        self.assertEqual([r["name"] for r in selected],
                         [r["name"] for r in select_balanced(rows, 8, seed=9)])
        with self.assertRaises(ValueError):
            select_balanced(rows, 7)
        with self.assertRaises(ValueError):
            select_balanced(rows, 16)

    def test_statistics_do_not_count_unknown_value_as_draw(self):
        rows = [record(1, 1), record(2, -1), record(1, 0)]
        rows[-1]["value_valid"] = False
        stats = value_statistics([0, 0, 0], rows)
        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["mse"], 1)
        self.assertEqual(stats["zero_baseline_mse"], 1)
        self.assertEqual(stats["outcome_accuracy"], 0)
        self.assertEqual(stats["label_counts"], {"1": 1, "-1": 1})


if __name__ == "__main__":
    unittest.main()
