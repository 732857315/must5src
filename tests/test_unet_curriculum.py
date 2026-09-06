import unittest
import random
from unittest.mock import patch

import numpy as np

from exact_solver import ExactSolver
from arena import PlayerAB
from game import BLACK, WHITE, FORBIDDEN, PERMS, LINE_INDICES, empty_count, legal_moves, player_at, winner
from unet_curriculum import build_curriculum_dataset, canonical_key, dataset_report, _tactical_position, base_key, _exact_analysis, _teach
from unet_data import augment_sample, make_play_sample


def swap(state):
    return sum((3 - cell if cell in (BLACK, WHITE) else cell) << (2 * i)
               for i in range(25) for cell in [((state >> (2 * i)) & 3)])


class CurriculumTests(unittest.TestCase):
    def test_canonical_identity_combines_d4_and_color_side_reversal(self):
        state = BLACK << 24 | WHITE << 2 | FORBIDDEN << 8
        key = canonical_key(state, WHITE)
        for perm in PERMS:
            transformed = sum(((state >> (2 * old)) & 3) << (2 * new) for old, new in enumerate(perm))
            self.assertEqual(canonical_key(transformed, WHITE), key)
            self.assertEqual(canonical_key(swap(transformed), BLACK), key)
        self.assertNotEqual(canonical_key(state, BLACK), key)

    def test_color_augmentation_preserves_red_empty_pixels_and_identity(self):
        state = BLACK << 24 | WHITE << 2 | FORBIDDEN << 8
        prediction = np.zeros(25, dtype=np.float32)
        prediction[8] = 1
        sample = make_play_sample(state, WHITE, 9, prediction)
        changed = augment_sample(sample, 0, color_swap=True)
        self.assertEqual(changed.side, BLACK)
        self.assertEqual(changed.state, swap(state))
        self.assertEqual(canonical_key(changed.state, changed.side), canonical_key(state, WHITE))
        np.testing.assert_array_equal(changed.rgb.reshape(3, 25)[:, 8], sample.rgb.reshape(3, 25)[:, 8])
        np.testing.assert_array_equal(changed.target, sample.target)
        np.testing.assert_array_equal(changed.legal_mask, sample.legal_mask)
        self.assertFalse(np.array_equal(changed.rgb.reshape(3, 25)[:, 12], sample.rgb.reshape(3, 25)[:, 12]))

    def test_stages_have_valid_unique_expert_labels_and_no_terminal_samples(self):
        used = set()
        for stage in (1, 2, 3):
            rows = build_curriculum_dataset(stage, 96, seed=111 + stage, exclude_keys=used)
            self.assertEqual(len(rows), 96)
            keys = {row["key"] for row in rows}
            self.assertEqual(len(keys), len(rows))
            self.assertFalse(used & keys)
            used.update(keys)
            report = dataset_report(rows)
            self.assertEqual(sum(report["side"].values()), 96)
            for row in rows:
                state, side, target = row["state"], row["side"], row["target"]
                self.assertEqual(winner(state), 0)
                self.assertTrue(legal_moves(state))
                self.assertEqual(player_at(state), side)
                self.assertEqual(row["key"], base_key(state))
                self.assertEqual(target.shape, (25,))
                self.assertEqual(target.dtype, np.float32)
                self.assertTrue(np.isfinite(target).all())
                self.assertTrue((target >= 0).all())
                self.assertAlmostEqual(float(target.sum()), 1, places=6)
                self.assertTrue(all(i in legal_moves(state) for i in np.flatnonzero(target)))
                if row["value_kind"] == "exact":
                    self.assertIn(row["value"], (-1, 0, 1))
                    self.assertIn(row["outcome"], ("win", "draw", "loss"))
                    if empty_count(state) <= 5:
                        value, optimal = ExactSolver().analyze(state)
                        self.assertEqual(row["value"], value)
                        self.assertTrue(set(np.flatnonzero(target)) <= set(optimal))
                else:
                    self.assertEqual(row["outcome"], "unknown")
                    self.assertLess(abs(row["value"]), 1)
            if stage == 1:
                self.assertEqual(set(report["mask_count"]), {"0"})
                self.assertIn("loss", report["outcome"])
            elif stage == 2:
                self.assertTrue(set(report["mask_count"]) <= {"5", "10"})
            else:
                self.assertEqual(set(report["mask_count"]), {str(i) for i in range(1, 25)})
                self.assertIn("loss", report["outcome"])

    def test_plain_tactical_templates_cover_each_line_and_each_color(self):
        rng = random.Random(807)
        for mode in ("win", "block"):
            for ordinal in range(24):
                side = BLACK if ordinal % 2 == 0 else WHITE
                position = None
                for _ in range(100):
                    position = _tactical_position(1, rng, ordinal, mode, side)
                    if position is not None:
                        break
                self.assertIsNotNone(position, (mode, ordinal))
                state, actual_side, _ = position
                self.assertEqual(actual_side, side)
                line = LINE_INDICES[ordinal // 2]
                attack_side = side if mode == "win" else 3 - side
                cells = [(state >> (2 * index)) & 3 for index in line]
                self.assertEqual(cells.count(attack_side), 4)
                self.assertEqual(cells.count(0), 1)

    def test_generation_is_reproducible_and_respects_exclusions(self):
        first = build_curriculum_dataset(1, 32, seed=37)
        replay = build_curriculum_dataset(1, 32, seed=37)
        for a, b in zip(first, replay):
            self.assertEqual((a["state"], a["side"], a["source"]), (b["state"], b["side"], b["source"]))
            np.testing.assert_array_equal(a["target"], b["target"])
        first_keys = {row["key"] for row in first}
        fresh = build_curriculum_dataset(1, 32, seed=37, exclude_keys=first_keys)
        self.assertFalse(first_keys & {row["key"] for row in fresh})

    def test_invalid_requests_fail_before_generating_data(self):
        for stage in (0, 4, True, "missing"):
            with self.assertRaises(ValueError):
                build_curriculum_dataset(stage, 1)
        for count in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                build_curriculum_dataset(1, count)
        self.assertEqual(build_curriculum_dataset("plain", 0), [])



class TeacherTemperatureTests(unittest.TestCase):
    @staticmethod
    def entropy(target):
        positive = target[target > 0].astype(np.float64)
        return float(-(positive * np.log(positive)).sum())

    def test_lower_temperature_preserves_best_moves_and_reduces_quiet_entropy(self):
        state = sum(BLACK << (2 * i) for i in (1, 2, 3)) | sum(WHITE << (2 * i) for i in (10, 17, 24))
        searcher = PlayerAB(None, "temperature-test", depth=2)
        for side in (BLACK, WHITE):
            original, value, kind = _teach(state, side, searcher, teacher_temperature=0.15)
            sharper, new_value, new_kind = _teach(state, side, searcher)
            self.assertEqual((value, kind), (new_value, new_kind))
            self.assertEqual(kind, "heuristic")
            np.testing.assert_array_equal(np.flatnonzero(original == original.max()),
                                          np.flatnonzero(sharper == sharper.max()))
            self.assertLess(self.entropy(sharper), self.entropy(original))
            self.assertAlmostEqual(float(sharper.sum()), 1, places=6)
            self.assertTrue(set(np.flatnonzero(sharper)) <= set(legal_moves(state)))

    def test_truly_equal_scores_keep_their_uniform_distribution(self):
        free = {0, 1, 5, 6, 10, 11}
        state = sum(FORBIDDEN << (2 * i) for i in range(25) if i not in free)
        searcher = PlayerAB(None, "temperature-test", depth=2)
        for side in (BLACK, WHITE):
            targets = [_teach(state, side, searcher, temperature)[0]
                       for temperature in (0.15, 0.025, 0.015)]
            for target in targets:
                np.testing.assert_array_equal(target, targets[0])
                np.testing.assert_allclose(target[sorted(free)], np.full(6, 1 / 6), rtol=1e-6)

    def test_temperature_does_not_change_exact_or_forced_labels(self):
        win = sum(BLACK << (2 * i) for i in (0, 1, 2, 3))
        draw = sum(FORBIDDEN << (2 * i) for i in range(25) if i not in (0, 1, 5, 6))
        searcher = PlayerAB(None, "temperature-test", depth=2)
        for state in (win, draw):
            for side in (BLACK, WHITE):
                original = _teach(state, side, searcher, 0.15)
                changed = _teach(state, side, searcher, 0.025)
                np.testing.assert_array_equal(original[0], changed[0])
                self.assertEqual(original[1:], changed[1:])

    def test_invalid_temperatures_fail_even_for_an_empty_dataset(self):
        for temperature in (0, -1, float("nan"), float("inf"), -float("inf"), True, "0.025", None):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                build_curriculum_dataset(1, 0, teacher_temperature=temperature)
            with self.assertRaises(ValueError):
                _teach(0, BLACK, None, teacher_temperature=temperature)

    def test_build_passes_temperature_to_both_actor_labels(self):
        state = BLACK << (2 * 12)
        teacher_result = (np.zeros(25, dtype=np.float32), 0, "heuristic")
        with patch("unet_curriculum._tactical_position", return_value=(state, BLACK, "tactical_win")), \
             patch("unet_curriculum._pattern_position", return_value=(state, BLACK, "pattern", "p_00111", "edge_three_local", 0)), \
             patch("unet_curriculum._teach", return_value=teacher_result) as teach:
            build_curriculum_dataset(1, 1, teacher_temperature=0.015)
        self.assertEqual(len(teach.call_args_list), 2)
        self.assertEqual([call.args[1] for call in teach.call_args_list], [BLACK, WHITE])
        self.assertTrue(all(call.args[3] == 0.015 for call in teach.call_args_list))


if __name__ == "__main__":
    unittest.main()
