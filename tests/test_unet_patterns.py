from functools import lru_cache
import itertools
import random
import unittest

import numpy as np

from game import BLACK, WHITE, LINE_INDICES, legal_moves, player_at, winner
from unet_curriculum import (_exact_analysis, _pattern_position, build_curriculum_dataset,
                             feasible_pattern_catalogue, base_key)
from unet_patterns import catalogue_report, pattern_catalogue


@lru_cache(None)
def independent_value(cells, side):
    lines = [cells[r * 5:r * 5 + 5] for r in range(5)]
    lines += [cells[c::5] for c in range(5)]
    lines += [tuple(cells[i * 6] for i in range(5)), tuple(cells[4 + i * 4] for i in range(5))]
    for line in lines:
        if line[0] in (1, 2) and len(set(line)) == 1:
            return 1 if line[0] == side else -1
    moves = [i for i, cell in enumerate(cells) if cell == 0]
    if not moves:
        return 0
    return max(-independent_value(cells[:i] + (side,) + cells[i + 1:], 3 - side) for i in moves)


class PatternTests(unittest.TestCase):
    def test_catalogue_covers_all_1024_encodings_modulo_its_stated_equivalence(self):
        expected = set()
        for raw in itertools.product(range(4), repeat=5):
            if max(raw.count(1), raw.count(2)) < 3:
                continue
            main = 1 if raw.count(1) >= 3 else 2
            normalized = tuple(1 if x == main else 2 if x in (1, 2) else x for x in raw)
            expected.add(min(normalized, normalized[::-1]))
        rows = pattern_catalogue()
        self.assertEqual({tuple(row["cells"]) for row in rows}, expected)
        report = catalogue_report()
        self.assertEqual(report["enumerated_line_encodings"], 1024)
        self.assertEqual(report["retained_patterns"], 58)
        self.assertEqual(len(report["terminal_archive"]), 1)
        self.assertFalse(report["terminal_archive"][0]["policy_eligible"])
        self.assertEqual(len(report["patterns"]), 57)

    def test_local_open_three_and_blocked_four_have_distinct_semantics(self):
        rows = {tuple(row["cells"]): row for row in pattern_catalogue()}
        self.assertEqual(rows[(0, 1, 1, 1, 0)]["kind"], "open_three_local")
        self.assertEqual(rows[(1, 1, 0, 1, 1)]["subtype"], "internal_gap_four")
        for blocked in ((1, 1, 1, 1, 2), (1, 1, 1, 1, 3)):
            self.assertEqual(rows[blocked]["kind"], "blocked_four")
            self.assertFalse(rows[blocked]["can_complete_five_in_this_window"])

    def test_all_geometrically_feasible_motifs_can_be_embedded_without_terminal_targets(self):
        rng = random.Random(894)
        for stage, expected_count in ((1, 28), (2, 38), (3, 57)):
            patterns = feasible_pattern_catalogue(stage)
            self.assertEqual(len(patterns), expected_count)
            for ordinal, pattern in enumerate(patterns):
                side = BLACK if ordinal % 2 == 0 else WHITE
                candidate = _pattern_position(stage, rng, pattern, ordinal, side)
                self.assertIsNotNone(candidate, (stage, pattern["id"]))
                state, actor, source, pattern_id, kind, line_index = candidate
                self.assertEqual((actor, source, pattern_id, kind),
                                 (side, "pattern", pattern["id"], pattern["kind"]))
                self.assertEqual(winner(state), 0)
                self.assertTrue(legal_moves(state))
                self.assertEqual(player_at(state), side)
                cells = tuple((state >> (2 * i)) & 3 for i in LINE_INDICES[line_index])
                main = BLACK if ordinal % 2 == 0 else WHITE
                normalized = tuple(1 if x == main else 2 if x in (1, 2) else x for x in cells)
                self.assertEqual(list(normalized), pattern["cells"])

    def test_counter_actor_is_recomputed_and_matches_independent_endgame_values(self):
        rows = build_curriculum_dataset(3, 80, seed=939)
        checked = 0
        changed = 0
        for row in rows:
            state, side = row["state"], row["side"]
            self.assertEqual(row["key"], base_key(state))
            target = row["counter_target"]
            self.assertAlmostEqual(float(target.sum()), 1, places=6)
            self.assertTrue(set(np.flatnonzero(target)) <= set(legal_moves(state)))
            changed += not np.array_equal(row["target"], target)
            if len(legal_moves(state)) <= 5:
                cells = tuple((state >> (2 * i)) & 3 for i in range(25))
                expected = independent_value(cells, 3 - side)
                value, optimal = _exact_analysis(state, 3 - side)
                self.assertEqual(value, expected)
                self.assertEqual(row["counter_value"], expected)
                self.assertEqual(row["counter_value_kind"], "exact")
                self.assertTrue(set(np.flatnonzero(target)) <= set(optimal))
                checked += 1
        self.assertGreater(checked, 10)
        self.assertGreater(changed, 0)

    def test_explicit_endgame_oracle_rejects_large_searches_and_has_no_terminal_moves(self):
        with self.assertRaises(ValueError):
            _exact_analysis(0, BLACK)
        cells = [3] * 25
        cells[:5] = [BLACK] * 5
        state = sum(cell << (2 * i) for i, cell in enumerate(cells))
        self.assertEqual(_exact_analysis(state, WHITE), (-1, []))
        self.assertEqual(_exact_analysis(state, BLACK), (1, []))


if __name__ == "__main__":
    unittest.main()
