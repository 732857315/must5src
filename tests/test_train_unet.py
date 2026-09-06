"""Data splitting/augmentation and evaluation tests; fit/backward are never called."""

import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from game import BLACK, WHITE, FORBIDDEN, legal_moves, winner
from unet_codec import normalize_grid, encode_rgb, policy_rgb
from unet_data import canonical_key
from unet_curriculum import base_key
import train_unet as trainer


STATE = (BLACK << 24) | (WHITE << 2) | (FORBIDDEN << 8) | (BLACK << 18)


def pack(cells):
    return sum(int(cell) << (2 * i) for i, cell in enumerate(np.asarray(cells).reshape(-1)))


def record(state=STATE, side=BLACK, target=None, source="test", value=0):
    if target is None:
        target = np.zeros(25, dtype=np.float32)
        target[legal_moves(state)] = 1 / len(legal_moves(state))
    return {"state": state, "side": side, "target": np.asarray(target, dtype=np.float32),
            "stage": 1, "source": source, "value": value, "value_kind": "exact",
            "outcome": "win" if value == 1 else "loss" if value == -1 else "draw",
            "pattern": "test_pattern"}


def transform(cells, symmetry):
    result = np.rot90(cells, symmetry % 4)
    return np.fliplr(result) if symmetry >= 4 else result


def equivalent_records(original):
    cells = normalize_grid(original["state"])
    target = original["target"].reshape(5, 5)
    for symmetry in range(8):
        moved = transform(cells, symmetry)
        moved_target = transform(target, symmetry).copy().reshape(25)
        for swap in (False, True):
            colored = np.where((moved == 1) | (moved == 2), 3 - moved, moved) if swap else moved
            yield record(pack(colored), 3 - original["side"] if swap else original["side"], moved_target)


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, rgb, side):
        self.calls.append((rgb.clone(), side.clone()))
        return torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5).repeat(len(rgb), 1, 1, 1) / 10


class TrainingDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_split_keeps_d4_color_side_equivalence_groups_together_and_stable(self):
        rng = random.Random(874)
        bases = []
        while len(bases) < 32:
            cells = [rng.choices((0, 1, 2, 3), (8, 2, 2, 1))[0] for _ in range(25)]
            state = pack(cells)
            if not winner(state) and legal_moves(state):
                bases.append(record(state, rng.choice((BLACK, WHITE))))
        records = [item for base in bases for item in equivalent_records(base)]
        # Both possible actors on the identical physical board must stay in
        # the same partition, including every transformed/color-swapped copy.
        records += [{**item, "side": 3 - item["side"]} for item in records]
        train, validation = trainer.split_records(records)
        train_boards = {base_key(item["state"]) for item in train}
        validation_boards = {base_key(item["state"]) for item in validation}
        self.assertTrue(train_boards.isdisjoint(validation_boards))
        train_keys = {canonical_key(item["state"], item["side"]) for item in train}
        validation_keys = {canonical_key(item["state"], item["side"]) for item in validation}
        self.assertTrue(train_keys)
        self.assertTrue(validation_keys)
        self.assertTrue(train_keys.isdisjoint(validation_keys))
        rng.shuffle(records)
        reordered_train, reordered_validation = trainer.split_records(records)
        self.assertEqual(train_keys, {canonical_key(item["state"], item["side"]) for item in reordered_train})
        self.assertEqual(validation_keys, {canonical_key(item["state"], item["side"]) for item in reordered_validation})
        for base in bases:
            orbit_keys = {canonical_key(item["state"], item["side"]) for item in equivalent_records(base)}
            self.assertEqual(len(orbit_keys), 1)
            self.assertTrue(orbit_keys <= train_keys or orbit_keys <= validation_keys)

    def test_single_equivalence_group_cannot_be_reported_as_independent_validation(self):
        with self.assertRaisesRegex(ValueError, "independent validation"):
            trainer.split_records(list(equivalent_records(record())))

    def test_actor_expansion_uses_independent_counter_labels_and_preserves_board(self):
        own_target = np.zeros(25, dtype=np.float32)
        own_target[8] = 1
        counter_target = np.zeros(25, dtype=np.float32)
        counter_target[10] = 1
        bases = []
        for side in (BLACK, WHITE):
            base = record(side=side, target=own_target, value=1)
            base.update(counter_target=counter_target.copy(), counter_value=-0.37,
                        counter_value_kind="heuristic", counter_outcome="unknown")
            bases.append(base)
        expanded = trainer.expand_actor_records(bases)
        self.assertEqual(len(expanded), 4)
        for index, base in enumerate(bases):
            own, counter = expanded[2 * index:2 * index + 2]
            self.assertEqual(own["side"], base["side"])
            self.assertEqual(counter["side"], 3 - base["side"])
            self.assertEqual(own["state"], base["state"])
            self.assertEqual(counter["state"], base["state"])
            np.testing.assert_array_equal(own["target"], own_target)
            np.testing.assert_array_equal(counter["target"], counter_target)
            self.assertEqual(own["value"], 1)
            self.assertEqual(counter["value"], -0.37)
            self.assertEqual(counter["value_kind"], "heuristic")
            self.assertEqual(counter["outcome"], "unknown")
            self.assertTrue(counter["counterfactual_actor"])
            self.assertEqual(counter["stage"], base["stage"])
            self.assertEqual(counter["pattern"], base["pattern"])
            self.assertNotIn("counterfactual_actor", base)
            np.testing.assert_array_equal(base["target"], own_target)

    def test_all_augmentations_keep_state_side_target_and_continuous_red_aligned(self):
        target = np.arange(1, 26, dtype=np.float32)
        opponent = np.arange(25, 0, -1, dtype=np.float32)
        occupied = [i for i in range(25) if i not in legal_moves(STATE)]
        target[occupied] = 0
        opponent[occupied] = 0
        target /= target.sum()
        opponent /= opponent.sum()
        records = [record(target=target)] * 16
        encoded = trainer.encode_records(records)
        encoded["opponent"] = torch.from_numpy(opponent).reshape(1, 1, 5, 5).repeat(16, 1, 1, 1)
        originals = {key: value.clone() for key, value in encoded.items()}
        symmetries = torch.arange(8).repeat_interleave(2)
        swap_draws = torch.tensor([0.9, 0.1] * 8)
        with patch.object(trainer.torch, "randint", return_value=symmetries), \
             patch.object(trainer.torch, "rand", return_value=swap_draws):
            rgb, sides, targets, states = trainer.augmented_batch(encoded, torch.arange(16), None)
        cells = normalize_grid(STATE)
        for index in range(16):
            symmetry, swap = index // 2, bool(index % 2)
            expected_cells = transform(cells, symmetry)
            if swap:
                expected_cells = np.where((expected_cells == BLACK) | (expected_cells == WHITE),
                                          3 - expected_cells, expected_cells)
            expected_state = pack(expected_cells)
            expected_target = transform(target.reshape(5, 5), symmetry)
            expected_opponent = transform(opponent.reshape(5, 5), symmetry)
            self.assertEqual(states[index].item(), expected_state)
            self.assertEqual(sides[index].item(), WHITE if swap else BLACK)
            np.testing.assert_array_equal(targets[index, 0].numpy(), expected_target)
            np.testing.assert_allclose(rgb[index].numpy(),
                                       policy_rgb(expected_state, expected_opponent, "red", levels=None), atol=1e-7)
            illegal = normalize_grid(expected_state) != 0
            self.assertTrue(np.all(targets[index, 0].numpy()[illegal] == 0))
        for key, original in originals.items():
            torch.testing.assert_close(encoded[key], original, rtol=0, atol=0)

    def test_unaugmented_batch_is_exact_and_needs_no_rng(self):
        records = [record(side=WHITE), record(side=BLACK)]
        encoded = trainer.encode_records(records)
        rgb, sides, targets, states = trainer.augmented_batch(encoded, torch.tensor([1, 0]), None, augment=False)
        np.testing.assert_array_equal(rgb[0].numpy(), encode_rgb(STATE))
        self.assertEqual(sides.tolist(), [BLACK, WHITE])
        self.assertEqual(states.tolist(), [STATE, STATE])
        torch.testing.assert_close(targets[0], encoded["targets"][1])

    def test_frozen_opponent_inputs_use_opposite_side_and_no_future_target(self):
        first_target = np.zeros(25, dtype=np.float32)
        first_target[8] = 1
        second_target = np.zeros(25, dtype=np.float32)
        second_target[10] = 1
        encoded = trainer.encode_records([record(target=first_target), record(target=second_target)])
        model = RecordingModel()
        trainer.attach_opponent_predictions(encoded, model, batch_size=1)
        self.assertFalse(model.training)
        self.assertEqual(len(model.calls), 2)
        for image, sides in model.calls:
            self.assertEqual(sides.tolist(), [WHITE])
            np.testing.assert_array_equal(image[0].numpy(), encode_rgb(STATE))
        torch.testing.assert_close(encoded["opponent"][0], encoded["opponent"][1], rtol=0, atol=0)
        self.assertFalse(encoded["opponent"].requires_grad)
        self.assertAlmostEqual(encoded["opponent"][0].sum().item(), 1.0, places=6)
        self.assertTrue(torch.all(encoded["opponent"][:, 0].reshape(2, 25)[:, [1, 4, 9, 12]] == 0))

    def test_evaluation_reports_consistent_source_side_and_outcome_counts(self):
        target = np.zeros(25, dtype=np.float32)
        target[24] = 1
        records = [record(side=BLACK, target=target, source="win", value=1),
                   record(side=WHITE, target=target, source="loss", value=-1)]
        metrics = trainer.evaluate(RecordingModel(), trainer.encode_records(records), records, batch_size=1)
        self.assertEqual(metrics["count"], 2)
        self.assertEqual(metrics["expert_top1"], 1)
        self.assertTrue(np.isfinite(metrics["cross_entropy"]))
        self.assertEqual(metrics["source_balanced_top1"], 1)
        for key in ("by_source", "by_side", "by_outcome", "by_stage", "by_pattern"):
            self.assertEqual(sum(item["count"] for item in metrics[key].values()), 2)
            self.assertEqual(sum(item["correct"] for item in metrics[key].values()), 2)


if __name__ == "__main__":
    unittest.main()
