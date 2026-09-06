"""Audit transformations and validation without running a training loop."""

import copy
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from global_data import canonical_board_key, physical_board_key, split_global_records
from global_model import CHANNEL_NAMES, GlobalBoardNet, global_loss
from train_global import (augment_record, batches, checkpoint_score, evaluate,
                          initialize_global_weights, main, policy_metrics, value_metrics)


def fixture(shape=(6, 7), side=1, target=(2, 5), game_id="game-0"):
    board = np.zeros(shape, dtype=np.uint8)
    board[0, 1], board[1, 4], board[3, 0], board[4, 3] = 1, 2, 3, 1
    opponent = np.zeros(shape, dtype=np.float32)
    local = np.zeros(shape, dtype=np.float32)
    policy = np.zeros(shape, dtype=np.float32)
    opponent[5, 0], local[target], policy[target] = 1, 1, 1
    return dict(board=board, side=side, opponent_policy=opponent, local_policy=local,
                target_policy=policy, value=-1.0 if side == 1 else 1.0,
                value_valid=True, policy_source="searched_move", game_id=game_id,
                physical_key=physical_board_key(board))


def mapped_cell(cell, shape, turns, reflect):
    """Coordinate arithmetic independent of NumPy's array transformation."""
    row, column = cell
    height, width = shape
    for _ in range(turns % 4):
        row, column = width - 1 - column, row
        height, width = width, height
    if reflect:
        column = width - 1 - column
    return row, column


class EchoLocal(torch.nn.Module):
    def __init__(self, uniform=False):
        super().__init__()
        self.uniform = uniform
        self.inputs_seen = []
        self.gradient_modes = []

    def forward(self, inputs):
        self.inputs_seen.append(inputs.clone())
        self.gradient_modes.append(torch.is_grad_enabled())
        logits = torch.zeros_like(inputs[:, 4:5]) if self.uniform else inputs[:, 4:5] * 20
        return logits, inputs[:, 6, 0, 0]


class ControlledRng:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)

    def shuffle(self, values):
        pass

    def integers(self, upper):
        return 0

    def random(self):
        return next(self.probabilities)


class GlobalAugmentationTests(unittest.TestCase):
    def test_every_symmetry_and_joint_color_change_aligns_labels(self):
        record = fixture()
        original = {key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in record.items()}
        actor_key = canonical_board_key(record["board"], record["side"])
        for turns in range(4):
            for reflect in (False, True):
                for swap in (False, True):
                    with self.subTest(turns=turns, reflect=reflect, swap=swap):
                        inputs, board, target, value, valid = augment_record(record, turns, reflect, swap)
                        side = 3 - record["side"] if swap else record["side"]
                        self.assertEqual(canonical_board_key(board, side), actor_key)
                        for row, column in np.ndindex(record["board"].shape):
                            cell = mapped_cell((row, column), record["board"].shape, turns, reflect)
                            stone = int(record["board"][row, column])
                            expected = 3 - stone if swap and stone in (1, 2) else stone
                            self.assertEqual(board[cell], expected)
                            for channel, name in ((3, "opponent_policy"), (4, "local_policy")):
                                self.assertEqual(inputs[channel][cell], record[name][row, column])
                            self.assertEqual(target[cell], record["target_policy"][row, column])
                        np.testing.assert_array_equal(inputs[5], board == 0)
                        self.assertTrue(np.all(inputs[6] == (-1 if side == 1 else 1)))
                        np.testing.assert_array_equal(inputs[7, :, 0],
                                                      np.linspace(-1, 1, board.shape[0], dtype=np.float32))
                        np.testing.assert_array_equal(inputs[8, 0, :],
                                                      np.linspace(-1, 1, board.shape[1], dtype=np.float32))
                        self.assertEqual(value, record["value"])
                        self.assertEqual(valid, record["value_valid"])
                        self.assertEqual(float(target[board != 0].sum()), 0)
        for key in ("board", "opponent_policy", "local_policy", "target_policy"):
            np.testing.assert_array_equal(record[key], original[key])

    def test_drop_auxiliary_removes_only_the_two_maps(self):
        record = fixture()
        full = augment_record(record, 1, True, True)
        dropped = augment_record(record, 1, True, True, drop_auxiliary=True)
        self.assertTrue(np.all(dropped[0][3:5] == 0))
        np.testing.assert_array_equal(dropped[0][[0, 1, 2, 5, 6, 7, 8]],
                                      full[0][[0, 1, 2, 5, 6, 7, 8]])
        for original, changed in zip(full[1:3], dropped[1:3]):
            np.testing.assert_array_equal(original, changed)
        self.assertEqual(full[3:], dropped[3:])

    def test_thirty_percent_dropout_uses_one_draw_for_both_maps(self):
        rows = [fixture() for _ in range(4)]
        packed, _ = next(batches(rows, 4, ControlledRng([0.0, 0.299, 0.3, 0.9]),
                                auxiliary_dropout=0.3))
        inputs = packed[0].numpy()
        np.testing.assert_array_equal(inputs[:2, 3:5], 0)
        np.testing.assert_array_equal(inputs[2:, 3:5].sum(axis=(2, 3)), 1)
        self.assertTrue(np.all(packed[2].sum(dim=(1, 2)).numpy() == 1))

    def test_mixed_rectangles_keep_batch_shapes_and_labels_legal(self):
        rows = [fixture(shape) for shape in ((6, 7), (7, 6), (6, 6)) for _ in range(3)]
        for seed in (1, 2, 3):
            count = 0
            for (inputs, boards, target, values, valid), group in batches(
                    rows, 2, np.random.default_rng(seed), auxiliary_dropout=0.3):
                count += len(group)
                self.assertEqual(tuple(inputs.shape), (len(group), 9, *boards.shape[-2:]))
                self.assertEqual(tuple(target.shape), tuple(boards.shape))
                self.assertTrue(np.all(target.numpy()[boards != 0] == 0))
                np.testing.assert_allclose(target.sum(dim=(1, 2)).numpy(), 1)
                self.assertEqual(values.dtype, torch.float32)
                self.assertEqual(valid.dtype, torch.bool)
            self.assertEqual(count, len(rows))


class GlobalValidationTests(unittest.TestCase):
    def test_zero_auxiliary_metric_cannot_reintroduce_local_policy(self):
        record = fixture()
        model = EchoLocal(uniform=True)
        ordinary = evaluate(model, [record], 1)
        self.assertEqual(ordinary["policy"]["top1"], 0)
        self.assertEqual(ordinary["combined_policy"]["top1"], 1)
        ablated = evaluate(model, [record], 1, zero_auxiliary=True)
        self.assertEqual(ablated["combined_policy"], ablated["policy"])
        self.assertEqual(ablated["combined_policy"]["top1"], 0)
        self.assertTrue(torch.all(model.inputs_seen[-1][:, 3:5] == 0))
        self.assertFalse(model.training)
        self.assertEqual(model.gradient_modes, [False, False])

    def test_evaluation_preserves_alignment_when_shape_buckets_reorder_rows(self):
        rows = [fixture((6, 7), 1, (2, 5)), fixture((6, 6), 2, (2, 4)),
                fixture((6, 7), 2, (4, 5))]
        model = EchoLocal()
        result = evaluate(model, rows, 2)
        self.assertEqual(result["policy"]["correct"], 3)
        self.assertEqual(result["value"]["total"], 3)
        self.assertEqual(result["value"]["mse"], 0)
        self.assertEqual(result["policy"]["per_size"]["6x7"]["total"], 2)
        self.assertEqual(result["policy"]["per_size"]["6x6"]["total"], 1)

    def test_policy_top1_requires_maximum_target_and_accepts_exact_ties(self):
        record = fixture()
        record["target_policy"].fill(0)
        record["target_policy"][2, 2:5] = [0.4, 0.3, 0.3]
        probability = np.zeros(record["board"].shape)
        probability[2, 3] = 1
        self.assertEqual(policy_metrics([probability], [record])["top1"], 0)
        record["target_policy"][2, 2:5] = [0.5, 0.5, 0.0]
        self.assertEqual(policy_metrics([probability], [record])["top1"], 1)

    def test_invalid_value_rows_remain_masked_after_augmentation(self):
        known, unknown = fixture(side=1), fixture(side=2)
        unknown["value_valid"] = False
        rows = [known, unknown]
        (inputs, boards, target, labels, masks), _ = next(
            batches(rows, 2, ControlledRng([0.9, 0.9]), auxiliary_dropout=0.3))
        logits = torch.zeros((2, 1, *boards.shape[-2:]), requires_grad=True)
        values = torch.tensor([0.25, -0.75], requires_grad=True)
        losses = global_loss(logits, values, target, labels, masks, boards,
                             value_weight=0.3, return_components=True)
        self.assertAlmostEqual(losses["value_loss"].item(), 1.25 ** 2)
        losses["loss"].backward()
        self.assertNotEqual(values.grad[0].item(), 0)
        self.assertEqual(values.grad[1].item(), 0)
        metrics = value_metrics([0.25, -0.75], rows)
        self.assertEqual(metrics["total"], 1)
        self.assertAlmostEqual(metrics["mse"], 1.25 ** 2)
        unknown["value"] = -1
        self.assertEqual(metrics, value_metrics([0.25, -0.75], rows))
        self.assertEqual(value_metrics([0.0], [unknown])["total"], 0)

    def test_game_split_precedes_augmentation_and_rejects_symmetry_leakage(self):
        first, second = fixture(game_id="a"), fixture(game_id="b")
        second["board"][5, 5] = 2
        second["physical_key"] = physical_board_key(second["board"])
        first_other_ply = fixture(game_id="a")
        first_other_ply["board"][2, 0] = 2
        first_other_ply["physical_key"] = physical_board_key(first_other_ply["board"])
        train, validation = split_global_records([first, second, first_other_ply], seed=4)
        self.assertTrue({r["game_id"] for r in train}.isdisjoint(r["game_id"] for r in validation))
        self.assertTrue({r["physical_key"] for r in train}.isdisjoint(r["physical_key"] for r in validation))
        augmented = augment_record(first, 1, True, True)[1]
        repeated = dict(second, board=augmented, physical_key=physical_board_key(augmented))
        with self.assertRaisesRegex(ValueError, "multiple game groups"):
            split_global_records([first, repeated], seed=4)



class GlobalInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.local_dir = self.root / "local"
        self.local_dir.mkdir()
        self.local_hashes = {}
        for name in ("opponent.pt", "play.pt"):
            path = self.local_dir / name
            path.write_bytes(("unchanged-" + name).encode())
            self.local_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.source_model = GlobalBoardNet(input_mode="relative_rgb")
        with torch.no_grad():
            for index, parameter in enumerate(self.source_model.parameters()):
                parameter.fill_((index + 1) / 1000)
        self.payload = dict(format="gomoku_global_v1", role="global", trained=True,
                            base_channels=self.source_model.base_channels,
                            token_dim=self.source_model.token_dim,
                            attention_heads=self.source_model.attention_heads,
                            input_mode=self.source_model.input_mode, input_channels=list(CHANNEL_NAMES),
                            state_dict={name: value.clone() for name, value in self.source_model.state_dict().items()},
                            parameters=sum(parameter.numel() for parameter in self.source_model.parameters()),
                            local_model_sha256=copy.deepcopy(self.local_hashes), best_epoch=96,
                            optimizer={"state": {"sentinel": "must not be restored"}},
                            scheduler={"last_epoch": 99})
        self.source = self.root / "source" / "global.pt"
        self.source.parent.mkdir()
        self.save()

    def save(self):
        torch.save(self.payload, self.source)

    def test_initializes_exact_weights_with_same_byte_hash_and_fresh_training_metadata(self):
        model = GlobalBoardNet(input_mode="relative_rgb")
        original = self.source.read_bytes()
        self.assertTrue(any(not torch.equal(model.state_dict()[name], value)
                            for name, value in self.source_model.state_dict().items()))
        metadata = initialize_global_weights(model, self.source, self.local_hashes)
        for name, value in self.source_model.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        self.assertEqual(metadata["sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(metadata["path"], str(self.source.resolve()))
        self.assertEqual(metadata["source_best_epoch"], 96)
        self.assertEqual(metadata["mode"], "weights_only")
        self.assertFalse(metadata["optimizer_restored"])
        self.assertFalse(metadata["scheduler_restored"])
        self.assertNotIn("optimizer", metadata)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_identical_local_weights_may_move_directory_but_cannot_swap_roles(self):
        moved = {str(self.root / "moved" / Path(path).name): checksum
                 for path, checksum in self.local_hashes.items()}
        model = GlobalBoardNet(input_mode="relative_rgb")
        metadata = initialize_global_weights(model, self.source, moved)
        self.assertEqual(set(metadata["local_model_role_sha256"]), {"opponent.pt", "play.pt"})
        swapped = dict(zip(moved, reversed(list(moved.values()))))
        with self.assertRaisesRegex(ValueError, "different local model"):
            initialize_global_weights(model, self.source, swapped)

    def test_all_metadata_must_match_before_any_destination_weight_changes(self):
        original_payload = copy.deepcopy(self.payload)
        cases = (("format", "other"), ("role", "play"), ("trained", False), ("trained", 1),
                 ("base_channels", True), ("base_channels", 16.0), ("base_channels", 8),
                 ("token_dim", 16), ("attention_heads", 2), ("input_mode", "absolute_rgb"),
                 ("input_channels", list(reversed(CHANNEL_NAMES))), ("input_channels", 9),
                 ("parameters", 1), ("parameters", True), ("best_epoch", 0), ("max_token_side", 4),
                 ("local_model_sha256", {}))
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self.payload = copy.deepcopy(original_payload)
                self.payload[field] = value
                self.save()
                model = GlobalBoardNet(input_mode="relative_rgb")
                before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
                with self.assertRaises(ValueError):
                    initialize_global_weights(model, self.source, self.local_hashes)
                for name, tensor in before.items():
                    self.assertTrue(torch.equal(model.state_dict()[name], tensor))

    def test_missing_channels_or_local_identity_is_not_silently_accepted(self):
        original = copy.deepcopy(self.payload)
        for field in ("input_channels", "local_model_sha256", "trained", "role", "format"):
            self.payload = copy.deepcopy(original)
            self.payload.pop(field)
            self.save()
            with self.subTest(field=field), self.assertRaises(ValueError):
                initialize_global_weights(GlobalBoardNet(input_mode="relative_rgb"), self.source, self.local_hashes)

    def test_corrupt_tensor_keys_shapes_dtypes_and_nonfinite_values_do_not_partially_load(self):
        original = copy.deepcopy(self.payload)
        for fault in ("missing", "extra", "shape", "dtype", "nan", "infinity", "not_tensor"):
            self.payload = copy.deepcopy(original)
            weights = self.payload["state_dict"]
            last = list(weights)[-1]
            if fault == "missing":
                weights.pop(last)
            elif fault == "extra":
                weights["unexpected"] = torch.zeros(1)
            elif fault == "shape":
                weights[last] = torch.zeros(17, dtype=weights[last].dtype)
            elif fault == "dtype":
                weights[last] = weights[last].double()
            elif fault == "not_tensor":
                weights[last] = [0]
            else:
                weights[last].fill_(float("nan") if fault == "nan" else float("inf"))
            self.save()
            model = GlobalBoardNet(input_mode="relative_rgb")
            before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                initialize_global_weights(model, self.source, self.local_hashes)
            for name, tensor in before.items():
                self.assertTrue(torch.equal(model.state_dict()[name], tensor))

    def test_local_hash_schema_cannot_omit_duplicate_or_rename_roles(self):
        original = copy.deepcopy(self.payload)
        checksum = next(iter(self.local_hashes.values()))
        for hashes in ({"opponent.pt": checksum},
                       {"opponent.pt": checksum, "other.pt": checksum},
                       {"one/opponent.pt": checksum, "two/opponent.pt": checksum},
                       {"opponent.pt": "not-a-hash", "play.pt": checksum}):
            self.payload = copy.deepcopy(original)
            self.payload["local_model_sha256"] = hashes
            self.save()
            with self.subTest(hashes=hashes), self.assertRaises(ValueError):
                initialize_global_weights(GlobalBoardNet(input_mode="relative_rgb"), self.source, self.local_hashes)

    def test_legacy_missing_input_mode_means_absolute_only(self):
        self.payload.pop("input_mode")
        self.save()
        initialized = GlobalBoardNet(input_mode="absolute_rgb")
        metadata = initialize_global_weights(initialized, self.source, self.local_hashes)
        self.assertEqual(metadata["input_mode"], "absolute_rgb")
        with self.assertRaisesRegex(ValueError, "input_mode"):
            initialize_global_weights(GlobalBoardNet(input_mode="relative_rgb"), self.source, self.local_hashes)

    def test_cli_validates_and_records_weight_initialization_before_data_generation(self):
        output = self.root / "new-run"
        arguments = ["--init-checkpoint", str(self.source), "--output-dir", str(output),
                     "--local-models", str(self.local_dir), "--input-mode", "relative_rgb",
                     "--value-weight", "0", "--threads", "1"]
        source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        with patch("train_global.build_global_dataset", side_effect=RuntimeError("stop before data generation")) as generate:
            with self.assertRaisesRegex(RuntimeError, "stop before data generation"):
                main(arguments)
        generate.assert_called_once()
        config = json.loads((output / "config.json").read_text())
        self.assertEqual(config["initialization"]["sha256"], source_hash)
        self.assertEqual(config["checkpoint_selection"], "real_game_policy_else_informative_constructed_policy_only")
        self.assertFalse(config["initialization"]["optimizer_restored"])
        self.assertFalse((output / "global.pt").exists())
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), source_hash)

    def test_cli_rejects_bad_identity_before_creating_output_or_generating_data(self):
        self.payload["local_model_sha256"][next(iter(self.local_hashes))] = "0" * 64
        self.save()
        output = self.root / "new-run"
        with patch("train_global.build_global_dataset") as generate:
            with self.assertRaisesRegex(ValueError, "different local model"):
                main(["--init-checkpoint", str(self.source), "--output-dir", str(output),
                      "--local-models", str(self.local_dir), "--input-mode", "relative_rgb", "--threads", "1"])
        generate.assert_not_called()
        self.assertFalse(output.exists())

    def test_cli_cannot_overwrite_initial_checkpoint_or_resume_optimizer_in_place(self):
        original = self.source.read_bytes()
        for output, extra in ((self.source.parent, []), (self.root / "reuse", ["--reuse-data"])):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--init-checkpoint", str(self.source), "--output-dir", str(output), *extra])
        existing = self.root / "existing-run"
        existing.mkdir()
        (existing / "config.json").write_text("preserve")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--init-checkpoint", str(self.source), "--output-dir", str(existing)])
        self.assertEqual((existing / "config.json").read_text(), "preserve")
        self.assertEqual(self.source.read_bytes(), original)


class GlobalCheckpointSelectionTests(unittest.TestCase):
    @staticmethod
    def metrics(top1=.5, mse=1.0, real_total=12, informative=.4):
        return dict(real_games={"combined_policy": dict(total=real_total, top1=top1)},
                    constructed={"informative_combined_policy": dict(total=3, top1=informative),
                                 "combined_policy": dict(total=100, top1=1.0)},
                    value={"mse": mse})

    def test_positive_value_weight_preserves_original_selection_rule(self):
        metrics = self.metrics()
        self.assertAlmostEqual(checkpoint_score(metrics), .48)
        for weight in (.001, .3, 1.0):
            self.assertAlmostEqual(checkpoint_score(metrics, value_weight=weight), .48)

    def test_zero_value_weight_does_not_read_or_penalize_untrained_value_metrics(self):
        metrics = self.metrics(mse=4)
        self.assertEqual(checkpoint_score(metrics, value_weight=0), .5)
        metrics.pop("value")
        self.assertEqual(checkpoint_score(metrics, value_weight=0), .5)
        better_policy = self.metrics(top1=.6, mse=4)
        worse_policy = self.metrics(top1=.59, mse=0)
        self.assertGreater(checkpoint_score(better_policy, value_weight=0),
                           checkpoint_score(worse_policy, value_weight=0))

    def test_policy_only_fallback_still_excludes_uniform_proved_loss_samples(self):
        metrics = self.metrics(real_total=0)
        self.assertEqual(checkpoint_score(metrics, value_weight=0), .4)
        self.assertAlmostEqual(checkpoint_score(metrics), .38)

    def test_value_weight_must_be_finite_nonnegative_and_not_boolean(self):
        for value in (True, -1, float("nan"), float("inf"), "0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                checkpoint_score(self.metrics(), value_weight=value)


if __name__ == "__main__":
    unittest.main()
