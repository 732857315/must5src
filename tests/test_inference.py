import importlib
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import export_mine
import inference
import verify_repo
from game import BLACK, WHITE, FORBIDDEN


class EncodingTests(unittest.TestCase):
    def test_color_swap_and_two_channel_layout(self):
        state = BLACK | (WHITE << 2) | (BLACK << 48)
        black = inference.encode_state(state, BLACK)
        white = inference.encode_state(state, WHITE)
        np.testing.assert_array_equal(black.reshape(-1)[[0, 1, 2, 24]], [1, 2, 0, 1])
        np.testing.assert_array_equal(white.reshape(-1)[[0, 1, 2, 24]], [2, 1, 0, 2])
        two = inference.encode_state(state, WHITE, "two")
        np.testing.assert_array_equal(two[0, 0], (white[0, 0] == 1).astype(np.float32))
        np.testing.assert_array_equal(two[0, 1], (white[0, 0] == 2).astype(np.float32))

    def test_rejects_corrupt_state_and_unknown_encoding(self):
        for state in (-1, 1 << 50):
            with self.subTest(state=state), self.assertRaises(ValueError):
                inference.encode_state(state, BLACK)
        with self.assertRaises(ValueError):
            inference.encode_state(0, 3)
        with self.assertRaises(ValueError):
            inference.encode_state(0, BLACK, "unknown")

    def test_boundary_cells_are_unchanged_when_player_perspective_swaps(self):
        state = FORBIDDEN | (BLACK << 2) | (WHITE << 4) | (FORBIDDEN << 48)
        black = inference.encode_state(state, BLACK)
        white = inference.encode_state(state, WHITE)
        self.assertEqual(black.shape, (1, 1, 5, 5))
        np.testing.assert_array_equal(black.reshape(-1)[[0, 1, 2, 3, 24]], [3, 1, 2, 0, 3])
        np.testing.assert_array_equal(white.reshape(-1)[[0, 1, 2, 3, 24]], [3, 2, 1, 0, 3])

    def test_reference_encoding_rejects_boundaries(self):
        for me in (BLACK, WHITE):
            with self.subTest(me=me), self.assertRaisesRegex(ValueError, "does not support forbidden"):
                inference.encode_state(FORBIDDEN << 48, me, "two")

    def test_mat_roundtrip_preserves_two_channels_and_owns_input(self):
        x = np.arange(50, dtype=np.float32).reshape(1, 2, 5, 5)
        expected = x[0].copy()
        mat = inference.np_to_mat(x)
        x.fill(-1)
        np.testing.assert_array_equal(np.array(mat), expected)

    def test_rejects_batch_bad_shape_and_nonfinite_inputs(self):
        for shape in ((2, 1, 5, 5), (1, 3, 5, 5), (1, 1, 15, 15), (25,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                inference.validate_input(np.zeros(shape))
        for bad in (np.nan, np.inf, -np.inf):
            x = np.zeros((1, 1, 5, 5))
            x[0, 0, 0, 0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                inference.validate_input(x)


class CheckedInferenceTests(unittest.TestCase):
    def setUp(self):
        self.x = np.zeros((1, 1, 5, 5), dtype=np.float32)
        self.extractor = Mock()
        self.extractor.input.return_value = 0
        self.extractor.extract.side_effect = [(0, np.arange(25, dtype=np.float32)), (0, np.array([0.5]))]
        self.net = Mock()
        self.net.create_extractor.return_value = self.extractor

    def test_output_lifetime_is_independent_of_extractor(self):
        policy = np.arange(25, dtype=np.float32)
        value = np.array([0.5], dtype=np.float32)
        self.extractor.extract.side_effect = [(0, policy), (0, value)]
        p, v = inference.ncnn_infer(self.net, self.x)
        policy.fill(99)
        value.fill(99)
        np.testing.assert_array_equal(p, np.arange(25))
        self.assertEqual(v[0], 0.5)

    def test_input_failure_stops_extraction(self):
        self.extractor.input.return_value = -1
        with self.assertRaisesRegex(RuntimeError, "input"):
            inference.ncnn_infer(self.net, self.x)
        self.extractor.extract.assert_not_called()

    def test_each_extract_failure_is_reported(self):
        for outputs in ([(-1, None)], [(0, np.zeros(25)), (-100, None)]):
            with self.subTest(outputs=outputs):
                self.extractor.extract.side_effect = outputs
                with self.assertRaisesRegex(RuntimeError, "extract"):
                    inference.ncnn_infer(self.net, self.x)

    def test_rejects_bad_or_nonfinite_output(self):
        cases = [(np.zeros(24), [0]), (np.zeros(25), [0, 1]),
                 (np.full(25, np.nan), [0]), (np.zeros(25), [np.inf])]
        for policy, value in cases:
            with self.subTest(policy=policy, value=value), self.assertRaises(ValueError):
                inference.validate_outputs(policy, value)

    def test_missing_files_fail_before_native_loading(self):
        with tempfile.TemporaryDirectory() as directory, patch("inference.ncnn.Net") as constructor:
            with self.assertRaises(FileNotFoundError):
                inference.load_net(Path(directory) / "missing.param", Path(directory) / "missing.bin")
            constructor.assert_not_called()

    def test_failed_model_load_return_codes_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            param, binary = Path(directory) / "model.param", Path(directory) / "model.bin"
            param.touch()
            binary.touch()
            for param_code, model_code, message in ((-1, 0, "load_param"), (0, -1, "load_model")):
                with self.subTest(message=message), patch("inference.ncnn.Net", return_value=self.net):
                    self.net.load_param.return_value = param_code
                    self.net.load_model.return_value = model_code
                    with self.assertRaisesRegex(RuntimeError, message):
                        inference.load_net(param, binary)

    def test_boundary_inference_remains_raw_until_action_masking(self):
        state = FORBIDDEN << 48
        raw_policy, raw_value = inference.ncnn_infer(self.net, inference.encode_state(state, BLACK))
        self.assertTrue(np.isfinite(raw_policy).all())
        self.assertEqual(raw_policy[24], 24)
        self.assertEqual(raw_value[0], 0.5)
        self.assertEqual(inference.choose_legal_move(raw_policy, state), 23)
        self.assertEqual(raw_policy[24], 24)


class PolicyMaskTests(unittest.TestCase):
    def test_stones_and_forbidden_cells_cannot_win_argmax(self):
        state = BLACK | (WHITE << 2) | (FORBIDDEN << 4)
        raw = np.arange(25, dtype=np.float32)
        raw[:3] = (1000, 2000, 3000)
        expected = raw.copy()
        masked = inference.legal_policy(raw, state)
        self.assertTrue(np.isneginf(masked[:3]).all())
        np.testing.assert_array_equal(masked[3:], raw[3:])
        np.testing.assert_array_equal(raw, expected)
        self.assertEqual(inference.choose_legal_move(raw, state), 24)

    def test_no_empty_cells_returns_none_instead_of_cell_zero(self):
        state = (1 << 50) - 1
        raw = np.zeros(25)
        self.assertTrue(np.isneginf(inference.legal_policy(raw, state)).all())
        self.assertIsNone(inference.choose_legal_move(raw, state))
        # A single empty cell still wins even if its original score is low.
        state &= ~(3 << 20)
        raw[10] = -10000
        self.assertEqual(inference.choose_legal_move(raw, state), 10)

    def test_mask_accepts_only_finite_raw_policy(self):
        for raw in (np.zeros(24), np.full(25, np.nan), np.full(25, -np.inf)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                inference.legal_policy(raw, 0)


class VerificationTests(unittest.TestCase):
    def test_discrete_windows_cover_padding_and_blocked_cells(self):
        samples = list(verify_repo.verification_inputs(samples=12, channels=1, input_mode="discrete"))
        for sample in samples:
            self.assertEqual(sample.shape, (1, 1, 5, 5))
            self.assertEqual(set(np.unique(sample)), {0, 1, 2, 3})
            self.assertTrue(sample.flags.c_contiguous)
        self.assertTrue((samples[0][0, 0, :2, :] == 3).all())
        self.assertTrue((samples[0][0, 0, :, :2] == 3).all())
        self.assertEqual(samples[2][0, 0, 1, 1], 3)
        self.assertEqual(samples[2][0, 0, 3, 3], 3)

    def test_reference_discrete_inputs_are_mutually_exclusive_planes(self):
        for x in verify_repo.verification_inputs(samples=5, channels=2, input_mode="discrete"):
            self.assertEqual(x.shape, (1, 2, 5, 5))
            self.assertTrue(np.isin(x, [0, 1]).all())
            self.assertTrue((x.sum(axis=1) <= 1).all())

    def test_uniform_mode_reproduces_original_random_inputs(self):
        rng = np.random.default_rng(0)
        for actual in verify_repo.verification_inputs(samples=5, channels=1, input_max=2, seed=0):
            expected = rng.uniform(0, 2, (1, 1, 5, 5)).astype(np.float32)
            np.testing.assert_array_equal(actual, expected)

    def test_command_defaults_to_legacy_uniform_inputs(self):
        with patch("verify_repo.verify_models", return_value=(0, 0)) as verify:
            with patch("builtins.print"):
                self.assertEqual(verify_repo.main([]), 0)
        self.assertEqual(verify.call_args.kwargs["input_mode"], "uniform")

    def test_mixed_mode_includes_boundaries_with_one_sample(self):
        x = next(verify_repo.verification_inputs(samples=1, channels=1, input_mode="mixed"))
        self.assertIn(3, np.unique(x))
        with self.assertRaises(ValueError):
            next(verify_repo.verification_inputs(input_mode="unknown"))

    def test_import_does_not_load_models(self):
        with patch("inference.load_net") as loader, patch("onnxruntime.InferenceSession") as session:
            importlib.reload(verify_repo)
            loader.assert_not_called()
            session.assert_not_called()
        # Restore direct imports after the patched reload.
        importlib.reload(verify_repo)

    def test_tolerance_is_an_actual_failure_condition(self):
        reference = (np.zeros(25), np.zeros(1))
        actual = (np.full(25, 0.02), np.full(1, 0.005))
        error = verify_repo.compare_outputs(reference, actual, *verify_repo.TOLERANCES["fp16"])
        self.assertAlmostEqual(error[0], 0.02)
        with self.assertRaisesRegex(AssertionError, "policy"):
            verify_repo.compare_outputs(reference, actual, *verify_repo.TOLERANCES["fp32"])
        with self.assertRaisesRegex(AssertionError, "value"):
            verify_repo.compare_outputs(reference, (np.zeros(25), np.array([0.1])), 0.05, 0.01)

    def test_nonfinite_outputs_and_tolerances_cannot_pass(self):
        valid = (np.zeros(25), np.zeros(1))
        with self.assertRaises(ValueError):
            verify_repo.compare_outputs(valid, (np.full(25, np.nan), np.zeros(1)), 1, 1)
        for tolerance in (np.nan, np.inf, -1):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                verify_repo.compare_outputs(valid, valid, tolerance, 0.01)

    def test_onnx_shapes_and_output_count_are_validated(self):
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name="input")]
        session.run.return_value = [np.zeros((1, 25)), np.zeros((1,))]
        p, v = verify_repo.onnx_infer(session, np.zeros((1, 2, 5, 5)))
        self.assertEqual((p.shape, v.shape), ((25,), (1,)))
        session.run.return_value = [np.zeros(25)]
        with self.assertRaisesRegex(ValueError, "outputs"):
            verify_repo.onnx_infer(session, np.zeros((1, 2, 5, 5)))


class ExportTests(unittest.TestCase):
    @staticmethod
    def fake_onnx_export(net, dummy, path, **kwargs):
        if tuple(dummy.shape) != (1, 1, 5, 5) or not bool((dummy == 3).any()):
            raise AssertionError("export example must remain 5x5 and include forbidden cells")
        Path(path).write_bytes(b"mock onnx")

    @staticmethod
    def fake_conversion(onnx_path, directory):
        (Path(directory) / "mine5x5.param").write_text("mock param", encoding="utf-8")
        (Path(directory) / "mine5x5.bin").write_bytes(b"mock binary")

    def test_convert_passes_fp32_and_checks_process_status(self):
        with tempfile.TemporaryDirectory(prefix="export path with spaces ") as directory:
            with patch("export_mine.subprocess.run") as run:
                export_mine.convert_ncnn(Path(directory) / "mine5x5.onnx", directory,
                                         executable="pnnx executable.exe")
            args, kwargs = run.call_args
            self.assertIn("fp16=0", args[0])
            self.assertIn("inputshape=[1,1,5,5]f32", args[0])
            self.assertIn("ncnnbin=mine5x5.bin", args[0])
            self.assertEqual(args[0][0], "pnnx executable.exe")
            self.assertEqual(kwargs["cwd"], Path(directory).resolve())
            self.assertTrue(kwargs["check"])

    def test_conversion_failure_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("export_mine.subprocess.run", side_effect=subprocess.CalledProcessError(1, "pnnx")):
                with self.assertRaises(subprocess.CalledProcessError):
                    export_mine.convert_ncnn(Path(directory) / "model.onnx", directory, executable="pnnx")

    def test_existing_artifacts_require_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model = output / "mine5x5.bin"
            model.write_bytes(b"original model")
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                export_mine.check_output_paths(output)
            self.assertEqual(export_mine.check_output_paths(output, overwrite=True), output.resolve())
            self.assertEqual(model.read_bytes(), b"original model")

    def test_directory_named_as_artifact_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "mine5x5.bin").mkdir()
            with self.assertRaises(ValueError):
                export_mine.check_output_paths(directory, overwrite=True)

    def test_failed_verification_preserves_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "weights.pt"
            checkpoint.write_bytes(b"checkpoint")
            output = root / "output"
            output.mkdir()
            binary = output / "mine5x5.bin"
            binary.write_bytes(b"original binary")
            with patch("az.GomokuNet5x5"), patch("torch.load", return_value={}):
                with patch("torch.onnx.export", side_effect=self.fake_onnx_export):
                    with patch("export_mine.convert_ncnn", side_effect=self.fake_conversion):
                        with patch("export_mine.verify_models", side_effect=AssertionError("precision failed")):
                            with self.assertRaisesRegex(AssertionError, "precision failed"):
                                export_mine.export_checkpoint(checkpoint, output, overwrite=True)
            self.assertEqual(binary.read_bytes(), b"original binary")
            self.assertFalse((output / "mine5x5.onnx").exists())
            self.assertEqual(checkpoint.read_bytes(), b"checkpoint")
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["output", "weights.pt"])

    def test_successful_export_publishes_only_after_strict_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "weights.pt"
            checkpoint.write_bytes(b"checkpoint")
            output = root / "output"

            def verified(*args, **kwargs):
                self.assertFalse(output.exists())
                self.assertEqual((kwargs["policy_atol"], kwargs["value_atol"]),
                                 verify_repo.TOLERANCES["fp32"])
                self.assertEqual(kwargs["input_mode"], "mixed")
                return 1e-6, 1e-7

            with patch("az.GomokuNet5x5"), patch("torch.load", return_value={}):
                with patch("torch.onnx.export", side_effect=self.fake_onnx_export):
                    with patch("export_mine.convert_ncnn", side_effect=self.fake_conversion):
                        with patch("export_mine.verify_models", side_effect=verified):
                            published, maxp, maxv = export_mine.export_checkpoint(checkpoint, output)
            self.assertEqual(published, output.resolve())
            self.assertEqual((maxp, maxv), (1e-6, 1e-7))
            self.assertEqual((output / "mine5x5.bin").read_bytes(), b"mock binary")
            self.assertEqual(checkpoint.read_bytes(), b"checkpoint")


if __name__ == "__main__":
    unittest.main()
