"""Check ONNX/ncnn agreement without modifying model files."""

import argparse

import numpy as np

from inference import load_net, ncnn_infer, validate_input, validate_outputs


# Existing ncnn files have FP16 weights. New FP32 exports use tighter tolerances.
TOLERANCES = {"fp16": (0.05, 0.01), "fp32": (0.0001, 0.00001)}
INPUT_MODES = ("mixed", "uniform", "discrete")


def verification_inputs(*, samples=20, channels=2, input_max=1.0, seed=0, input_mode="uniform"):
    """Generate reproducible raw inputs and valid discrete local windows.

    Single-channel discrete inputs include corner/edge padding and blocked cells
    as 3. The reference two-channel format only supports ordinary 0/1/2 boards.
    Uniform mode retains the original floating-point verification sequence.
    """
    if samples < 1 or channels not in (1, 2):
        raise ValueError("samples must be positive and channels must be 1 or 2")
    if not np.isfinite(input_max) or input_max <= 0:
        raise ValueError("input_max must be finite and positive")
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}")
    rng = np.random.default_rng(seed)
    discrete_index = 0
    for index in range(samples):
        discrete = input_mode == "discrete" or (input_mode == "mixed" and index % 2 == 0)
        if not discrete:
            yield rng.uniform(0, input_max, (1, channels, 5, 5)).astype(np.float32)
            continue
        cells = rng.integers(0, 3, (5, 5))
        if channels == 1:
            kind = discrete_index % 3
            if kind == 0:
                cells[:2, :] = 3
                cells[:, :2] = 3
            elif kind == 1:
                cells[:2, :] = 3
            else:
                cells[1, 1] = 3
                cells[3, 3] = 3
            cells[2, 2:5] = (0, 1, 2)
            cells = np.rot90(cells, (discrete_index // 3) % 4)
            yield np.ascontiguousarray(cells, dtype=np.float32)[None, None]
        else:
            yield np.stack((cells == 1, cells == 2)).astype(np.float32)[None]
        discrete_index += 1


def onnx_infer(ort_sess, x):
    x = validate_input(x)
    inputs = ort_sess.get_inputs()
    if len(inputs) != 1:
        raise ValueError("expected exactly one ONNX input")
    out = ort_sess.run(None, {inputs[0].name: x})
    if len(out) != 2:
        raise ValueError("expected policy and value ONNX outputs")
    return validate_outputs(*out)


def compare_outputs(reference, actual, policy_atol, value_atol):
    """Raise on shape, non-finite, or absolute-error tolerance violations."""
    tolerances = (policy_atol, value_atol)
    if any(not np.isfinite(t) or t < 0 for t in tolerances):
        raise ValueError("tolerances must be finite and nonnegative")
    reference = validate_outputs(*reference)
    actual = validate_outputs(*actual)
    errors = tuple(float(np.abs(a - b).max()) for a, b in zip(reference, actual))
    for name, error, tolerance in zip(("policy", "value"), errors, tolerances):
        if error > tolerance:
            raise AssertionError(f"{name} max diff {error:.8g} exceeds atol {tolerance:.8g}")
    return errors


def verify_models(onnx_path, param, bin, *, samples=20, channels=2,
                  input_max=1.0, seed=0, policy_atol=0.05, value_atol=0.01,
                  input_mode="uniform"):
    import onnxruntime as ort

    inputs = verification_inputs(samples=samples, channels=channels, input_max=input_max,
                                 seed=seed, input_mode=input_mode)
    first_input = next(inputs)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    net = load_net(param, bin)
    maxp = maxv = 0.0
    from itertools import chain
    for x in chain((first_input,), inputs):
        dp, dv = compare_outputs(onnx_infer(sess, x), ncnn_infer(net, x), policy_atol, value_atol)
        maxp, maxv = max(maxp, dp), max(maxv, dv)
    return maxp, maxv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default=".repo/webdemo/model_bs15_win5.onnx")
    parser.add_argument("--param", default="repo5x5.param")
    parser.add_argument("--bin", default="repo5x5.bin")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--channels", type=int, choices=(1, 2), default=2)
    parser.add_argument("--input-max", type=float, default=1.0)
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="uniform",
                        help="mixed includes discrete boundary windows; uniform reproduces the legacy inputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=TOLERANCES, default="fp16",
                        help="fp16 allows legacy weight quantization; fp32 uses strict tolerances")
    parser.add_argument("--policy-atol", type=float)
    parser.add_argument("--value-atol", type=float)
    args = parser.parse_args(argv)
    policy_atol, value_atol = TOLERANCES[args.precision]
    if args.policy_atol is not None:
        policy_atol = args.policy_atol
    if args.value_atol is not None:
        value_atol = args.value_atol
    maxp, maxv = verify_models(args.onnx, args.param, args.bin,
                             samples=args.samples, channels=args.channels, input_max=args.input_max,
                             seed=args.seed, policy_atol=policy_atol, value_atol=value_atol,
                             input_mode=args.input_mode)
    print(f"PASS ({args.precision}, {args.input_mode}, {args.samples} samples): policy max diff={maxp:.6f} "
          f"(atol={policy_atol:g}), value max diff={maxv:.6f} (atol={value_atol:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
