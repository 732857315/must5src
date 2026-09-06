"""Compare checkpoint architecture and tensors, ignoring run-specific metadata.

Exit 0 means every tensor meets the stated tolerance and architecture matches;
exit 1 means a mismatch, and exit 2 means invalid input. File SHA256 values are
recorded for identity only. They do not determine numeric equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
import sys

import torch


ARCHITECTURE_FIELDS = (
    "format", "role", "base_channels", "token_dim", "attention_heads",
    "input_mode", "input_channels", "max_token_side",
)


def load_checkpoint(path):
    path = Path(path).resolve()
    contents = path.read_bytes()
    # Load exactly the bytes whose identity is reported; do not re-open a file
    # that could have changed between hashing and loading.
    import io
    payload = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Checkpoint must contain a state_dict mapping: " + str(path))
    weights = payload["state_dict"]
    if not weights or any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
                          for key, value in weights.items()):
        raise ValueError("state_dict must map names to tensors: " + str(path))
    if any(value.layout != torch.strided for value in weights.values()):
        raise ValueError("Only dense strided checkpoint tensors are supported")
    architecture = {name: payload[name] for name in ARCHITECTURE_FIELDS if name in payload}
    # Architecture is a JSON contract. Reject arbitrary objects before comparisons.
    json.dumps(architecture, allow_nan=False)
    return architecture, weights, {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest()}


def compare_checkpoints(expected, actual, *, atol=1e-6, rtol=1e-5):
    for name, value in (("atol", atol), ("rtol", rtol)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(name + " must be finite and nonnegative")
    expected_arch, reference, expected_id = load_checkpoint(expected)
    actual_arch, candidate, actual_id = load_checkpoint(actual)
    architecture_equal = (json.dumps(expected_arch, sort_keys=True, allow_nan=False)
                          == json.dumps(actual_arch, sort_keys=True, allow_nan=False))
    missing = sorted(set(reference) - set(candidate))
    extra = sorted(set(candidate) - set(reference))
    tensors = []
    for name in sorted(set(reference) & set(candidate)):
        left, right = reference[name], candidate[name]
        row = {"name": name, "expected_shape": list(left.shape), "actual_shape": list(right.shape),
               "expected_dtype": str(left.dtype), "actual_dtype": str(right.dtype), "passed": False}
        if left.shape != right.shape or left.dtype != right.dtype:
            row["reason"] = "shape or dtype differs"
        elif not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
            row["reason"] = "nonfinite tensor values"
        else:
            row["exact"] = torch.equal(left, right)
            if left.is_floating_point() or left.is_complex():
                dtype = torch.complex128 if left.is_complex() else torch.float64
                ref, got = left.to(dtype), right.to(dtype)
                close = torch.isclose(got, ref, atol=atol, rtol=rtol, equal_nan=False)
                row["mismatched_values"] = int((~close).sum())
                maximum = float((got - ref).abs().max()) if left.numel() else 0.0
                row["max_abs_error"] = maximum if math.isfinite(maximum) else None
                row["passed"] = bool(close.all())
            else:
                row["passed"] = row["exact"]
                row["mismatched_values"] = int((left != right).sum())
                row["comparison"] = "exact integer or boolean equality"
        tensors.append(row)
    passed = architecture_equal and not missing and not extra and all(row["passed"] for row in tensors)
    return {"format": "must5_checkpoint_comparison_v1", "passed": passed,
            "expected": expected_id, "actual": actual_id,
            "atol": atol, "rtol": rtol,
            "float_comparison": "abs(actual - expected) <= atol + rtol * abs(expected)",
            "architecture_equal": architecture_equal,
            "expected_architecture": expected_arch, "actual_architecture": actual_arch,
            "missing_tensors": missing, "extra_tensors": extra,
            "tensor_count": len(tensors),
            "exact_tensor_count": sum(row.get("exact", False) for row in tensors),
            "tensors": tensors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--report", type=Path, help="Write a JSON report to a new file")
    args = parser.parse_args(argv)
    if args.report and args.report.exists():
        parser.error("--report already exists; choose a new file to preserve earlier results")
    try:
        report = compare_checkpoints(args.expected, args.actual, atol=args.atol, rtol=args.rtol)
        contents = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(contents)
    except (OSError, ValueError, RuntimeError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        parser.error(str(exc))
    print(contents, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
