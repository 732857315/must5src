"""Export a checkpoint to verified FP32 ONNX/ncnn files in an output directory."""

import argparse
from pathlib import Path
import subprocess
import tempfile

from verify_repo import TOLERANCES, verification_inputs, verify_models


ARTIFACT_NAMES = (
    "mine5x5.onnx", "mine5x5.param", "mine5x5.bin", "mine5x5_ncnn.py",
    "mine5x5_pnnx.py", "mine5x5.pnnx.param", "mine5x5.pnnx.bin",
    "mine5x5.pnnx.onnx", "mine5x5.pnnxsim.onnx",
)


def check_output_paths(output_dir, *, overwrite=False):
    """Validate all known outputs before starting potentially expensive export."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    collisions = [output_dir / name for name in ARTIFACT_NAMES if (output_dir / name).exists()]
    if collisions and not overwrite:
        names = ", ".join(path.name for path in collisions)
        raise FileExistsError(f"output already exists: {names}; use another --output-dir or --overwrite")
    if any(not path.is_file() for path in collisions):
        raise ValueError("an output artifact path is not a regular file")
    return output_dir


def convert_ncnn(onnx_path, output_dir, *, executable=None):
    """Pass fp16=0 directly: some pnnx Python wrappers drop it for input_shapes."""
    if executable is None:
        import pnnx
        executable = pnnx.EXEC_PATH
    output_dir = Path(output_dir).resolve()
    command = [
        str(executable), str(Path(onnx_path).resolve()), "inputshape=[1,1,5,5]f32",
        "device=cpu", "optlevel=2", "fp16=0",
        "ncnnparam=mine5x5.param", "ncnnbin=mine5x5.bin", "ncnnpy=mine5x5_ncnn.py",
        "pnnxparam=mine5x5.pnnx.param", "pnnxbin=mine5x5.pnnx.bin",
        "pnnxpy=mine5x5_pnnx.py", "pnnxonnx=mine5x5.pnnx.onnx",
    ]
    subprocess.run(command, cwd=output_dir, check=True)


def export_checkpoint(weight, output_dir, *, overwrite=False, samples=20):
    import torch
    from az import GomokuNet5x5

    weight = Path(weight).resolve()
    if not weight.is_file():
        raise FileNotFoundError(weight)
    if samples < 1:
        raise ValueError("samples must be positive")
    output_dir = check_output_paths(output_dir, overwrite=overwrite)
    if weight in {output_dir / name for name in ARTIFACT_NAMES}:
        raise ValueError("checkpoint must not be an output artifact path")
    net = GomokuNet5x5()
    state = torch.load(weight, map_location="cpu", weights_only=True)
    if not all(torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("checkpoint contains non-finite values")
    net.load_state_dict(state)
    net.eval()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mine5x5-export-", dir=output_dir.parent) as stage:
        stage = Path(stage)
        onnx_path = stage / "mine5x5.onnx"
        example = next(verification_inputs(samples=1, channels=1, input_mode="discrete"))
        torch.onnx.export(net, torch.from_numpy(example), str(onnx_path),
                          input_names=["in0"], output_names=["policy", "value"],
                          opset_version=13, dynamic_axes=None, external_data=False)
        convert_ncnn(onnx_path, stage)
        policy_atol, value_atol = TOLERANCES["fp32"]
        maxp, maxv = verify_models(onnx_path, stage / "mine5x5.param", stage / "mine5x5.bin",
                                 samples=samples, channels=1, input_max=2,
                                 policy_atol=policy_atol, value_atol=value_atol, input_mode="mixed")
        check_output_paths(output_dir, overwrite=overwrite)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ARTIFACT_NAMES:
            source = stage / name
            if source.is_file():
                source.replace(output_dir / name)
    return output_dir, maxp, maxv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weight", nargs="?", default="gomoku5x5_final.pt")
    parser.add_argument("--output-dir", default="exports/mine5x5")
    parser.add_argument("--overwrite", action="store_true",
                        help="explicitly replace existing model artifacts after verification")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args(argv)
    output_dir, maxp, maxv = export_checkpoint(args.weight, args.output_dir,
                                              overwrite=args.overwrite, samples=args.samples)
    print(f"FP32 export verified: policy max diff={maxp:.6g}, value max diff={maxv:.6g}")
    print(f"Saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
