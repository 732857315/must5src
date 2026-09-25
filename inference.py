"""Shared board encoding and checked inference for the 5x5 models."""

from pathlib import Path

import numpy as np

from game import BLACK, WHITE, EMPTY, FORBIDDEN, N, validate_state


def encode_state(s, me, encoding="bit"):
    """Swap stone colors for the player to move; keep forbidden cells as 3."""
    validate_state(s)
    if me not in (BLACK, WHITE):
        raise ValueError("me must be BLACK or WHITE")
    if encoding not in ("bit", "two"):
        raise ValueError("encoding must be 'bit' or 'two'")
    cells = np.array([(s >> (2 * i)) & 3 for i in range(N * N)], dtype=np.uint8).reshape(N, N)
    if encoding == "bit":
        encoded = np.where(cells == FORBIDDEN, FORBIDDEN,
                           np.where(cells == EMPTY, 0, np.where(cells == me, 1, 2)))
        return encoded.astype(np.float32)[None, None]
    if np.any(cells == FORBIDDEN):
        raise ValueError("two-channel reference encoding does not support forbidden cells (3)")
    return np.stack((cells == me, (cells != EMPTY) & (cells != me))).astype(np.float32)[None]


def validate_input(x):
    """Return one contiguous float32 board; batching is intentionally unsupported."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape not in ((1, 1, N, N), (1, 2, N, N)):
        raise ValueError(f"expected input (1, 1|2, {N}, {N}), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("input contains non-finite values")
    return np.ascontiguousarray(x)


def _check_return_code(code, operation):
    if code != 0:
        raise RuntimeError(f"ncnn {operation} failed with code {code}")


def load_net(param, bin):
    """Load ncnn weights and fail immediately on missing or rejected files."""
    param, bin = Path(param), Path(bin)
    for path in (param, bin):
        if not path.is_file():
            raise FileNotFoundError(path)
    import ncnn

    net = ncnn.Net()
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.opt.use_packing_layout = False
    _check_return_code(net.load_param(str(param)), f"load_param({param})")
    _check_return_code(net.load_model(str(bin)), f"load_model({bin})")
    return net


def np_to_mat(x):
    """Copy NCHW input into owned ncnn storage, preserving channel layout."""
    x = validate_input(x)
    import ncnn

    return ncnn.Mat(x[0]).clone()


def validate_policy(policy):
    policy = np.asarray(policy, dtype=np.float32).reshape(-1)
    if policy.size != N * N:
        raise ValueError(f"expected policy size {N * N}, got {policy.size}")
    if not np.isfinite(policy).all():
        raise ValueError("model policy contains non-finite values")
    return policy.copy()


def validate_outputs(policy, value):
    policy = validate_policy(policy)
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.size != 1:
        raise ValueError(f"expected value size 1, got {value.size}")
    if not np.isfinite(value).all():
        raise ValueError("model output contains non-finite values")
    # Copies keep returned arrays independent of an extractor's temporary storage.
    return policy, value.copy()


def legal_policy(policy, state):
    """Return an action-only view with stones and forbidden cells set to -inf.

    Raw inference remains finite and unchanged for numerical verification.
    An entirely unavailable window yields all -inf; use choose_legal_move to
    handle that case without accidentally selecting the first cell.
    """
    validate_state(state)
    masked = validate_policy(policy)
    available = np.array([((state >> (2 * i)) & 3) == EMPTY for i in range(N * N)])
    masked[~available] = -np.inf
    return masked


def choose_legal_move(policy, state):
    """Choose a legal local cell, or None when the window has no empty cell."""
    masked = legal_policy(policy, state)
    return int(np.argmax(masked)) if np.isfinite(masked).any() else None


def ncnn_infer(net, x):
    """Return unmasked, finite policy/value outputs for numerical comparison."""
    ex = net.create_extractor()
    _check_return_code(ex.input("in0", np_to_mat(x)), "input(in0)")
    code, policy = ex.extract("out0")
    _check_return_code(code, "extract(out0)")
    code, value = ex.extract("out1")
    _check_return_code(code, "extract(out1)")
    return validate_outputs(np.array(policy), np.array(value))
