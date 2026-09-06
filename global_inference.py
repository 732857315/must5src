"""Checked whole-board inference, independent of the two local U-Net weights."""

from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from board_rules import normalize_board, board_winner
from global_model import GlobalBoardNet, encode_global, masked_global_policy

CHECKPOINT_FORMAT = "gomoku_global_v1"
GLOBAL_WEIGHT = 0.7


def load_global_model(path):
    """Load a versioned CPU checkpoint without general pickle deserialization."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected a versioned gomoku_global_v1 whole-board checkpoint")
    if payload.get("role", "global") != "global":
        raise ValueError("Expected global model weights")
    configuration = {}
    for name, maximum in (("base_channels", 256), ("token_dim", 256), ("attention_heads", 32)):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= maximum:
            raise ValueError(f"Invalid checkpoint architecture field: {name}")
        configuration[name] = int(value)
    if configuration["token_dim"] % configuration["attention_heads"]:
        raise ValueError("token_dim must be divisible by attention_heads")
    weights = payload.get("state_dict")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Checkpoint state_dict must be a nonempty tensor dictionary")
    for name, tensor in weights.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError("Checkpoint contains invalid or nonfinite weights")
    if "trained" in payload and not isinstance(payload["trained"], bool):
        raise ValueError("Checkpoint trained flag must be boolean")
    configuration["input_mode"] = payload.get("input_mode", "absolute_rgb")
    model = GlobalBoardNet(**configuration)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model, {key: value for key, value in payload.items() if key != "state_dict"}


def _policy(values, board, name, *, terminal=False):
    values = np.asarray(values, dtype=np.float64)
    if (values.shape != board.shape or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1 + 1e-6)
            or np.any(values[board != 0] != 0)):
        raise ValueError(f"{name} must contain finite legal-cell probabilities matching the board")
    total = float(values.sum())
    if terminal:
        if total != 0:
            raise ValueError(f"{name} must be zero on a terminal board")
        return values.copy()
    if not np.isclose(total, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError(f"{name} must sum to one on a live board")
    return values / total


def analyze_global(board, side, local_analysis, global_model):
    """Return raw global and 70/30 global/local policies plus an actor-side value.

    The value is a neural estimate in [-1,1], not a win probability or a proof.
    Local inputs must belong to this same board and explicitly supplied actor.
    Terminal positions have zero action policies and no network-value estimate.
    """
    board = normalize_board(board)
    if isinstance(side, bool) or not isinstance(side, Integral) or side not in (1, 2):
        raise ValueError("side must be BLACK=1 or WHITE=2")
    side = int(side)
    for key in ("side", "my_side"):
        if key in local_analysis and local_analysis[key] != side:
            raise ValueError("Local analysis belongs to another acting side")
    terminal = bool(board_winner(board) or not np.any(board == 0))
    opponent = _policy(local_analysis["opponent_policy"], board, "opponent_policy", terminal=terminal)
    local = _policy(local_analysis["raw_play_policy"], board, "raw_play_policy", terminal=terminal)
    if terminal:
        return dict(global_policy=np.zeros(board.shape, dtype=np.float64),
                    combined_policy=np.zeros(board.shape, dtype=np.float64), value=None,
                    value_source="terminal_no_inference", global_weight=GLOBAL_WEIGHT)

    encoded = encode_global(board, side, opponent, local,
                            input_mode=getattr(global_model, "input_mode", "absolute_rgb"))
    encoded = np.asarray(encoded, dtype=np.float32)
    if encoded.shape != (9, *board.shape) or not np.isfinite(encoded).all():
        raise ValueError("Global encoder must return a finite 9xHxW array")
    try:
        device = next(global_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    global_model.eval()
    with torch.inference_mode():
        logits, values = global_model(torch.from_numpy(encoded.copy()).unsqueeze(0).to(device))
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 1, *board.shape):
            raise ValueError("Global policy output must have shape 1x1xHxW")
        if not torch.isfinite(logits).all():
            raise ValueError("Global policy contains nonfinite logits")
        if not isinstance(values, torch.Tensor) or tuple(values.shape) != (1,) or not torch.isfinite(values).all():
            raise ValueError("Global value output must be a finite one-element vector")
        value = float(values[0].cpu())
        if not -1.000001 <= value <= 1.000001:
            raise ValueError("Global value must lie in [-1,1]")
        probabilities = masked_global_policy(logits, [board])[0, 0].cpu().numpy()
    global_policy = _policy(probabilities, board, "global_policy")
    combined = GLOBAL_WEIGHT * global_policy + (1.0 - GLOBAL_WEIGHT) * local
    combined[board != 0] = 0.0
    combined /= combined.sum()
    return dict(global_policy=global_policy, combined_policy=combined, value=float(np.clip(value, -1, 1)),
                value_source="network_estimate", global_weight=GLOBAL_WEIGHT)


def evaluate_global_value(board, side, model):
    """Cheap leaf evaluation with zero auxiliary maps, an explicit trained mode.

    Nonterminal values are network estimates. Terminal outcomes are returned
    directly from board rules; search must never turn this evaluator into proof.
    """
    board = normalize_board(board)
    if isinstance(side, bool) or not isinstance(side, Integral) or side not in (1, 2):
        raise ValueError("side must be BLACK=1 or WHITE=2")
    side = int(side)
    winner = board_winner(board)
    if winner or not np.any(board == 0):
        return float(1 if winner == side else -1 if winner else 0)
    auxiliary = np.zeros(board.shape, dtype=np.float32)
    encoded = encode_global(board, side, auxiliary, auxiliary,
                            input_mode=getattr(model, "input_mode", "absolute_rgb"))
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.eval()
    with torch.inference_mode():
        _, values = model(torch.from_numpy(encoded.copy()).unsqueeze(0).to(device))
        if not isinstance(values, torch.Tensor) or tuple(values.shape) != (1,) or not torch.isfinite(values).all():
            raise ValueError("Global value output must be a finite one-element vector")
        value = float(values[0].cpu())
    if not -1.000001 <= value <= 1.000001:
        raise ValueError("Global value must lie in [-1,1]")
    return float(np.clip(value, -1, 1))
