"""Trainable whole-board policy/value model that consumes both local policies.

The nine input planes are RGB, continuous opponent/play probability, legal
moves, side, and row/column coordinates. Legacy inputs use absolute colors and
side (-1 black/+1 white). Optional relative_rgb encodes the actor as black, the
opponent as white, and fixes the encoded side to -1.
Three dilated residual blocks retain spatial detail. Global attention operates
on at most 8x8 pooled tokens, so its quadratic cost never depends on board area.
Different board shapes must be batched separately. Value is a prediction, not
a search certificate or a guarantee about the opening player.
"""

import math
from numbers import Integral

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from board_rules import BLACK, WHITE, EMPTY, normalize_board, board_winner
from unet_codec import EMPTY_RGB, BLACK_RGB, WHITE_RGB, FORBIDDEN_RGB


INPUT_CHANNELS = 9
INPUT_MODES = ("absolute_rgb", "relative_rgb")
MAX_TOKEN_SIDE = 8
CHANNEL_NAMES = ("red", "green", "blue", "opponent_policy", "play_policy",
                 "legal", "side", "row", "column")
_PALETTE = np.asarray((EMPTY_RGB, BLACK_RGB, WHITE_RGB, FORBIDDEN_RGB), dtype=np.float32) / 255


def encode_global(board, side, opp, play, *, input_mode="absolute_rgb"):
    """Encode one HxW board as contiguous float32 (9,H,W).

    Probability planes are already fused over the real board: their continuous
    values are preserved without display quantization or per-window rescaling.
    All-zero planes are allowed when no local prediction is available. Supplied
    probabilities must have legal support. Coordinates are regenerated here;
    augmentation should transform board/maps and then call this function.
    relative_rgb normalizes the actor to black and the opponent to white; empty
    and forbidden cells, probability coordinates, and caller board are preserved.
    """
    board = normalize_board(board)
    if isinstance(side, (bool, np.bool_)) or not isinstance(side, Integral) or side not in (BLACK, WHITE):
        raise ValueError("side must be explicit BLACK=1 or WHITE=2")
    if not isinstance(input_mode, str) or input_mode not in INPUT_MODES:
        raise ValueError("input_mode must be absolute_rgb or relative_rgb")
    if input_mode == "relative_rgb":
        if side == WHITE:
            board = np.array([0, 2, 1, 3], dtype=np.uint8)[board]
        side = BLACK
    legal = (board == EMPTY) & (board_winner(board) == EMPTY)
    probabilities = []
    for name, values in (("opp", opp), ("play", play)):
        values = np.asarray(values, dtype=np.float32)
        if values.shape != board.shape or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"{name} must have board shape and finite probabilities in [0,1]")
        if np.any(values[~legal] != 0):
            raise ValueError(f"{name} cannot assign probability to illegal or terminal cells")
        probabilities.append(values)
    height, width = board.shape
    result = np.empty((INPUT_CHANNELS, height, width), dtype=np.float32)
    result[:3] = _PALETTE[board].transpose(2, 0, 1)
    result[3], result[4] = probabilities
    result[5] = legal
    result[6] = -1.0 if side == BLACK else 1.0
    result[7] = (np.linspace(-1, 1, height, dtype=np.float32) if height > 1 else np.zeros(1))[:, None]
    result[8] = (np.linspace(-1, 1, width, dtype=np.float32) if width > 1 else np.zeros(1))[None, :]
    return result


def _norm(channels):
    # At least two channels per group also supports a single-cell board/batch.
    groups = next(value for value in range(min(8, channels // 2), 0, -1) if channels % value == 0)
    return nn.GroupNorm(groups, channels)


class _DilatedResidual(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            _norm(channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            _norm(channels),
        )

    def forward(self, value):
        return F.silu(value + self.body(value))


class GlobalBoardNet(nn.Module):
    """Small independent global policy/value net; default attention has 64 tokens.

    forward(x) returns logits (B,1,H,W) and tanh values (B,). Use
    masked_global_policy for legal normalized probabilities. This module does
    not load, share, freeze, or modify either 5x5 U-Net's weights.
    """
    def __init__(self, base_channels=16, token_dim=32, attention_heads=4, *, input_mode="absolute_rgb"):
        super().__init__()
        for name, value, minimum in (("base_channels", base_channels, 2), ("token_dim", token_dim, 2),
                                     ("attention_heads", attention_heads, 1)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if token_dim % attention_heads:
            raise ValueError("token_dim must be divisible by attention_heads")
        if not isinstance(input_mode, str) or input_mode not in INPUT_MODES:
            raise ValueError("input_mode must be absolute_rgb or relative_rgb")
        self.input_mode = input_mode
        self.role = "global"
        self.base_channels = int(base_channels)
        self.token_dim = int(token_dim)
        self.attention_heads = int(attention_heads)
        self.max_token_side = MAX_TOKEN_SIDE
        self.input_channels = INPUT_CHANNELS
        self.stem = nn.Sequential(nn.Conv2d(INPUT_CHANNELS, base_channels, 3, padding=1, bias=False),
                                  _norm(base_channels), nn.SiLU())
        self.spatial = nn.Sequential(*(_DilatedResidual(base_channels, rate) for rate in (1, 2, 4)))
        self.token_projection = nn.Conv2d(base_channels, token_dim, 1)
        self.attention = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=attention_heads, dim_feedforward=token_dim * 2,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.context_projection = nn.Conv2d(token_dim, base_channels, 1)
        self.fusion = nn.Sequential(nn.Conv2d(base_channels * 2, base_channels, 1, bias=False),
                                    _norm(base_channels), nn.SiLU())
        self.policy_head = nn.Conv2d(base_channels, 1, 1)
        self.value_head = nn.Sequential(nn.Linear(base_channels + token_dim, base_channels),
                                        nn.SiLU(), nn.Linear(base_channels, 1), nn.Tanh())

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 4 or inputs.shape[1] != INPUT_CHANNELS:
            raise ValueError("inputs must be a tensor shaped (B,9,H,W)")
        if not all(inputs.shape) or not inputs.is_floating_point() or not torch.isfinite(inputs).all():
            raise ValueError("inputs must have nonempty dimensions and finite floating-point values")
        if torch.any(inputs[:, :6] < 0) or torch.any(inputs[:, :6] > 1) or torch.any(inputs[:, 6:] < -1) or torch.any(inputs[:, 6:] > 1):
            raise ValueError("RGB/probability/legal planes must be in [0,1]; side/coordinates in [-1,1]")
        encoded = self.spatial(self.stem(inputs))
        token_shape = tuple(min(MAX_TOKEN_SIDE, size) for size in encoded.shape[-2:])
        tokens = F.adaptive_avg_pool2d(self.token_projection(encoded), token_shape)
        tokens = self.attention(tokens.flatten(2).transpose(1, 2))
        context = tokens.transpose(1, 2).reshape(inputs.shape[0], self.token_dim, *token_shape)
        context = F.interpolate(self.context_projection(context), size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fusion(torch.cat((encoded, context), dim=1))
        logits = self.policy_head(fused)
        pooled = torch.cat((fused.mean(dim=(-2, -1)), tokens.mean(dim=1)), dim=1)
        values = self.value_head(pooled).squeeze(-1)
        return logits, values


def _validate_logits(logits):
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[1] != 1 or not all(logits.shape):
        raise ValueError("logits must be shaped (B,1,H,W) with nonempty dimensions")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating-point values")


def _board_mask(logits, boards):
    if isinstance(boards, torch.Tensor):
        boards = boards.detach().cpu().numpy()
    try:
        boards = np.asarray(boards)
    except (TypeError, ValueError) as exc:
        raise ValueError("boards must form a batch of same-shape integer boards") from exc
    if boards.ndim == 2:
        boards = boards[None]
    expected = (logits.shape[0], *logits.shape[-2:])
    if boards.shape != expected:
        raise ValueError("boards must contain one same-shape board per logits item")
    masks = []
    for board in boards:
        board = normalize_board(board)
        masks.append((board == EMPTY) & (board_winner(board) == EMPTY))
    return torch.as_tensor(np.stack(masks), dtype=torch.bool, device=logits.device).flatten(1)


def masked_global_policy(logits, boards):
    """Normalize globally over legal cells; terminal and illegal cells are zero."""
    _validate_logits(logits)
    mask = _board_mask(logits, boards)
    flat = logits.flatten(1).masked_fill(~mask, -torch.inf)
    safe = torch.where(mask.any(1, keepdim=True), flat, torch.zeros_like(flat))
    return F.softmax(safe, dim=1).masked_fill(~mask, 0).reshape_as(logits)


def global_loss(logits, values, target_policy, target_values, value_mask, boards,
                *, value_weight=1.0, return_components=False, policy_mask=None,
                policy_weights=None, losing_mask=None, winning_mask=None):
    """Legal soft policy CE, partial proof constraints and labelled value MSE.

    The optional boolean policy_mask selects CE rows, whose targets sum to one;
    all other rows require zero targets. Nonnegative policy_weights scale CE
    before averaging over its active rows, without normalization by weight sum.
    Boolean losing_mask/winning_mask have shape (B,H,W), contain only legal
    moves and are disjoint. Each constraint averages over its own active rows.
    Proved losing moves remain in the legal softmax. If all legal moves lose,
    the entire policy term for that row is disabled and its CE target is zero.

    Omitted options preserve the original live-board CE and labelled value MSE.
    Unknown outcomes use value_mask=False instead of being relabelled draws.
    Optional components include the original loss/policy_loss/value_loss keys,
    three policy terms, and their scalar integer ce/losing/winning_count values.
    """
    _validate_logits(logits)
    if isinstance(value_weight, bool) or not isinstance(value_weight, (int, float)) or not math.isfinite(value_weight) or value_weight < 0:
        raise ValueError("value_weight must be finite and nonnegative")
    batch = logits.shape[0]
    mask = _board_mask(logits, boards)
    live = mask.any(1)

    def boolean_mask(value, shape, name):
        try:
            result = torch.as_tensor(value, device=logits.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{name} must be a boolean tensor shaped {shape}") from exc
        if result.dtype != torch.bool or tuple(result.shape) != shape:
            raise ValueError(f"{name} must be a boolean tensor shaped {shape}")
        return result

    requested_ce = live if policy_mask is None else boolean_mask(policy_mask, (batch,), "policy_mask")
    constraint_shape = (batch, *logits.shape[-2:])
    losing = torch.zeros_like(mask) if losing_mask is None else boolean_mask(losing_mask, constraint_shape, "losing_mask").flatten(1)
    winning = torch.zeros_like(mask) if winning_mask is None else boolean_mask(winning_mask, constraint_shape, "winning_mask").flatten(1)
    if torch.any((losing | winning) & ~mask):
        raise ValueError("proof constraints must contain only legal moves on live boards")
    if torch.any(losing & winning):
        raise ValueError("losing_mask and winning_mask must be disjoint")
    all_losing = live & (losing == mask).all(1)
    active = live & requested_ce & ~all_losing
    if policy_weights is None:
        weights = torch.ones(batch, dtype=logits.dtype, device=logits.device)
    else:
        try:
            weights = torch.as_tensor(policy_weights, device=logits.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("policy_weights must be finite nonnegative real values shaped (B,)") from exc
        if weights.shape != (batch,) or weights.dtype == torch.bool or weights.is_complex():
            raise ValueError("policy_weights must be finite nonnegative real values shaped (B,)")
        weights = weights.to(logits.dtype)
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("policy_weights must be finite nonnegative real values shaped (B,)")
    target_policy = torch.as_tensor(target_policy, dtype=logits.dtype, device=logits.device)
    allowed = ((batch, *logits.shape[-2:]), tuple(logits.shape), (batch, logits.shape[-2] * logits.shape[-1]))
    if tuple(target_policy.shape) not in allowed:
        raise ValueError("target_policy must be shaped (B,H,W), (B,1,H,W), or (B,H*W)")
    target_policy = target_policy.flatten(1)
    if not torch.isfinite(target_policy).all() or torch.any(target_policy < 0) or torch.any(target_policy.masked_select(~mask) != 0):
        raise ValueError("policy targets must be finite nonnegative values with legal support")
    if torch.any(target_policy[~active] != 0):
        raise ValueError("inactive CE rows must have exactly zero policy targets")
    if not torch.allclose(target_policy.sum(1), active.to(logits.dtype), atol=1e-5, rtol=1e-5):
        raise ValueError("active CE targets must sum to one and inactive targets to zero")
    flat_logits = logits.flatten(1)
    zero = flat_logits[:, 0].mul(0).sum()
    if active.any():
        selected_mask = mask[active]
        flat = flat_logits[active].masked_fill(~selected_mask, -torch.inf)
        log_probability = F.log_softmax(flat, dim=1).masked_fill(~selected_mask, 0)
        ce_term = (-(target_policy[active] * log_probability).sum(1) * weights[active]).mean()
    else:
        ce_term = zero

    def set_term(eligible, retained):
        if not eligible.any():
            return zero
        scores = flat_logits[eligible]
        normalizer = torch.logsumexp(scores.masked_fill(~mask[eligible], -torch.inf), dim=1)
        retained_mass = torch.logsumexp(scores.masked_fill(~retained[eligible], -torch.inf), dim=1)
        return (normalizer - retained_mass).mean()

    losing_active = live & losing.any(1) & ~all_losing
    winning_active = live & winning.any(1)
    losing_term = set_term(losing_active, mask & ~losing)
    winning_term = set_term(winning_active, winning)
    policy_term = ce_term + losing_term + winning_term
    values = torch.as_tensor(values, dtype=logits.dtype, device=logits.device)
    target_values = torch.as_tensor(target_values, dtype=logits.dtype, device=logits.device)
    value_mask = torch.as_tensor(value_mask, device=logits.device)
    if values.shape != (batch,) or target_values.shape != (batch,) or value_mask.shape != (batch,):
        raise ValueError("values, target_values and value_mask must be shaped (B,)")
    if not torch.isfinite(values).all() or not torch.isfinite(target_values).all() or torch.any(target_values.abs() > 1):
        raise ValueError("values must be finite and target_values within [-1,1]")
    if torch.any((value_mask != 0) & (value_mask != 1)):
        raise ValueError("value_mask must contain only boolean or 0/1 entries")
    labelled = value_mask.bool()
    value_term = F.mse_loss(values[labelled], target_values[labelled]) if labelled.any() else values.sum() * 0
    loss = policy_term + value_weight * value_term
    if return_components:
        return {"loss": loss, "policy_loss": policy_term, "value_loss": value_term,
                "ce_loss": ce_term, "losing_loss": losing_term, "winning_loss": winning_term,
                "ce_count": active.sum(), "losing_count": losing_active.sum(),
                "winning_count": winning_active.sum()}
    return loss
