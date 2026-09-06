"""Independent, small U-Nets for opponent prediction and own-move selection."""

import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from game import N, CELL_COUNT, BLACK, WHITE, EMPTY, legal_moves, validate_state, winner


def _group_norm(channels):
    return nn.GroupNorm(math.gcd(channels, 8), channels)


class _DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(),
        )


class _UNet5x5(nn.Module):
    def __init__(self, role, base_channels=16):
        super().__init__()
        if isinstance(base_channels, bool) or not isinstance(base_channels, int) or base_channels < 1:
            raise ValueError("base_channels must be a positive integer")
        self.role = role
        self.base_channels = base_channels
        self.encoder1 = _DoubleConv(3, base_channels)
        self.side_embedding = nn.Embedding(2, base_channels)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.encoder2 = _DoubleConv(base_channels, base_channels * 2)
        self.bottleneck = _DoubleConv(base_channels * 2, base_channels * 4)
        self.decoder2 = _DoubleConv(base_channels * 6, base_channels * 2)
        self.decoder1 = _DoubleConv(base_channels * 3, base_channels)
        self.policy_head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, rgb, side):
        """Map Bx3x5x5 RGB plus explicit actual color B (1/2) to Bx1x5x5 logits.

        side is the color this model is predicting, supplied from the actual
        game turn. A local window's stone counts do not determine that color.
        """
        if not isinstance(rgb, torch.Tensor) or rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, N, N):
            raise ValueError("rgb must be a tensor shaped (B,3,5,5)")
        if rgb.shape[0] < 1 or not rgb.is_floating_point():
            raise ValueError("rgb must have a nonempty floating-point batch")
        if not torch.isfinite(rgb).all() or torch.any(rgb < 0) or torch.any(rgb > 1):
            raise ValueError("rgb must be finite and within [0,1]")
        side = torch.as_tensor(side, device=rgb.device)
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if side.shape != (rgb.shape[0],) or side.dtype not in integer_types:
            raise ValueError("side must contain one explicit integer color per batch item")
        if torch.any((side != BLACK) & (side != WHITE)):
            raise ValueError("side colors must be BLACK=1 or WHITE=2")
        side = side.to(dtype=torch.long) - BLACK

        skip1 = self.encoder1(rgb) + self.side_embedding(side)[:, :, None, None]
        skip2 = self.encoder2(self.pool(skip1))
        encoded = self.bottleneck(self.pool(skip2))
        up2 = F.interpolate(encoded, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        decoded2 = self.decoder2(torch.cat((up2, skip2), dim=1))
        up1 = F.interpolate(decoded2, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        decoded1 = self.decoder1(torch.cat((up1, skip1), dim=1))
        return self.policy_head(decoded1)


class OpponentUNet5x5(_UNet5x5):
    def __init__(self, base_channels=16):
        super().__init__("opponent", base_channels)


class PlayUNet5x5(_UNet5x5):
    def __init__(self, base_channels=16):
        super().__init__("play", base_channels)


def _validate_logits(logits):
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or tuple(logits.shape[1:]) != (1, N, N):
        raise ValueError("logits must be a tensor shaped (B,1,5,5)")
    if logits.shape[0] < 1 or not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must have a nonempty batch of finite floating-point values")


def _state_mask(states, batch_size, device):
    if isinstance(states, Integral) and not isinstance(states, bool):
        states = [states]
    elif isinstance(states, torch.Tensor):
        if states.ndim != 1 or states.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("states must contain packed integer boards")
        states = states.detach().cpu().tolist()
    else:
        try:
            states = list(states)
        except TypeError as exc:
            raise ValueError("states must contain packed integer boards") from exc
    if len(states) != batch_size:
        raise ValueError("states must contain one board per batch item")
    mask = torch.zeros((batch_size, CELL_COUNT), dtype=torch.bool, device=device)
    for batch_index, state in enumerate(states):
        if isinstance(state, bool) or not isinstance(state, Integral):
            raise ValueError("states must contain packed integer boards")
        state = int(state)
        validate_state(state)
        if winner(state) == EMPTY:
            mask[batch_index, legal_moves(state)] = True
    return mask


def masked_probabilities(logits, states):
    """Return Bx1x5x5 probabilities, exactly zero on stones/3 and terminal boards."""
    _validate_logits(logits)
    mask = _state_mask(states, logits.shape[0], logits.device)
    flat = logits.reshape(logits.shape[0], CELL_COUNT)
    masked = flat.masked_fill(~mask, -torch.inf)
    # All-masked softmax would produce NaN. Terminal rows are zeroed below.
    safe = torch.where(mask.any(dim=1, keepdim=True), masked, torch.zeros_like(masked))
    probabilities = F.softmax(safe, dim=1).masked_fill(~mask, 0)
    return probabilities.reshape_as(logits)


def policy_loss(logits, targets, states):
    """Masked soft-target cross entropy; reject terminal, illegal, or empty targets."""
    _validate_logits(logits)
    batch_size = logits.shape[0]
    mask = _state_mask(states, batch_size, logits.device)
    targets = torch.as_tensor(targets, dtype=logits.dtype, device=logits.device)
    if tuple(targets.shape) not in ((batch_size, CELL_COUNT), (batch_size, N, N), (batch_size, 1, N, N)):
        raise ValueError("targets must be shaped (B,25), (B,5,5), or (B,1,5,5)")
    targets = targets.reshape(batch_size, CELL_COUNT)
    if not torch.isfinite(targets).all() or torch.any(targets < 0):
        raise ValueError("targets must be finite nonnegative probabilities")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("terminal boards have no policy training target")
    if torch.any(targets.masked_select(~mask) != 0):
        raise ValueError("targets cannot assign probability to occupied or forbidden cells")
    if not torch.allclose(targets.sum(dim=1), torch.ones(batch_size, dtype=targets.dtype, device=targets.device),
                          atol=1e-5, rtol=1e-5):
        raise ValueError("each target must sum to one; empty targets are invalid")
    flat = logits.reshape(batch_size, CELL_COUNT).masked_fill(~mask, -torch.inf)
    log_probabilities = F.log_softmax(flat, dim=1).masked_fill(~mask, 0)
    return -(targets * log_probabilities).sum(dim=1).mean()
