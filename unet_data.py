"""Auditable supervised samples for the opponent and move-selection U-Nets.

The caller supplies the side explicitly: a cropped board cannot determine the
global turn. Opponent labels are observed moves on the position BEFORE that
move. Play-model inputs use a prediction made from that same current position;
future observed opponent moves must never be substituted for this prediction.
Numeric arrays alone cannot establish how a prediction was obtained.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from game import BLACK, WHITE, EMPTY, CELL_COUNT, N, PERMS
from game import legal_moves, move_rejection_reason, validate_state, winner
from unet_codec import encode_rgb, policy_rgb


@dataclass(frozen=True)
class SupervisedSample:
    state: int
    rgb: np.ndarray
    side: int
    target: np.ndarray
    legal_mask: np.ndarray
    role: str


def _position(state, side):
    validate_state(state)
    if isinstance(side, bool) or not isinstance(side, Integral) or side not in (BLACK, WHITE):
        raise ValueError("side must explicitly be BLACK (1) or WHITE (2)")
    moves = legal_moves(state)
    if winner(state) != EMPTY or not moves:
        raise ValueError("cannot assign an action target to a terminal position")
    mask = np.zeros(CELL_COUNT, dtype=bool)
    mask[moves] = True
    return mask


def validate_policy(state, probabilities, *, name="policy"):
    """Require a normalized distribution supported only on empty cells."""
    values = np.asarray(probabilities)
    if values.shape not in ((CELL_COUNT,), (N, N)):
        raise ValueError(f"{name} must have shape (25,) or (5, 5)")
    if values.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real probabilities")
    values = values.astype(np.float64).reshape(CELL_COUNT)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative probabilities")
    if not np.isclose(values.sum(), 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError(f"{name} probabilities must sum to 1")
    mask = np.zeros(CELL_COUNT, dtype=bool)
    mask[legal_moves(state)] = True
    if (values[~mask] != 0).any():
        raise ValueError(f"{name} assigns probability to an occupied or forbidden cell")
    return values.astype(np.float32)


def _target(state, target):
    if isinstance(target, Integral) and not isinstance(target, bool):
        move = int(target)
        reason = move_rejection_reason(state, move)
        if reason is not None:
            raise ValueError(f"invalid target move: {reason}")
        values = np.zeros(CELL_COUNT, dtype=np.float32)
        values[move] = 1.0
        return values
    return validate_policy(state, target, name="target")


def make_opponent_sample(state, opponent_side, observed_move):
    """Encode the position immediately before a legally observed opponent move.

    The move has not yet been applied to ``state``. ``opponent_side`` is the
    color whose next action this model predicts, not the observer's color.
    """
    mask = _position(state, opponent_side)
    if isinstance(observed_move, bool) or not isinstance(observed_move, Integral):
        raise ValueError("observed_move must be a single integer action index")
    target = _target(state, observed_move)
    return SupervisedSample(state, encode_rgb(state), int(opponent_side), target, mask, "opponent")


def make_play_sample(state, side, target_policy, opponent_prediction):
    """Use a current-position opponent prediction as continuous red input.

    ``target_policy`` may be a chosen action or a normalized expert/MCTS policy.
    ``opponent_prediction`` must come from the opponent model on this very
    state; this function deliberately never derives it from the target or a
    later real move. Keep prediction and trajectory provenance at the caller.
    """
    mask = _position(state, side)
    target = _target(state, target_policy)
    prediction = validate_policy(state, opponent_prediction, name="opponent_prediction")
    rgb = policy_rgb(state, prediction, color="red", levels=None)
    return SupervisedSample(state, rgb, int(side), target, mask, "play")


def canonical_key(state, side):
    """Identity modulo D4 and simultaneous color/side reversal."""
    validate_state(state)
    if isinstance(side, bool) or not isinstance(side, Integral) or side not in (BLACK, WHITE):
        raise ValueError("side must explicitly be BLACK (1) or WHITE (2)")
    cells = [(state >> (2 * i)) & 3 for i in range(CELL_COUNT)]
    cells = [1 if cell == side else 2 if cell in (BLACK, WHITE) else cell for cell in cells]
    return min(sum(cells[old] << (2 * new) for old, new in enumerate(perm)) for perm in PERMS)


def augment_sample(sample, symmetry, color_swap=False):
    """Transform every field together; color reversal also flips the side.

    Black/white stones exchange colors, while empty-cell red probabilities and
    gray forbidden pixels remain intact. These copies keep one canonical ID.
    """
    if isinstance(symmetry, bool) or not isinstance(symmetry, Integral) or not 0 <= symmetry < len(PERMS):
        raise ValueError("symmetry must be an integer from 0 to 7")
    permutation = np.asarray(PERMS[int(symmetry)])
    state = sum(((sample.state >> (2 * old)) & 3) << (2 * new)
                for old, new in enumerate(PERMS[int(symmetry)]))
    rgb = np.empty_like(sample.rgb.reshape(3, CELL_COUNT))
    rgb[:, permutation] = sample.rgb.reshape(3, CELL_COUNT)
    target = np.empty_like(sample.target)
    target[permutation] = sample.target
    mask = np.empty_like(sample.legal_mask)
    mask[permutation] = sample.legal_mask
    side = sample.side
    if color_swap:
        cells = [(state >> (2 * i)) & 3 for i in range(CELL_COUNT)]
        cells = [3 - cell if cell in (BLACK, WHITE) else cell for cell in cells]
        state = sum(cell << (2 * i) for i, cell in enumerate(cells))
        side = BLACK + WHITE - side
        stones = np.array([cell in (BLACK, WHITE) for cell in cells])
        rgb[:, stones] = encode_rgb(state).reshape(3, CELL_COUNT)[:, stones]
    return SupervisedSample(state, rgb.reshape(3, N, N), side, target, mask, sample.role)
