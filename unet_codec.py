"""Absolute-color RGB encoding for the two 5x5 U-Nets.

Policy overlays are continuous by default. Display quantization must never be
used as the input to the play network or as a probability target.
"""

from numbers import Integral

import numpy as np

from game import N, CELL_COUNT, EMPTY, BLACK, WHITE, FORBIDDEN, validate_state


EMPTY_RGB = (255, 244, 220)
BLACK_RGB = (24, 27, 33)
WHITE_RGB = (255, 255, 255)
FORBIDDEN_RGB = (128, 128, 128)
RED_RGB = (230, 35, 45)
GREEN_RGB = (20, 180, 80)
_PALETTE = np.asarray((EMPTY_RGB, BLACK_RGB, WHITE_RGB, FORBIDDEN_RGB), dtype=np.float32)


def normalize_grid(state):
    """Copy packed state, 25 integer cells, or a 5x5 integer grid into uint8."""
    if isinstance(state, Integral) and not isinstance(state, (bool, np.bool_)):
        packed = int(state)
        validate_state(packed)
        return np.asarray([(packed >> (2 * i)) & 3 for i in range(CELL_COUNT)],
                          dtype=np.uint8).reshape(N, N)
    cells = np.asarray(state)
    if cells.shape not in ((CELL_COUNT,), (N, N)) or cells.dtype.kind not in "iu":
        raise ValueError("state must be a packed integer, 25 integer cells, or a 5x5 integer grid")
    if np.any(cells < EMPTY) or np.any(cells > FORBIDDEN):
        raise ValueError("board cells must be 0, 1, 2, or 3")
    return cells.astype(np.uint8, copy=True).reshape(N, N)


def encode_rgb(state):
    """Return float32 RGB (3,5,5) in [0,1], retaining absolute black/white."""
    cells = normalize_grid(state)
    image = _PALETTE[cells] / np.float32(255)
    return np.ascontiguousarray(image.transpose(2, 0, 1))


def policy_rgb(state, probabilities, color, levels=None):
    """Overlay probabilities on empty cells; stone/forbidden colors never change.

    levels=None preserves continuous probabilities for model input. A positive
    integer is display-only: zero stays clear and positive probabilities use
    levels shades, rounded upward so small positive values remain visible.
    The caller validates legal support and normalization when this is training
    data; this visual encoder accepts any 25 finite values in [0,1].
    """
    if color not in ("red", "green"):
        raise ValueError("color must be 'red' or 'green'")
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.size != CELL_COUNT:
        raise ValueError("probabilities must contain 25 values")
    probabilities = probabilities.reshape(N, N)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("probabilities must be finite and within [0,1]")
    if levels is not None:
        if isinstance(levels, (bool, np.bool_)) or not isinstance(levels, Integral) or levels < 1:
            raise ValueError("levels must be a positive integer or None")
        probabilities = np.ceil(probabilities * int(levels)) / int(levels)
    cells = normalize_grid(state)
    image = encode_rgb(cells)
    target = np.asarray(RED_RGB if color == "red" else GREEN_RGB, dtype=np.float32) / np.float32(255)
    alpha = np.where(cells == EMPTY, probabilities, np.float32(0))[None]
    return np.ascontiguousarray(image * (1 - alpha) + target[:, None, None] * alpha,
                                dtype=np.float32)
