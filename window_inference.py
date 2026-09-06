"""Read-only inference over overlapping 5x5 windows; no training or global search."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from game import BLACK, WHITE, EMPTY, other_player, winner
from inference import encode_state, legal_policy, load_net, ncnn_infer
from windows import BoardWindow, format_grid, iter_windows, pad_board, validate_grid


@dataclass(frozen=True)
class BoardPrediction:
    policy: np.ndarray
    legal_mask: np.ndarray
    move: tuple[int, int] | None
    window_count: int
    terminal: bool


def infer_window(net, window: BoardWindow, me):
    """Return legal local logits and a local value; caller supplies the GLOBAL turn."""
    policy, value = ncnn_infer(net, encode_state(window.state, me, "bit"))
    return legal_policy(policy, window.state), float(value[0])


def predict_board(net, grid, me):
    """Average local legal probabilities, then normalize over the real empty cells.

    This is a local policy aggregation, not a proof of the best global move.
    Local stone counts do not determine whose turn it is on the actual board.
    """
    other_player(me)
    rows = validate_grid(grid)
    cells = np.asarray(rows, dtype=np.uint8)
    legal = cells == EMPTY
    windows = list(iter_windows(rows))
    policy = np.zeros(cells.shape, dtype=np.float64)
    if not legal.any() or any(winner(window.state) != EMPTY for window in windows):
        return BoardPrediction(policy, np.zeros_like(legal), None, 0, True)
    counts = np.zeros(cells.shape, dtype=np.int32)
    used = 0
    for window in windows:
        # Fully occupied/forbidden windows provide no legal policy contribution.
        if not any(((window.state >> (2 * i)) & 3) == EMPTY for i in range(25)):
            continue
        logits, _ = infer_window(net, window, me)
        local_moves = np.flatnonzero(np.isfinite(logits))
        if not local_moves.size:
            continue
        weights = np.exp(logits[local_moves] - logits[local_moves].max())
        weights /= weights.sum()
        for index, weight in zip(local_moves, weights):
            coordinate = window.coordinates[int(index)]
            if coordinate is None:
                raise AssertionError("Forbidden padding escaped the policy mask")
            policy[coordinate] += float(weight)
            counts[coordinate] += 1
        used += 1
    if not np.all(counts[legal] > 0):
        raise RuntimeError("A playable cell was not covered by a window")
    policy[legal] /= counts[legal]
    policy /= policy.sum()
    best = int(policy.argmax())
    move = tuple(int(i) for i in np.unravel_index(best, policy.shape))
    return BoardPrediction(policy, legal, move, used, False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("param")
    parser.add_argument("bin")
    parser.add_argument("--board", required=True, help="JSON matrix: 0 empty, 1 black, 2 white, 3 forbidden")
    parser.add_argument("--side", required=True, choices=("black", "white"), help="actual board's player to move")
    args = parser.parse_args(argv)
    rows = validate_grid(json.loads(Path(args.board).read_text(encoding="utf-8")))
    me = BLACK if args.side == "black" else WHITE
    result = predict_board(load_net(args.param, args.bin), rows, me)
    print(format_grid(pad_board(rows)))
    print(json.dumps({"move": result.move, "window_count": result.window_count,
                      "terminal": result.terminal, "legal_mask": result.legal_mask.tolist(),
                      "policy": result.policy.tolist()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
