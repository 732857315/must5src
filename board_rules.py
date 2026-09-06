"""Full-board Gomoku rules for arbitrary rectangular 0/1/2/3 grids.

0 is empty, 1 black, 2 white, and 3 fixed forbidden/boundary. Five or more
contiguous stones win. A forbidden cell breaks every line through it.
Opening placement and whose turn it is are decisions of the caller.
"""

from numbers import Integral

import numpy as np


EMPTY, BLACK, WHITE, FORBIDDEN = 0, 1, 2, 3
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def normalize_board(grid):
    """Validate the encoding and return an independent C-order uint8 copy."""
    try:
        board = np.asarray(grid)
    except (TypeError, ValueError) as exc:
        raise ValueError("棋盘必须是规则的二维整数数组") from exc
    if board.ndim != 2 or not all(board.shape):
        raise ValueError("棋盘必须是非空二维数组")
    if board.dtype.kind not in "iu":
        raise ValueError("棋盘格必须使用整数：0空位、1黑棋、2白棋、3边界或禁下")
    if np.any(board < EMPTY) or np.any(board > FORBIDDEN):
        raise ValueError("棋盘格只能为0空位、1黑棋、2白棋、3边界或禁下")
    return np.array(board, dtype=np.uint8, order="C", copy=True)


def _side(side):
    if isinstance(side, bool) or not isinstance(side, Integral) or side not in (BLACK, WHITE):
        raise ValueError("行棋方必须为1黑棋或2白棋")
    return int(side)


def _winner(board):
    height, width = board.shape
    for row in range(height):
        for col in range(width):
            side = int(board[row, col])
            if side not in (BLACK, WHITE):
                continue
            for dr, dc in DIRECTIONS:
                before_r, before_c = row - dr, col - dc
                if 0 <= before_r < height and 0 <= before_c < width and board[before_r, before_c] == side:
                    continue
                length = 1
                r, c = row + dr, col + dc
                while 0 <= r < height and 0 <= c < width and board[r, c] == side:
                    length += 1
                    if length >= 5:
                        return side
                    r, c = r + dr, c + dc
    return EMPTY


def board_winner(grid):
    """Return BLACK/WHITE for a contiguous run of at least five, else EMPTY."""
    return _winner(normalize_board(grid))


def _empty_cells(board):
    return [(int(row), int(col)) for row, col in np.argwhere(board == EMPTY)]


def legal_cells(grid):
    """Return empty coordinates in row order; an ended game has no actions."""
    board = normalize_board(grid)
    return [] if _winner(board) != EMPTY else _empty_cells(board)


def _wins_at(board, row, col, side):
    """Check a hypothetical stone using both directions, without mutating."""
    height, width = board.shape
    for dr, dc in DIRECTIONS:
        length = 1
        for direction in (-1, 1):
            r, c = row + direction * dr, col + direction * dc
            while 0 <= r < height and 0 <= c < width and board[r, c] == side:
                length += 1
                if length >= 5:
                    return True
                r += direction * dr
                c += direction * dc
    return False


def _winning_cells(board, side, moves):
    return [(row, col) for row, col in moves if _wins_at(board, row, col, side)]


def winning_cells(grid, side):
    """All legal placements that immediately make >=5, checked globally."""
    side = _side(side)
    board = normalize_board(grid)
    if _winner(board) != EMPTY:
        return []
    return _winning_cells(board, side, _empty_cells(board))


def _coordinate(move):
    try:
        row, col = move
    except (TypeError, ValueError):
        return None
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (row, col)):
        return None
    return int(row), int(col)


def _rejection_reason(board, move):
    coordinate = _coordinate(move)
    if coordinate is None:
        return "禁下：落点越界或坐标无效，必须提供两个整数(row, col)"
    row, col = coordinate
    height, width = board.shape
    if not (0 <= row < height and 0 <= col < width):
        return "禁下：落点越界，超出整盘范围"
    cell = board[row, col]
    if cell == FORBIDDEN:
        return "禁下：该格为边界或禁下格，不能落子"
    if cell != EMPTY:
        return "禁下：该格已有棋子，不能重复落子"
    if _winner(board) != EMPTY:
        return "禁下：棋局已结束，不能继续落子"
    return None


def board_move_rejection_reason(grid, move):
    """Return a specific Chinese rejection reason, or None for a legal move."""
    return _rejection_reason(normalize_board(grid), move)


def apply_board_move(grid, move, side):
    """Return a new board; illegal moves raise ValueError with a Chinese reason."""
    side = _side(side)
    board = normalize_board(grid)
    # Materialize a coordinate once so one-shot iterators are handled safely.
    coordinate = _coordinate(move)
    reason = _rejection_reason(board, coordinate)
    if reason is not None:
        raise ValueError(reason)
    board[coordinate] = side
    return board


def tactical_candidates(grid, side):
    """Partition legal cells into immediate wins, opponent wins, and the rest.

    Returned groups contain exact one-move facts, not heuristic predictions.
    A winning cell takes priority when it also blocks an opponent win. Multiple
    remaining opponent winning cells are separate threats; no single blocking
    action covers them all when the current player has no immediate win.
    """
    side = _side(side)
    board = normalize_board(grid)
    if _winner(board) != EMPTY:
        return [], [], []
    moves = _empty_cells(board)
    wins = _winning_cells(board, side, moves)
    own = set(wins)
    blocks = [move for move in _winning_cells(board, BLACK + WHITE - side, moves) if move not in own]
    urgent = own | set(blocks)
    return wins, blocks, [move for move in moves if move not in urgent]


def swap_board_colors(grid):
    """Swap black/white while preserving empty and forbidden cells."""
    board = normalize_board(grid)
    return np.array([EMPTY, WHITE, BLACK, FORBIDDEN], dtype=np.uint8)[board]
