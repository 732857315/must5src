"""5x5 local views with explicit 2-bit forbidden padding and global coordinates."""

from dataclasses import dataclass
from numbers import Integral

from game import N, EMPTY, BLACK, WHITE, FORBIDDEN, Board, move_rejection_reason


WINDOW_RADIUS = N // 2


def validate_grid(grid):
    """Copy a nonempty rectangular board. 3 is a fixed forbidden cell, not a stone."""
    rows = tuple(tuple(row) for row in grid)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Board must be a nonempty rectangular grid")
    if any(isinstance(cell, bool) or not isinstance(cell, Integral) or cell not in (EMPTY, BLACK, WHITE, FORBIDDEN)
           for row in rows for cell in row):
        raise ValueError("Board cells must be integers 0, 1, 2, or 3 (forbidden)")
    return tuple(tuple(int(cell) for cell in row) for row in rows)


def pad_board(grid, width=WINDOW_RADIUS):
    """Surround the actual board with 3s; radius 2 covers every centered 5x5 window."""
    rows = validate_grid(grid)
    if isinstance(width, bool) or not isinstance(width, Integral) or width < 0:
        raise ValueError("Padding width must be a nonnegative integer")
    width = int(width)
    edge = (FORBIDDEN,) * width
    full_edge = (FORBIDDEN,) * (len(rows[0]) + 2 * width)
    return (full_edge,) * width + tuple(edge + row + edge for row in rows) + (full_edge,) * width


@dataclass(frozen=True)
class BoardWindow:
    state: int
    center: tuple[int, int]
    coordinates: tuple[tuple[int, int] | None, ...]

    @property
    def grid(self):
        cells = Board.unpack(self.state).grid
        return tuple(tuple(cells[r * N:(r + 1) * N]) for r in range(N))

    def move_hint(self, index):
        if isinstance(index, Integral) and not isinstance(index, bool):
            index = int(index)
        return move_rejection_reason(self.state, index) or "可落子"

    def to_global(self, index):
        """Map a playable local action to the real board; reject padding explicitly."""
        if isinstance(index, Integral) and not isinstance(index, bool):
            index = int(index)
        reason = move_rejection_reason(self.state, index)
        if reason is not None:
            raise ValueError(reason)
        coordinate = self.coordinates[index]
        if coordinate is None:
            raise ValueError("禁下：窗口边界不属于真实棋盘")
        return coordinate


def _extract(rows, row, col):
    height, width = len(rows), len(rows[0])
    cells, coordinates = [], []
    for local_row in range(N):
        real_row = row + local_row - WINDOW_RADIUS
        for local_col in range(N):
            real_col = col + local_col - WINDOW_RADIUS
            if 0 <= real_row < height and 0 <= real_col < width:
                cells.append(rows[real_row][real_col])
                coordinates.append((real_row, real_col))
            else:
                cells.append(FORBIDDEN)
                coordinates.append(None)
    return BoardWindow(Board(cells).pack(), (row, col), tuple(coordinates))


def extract_window(grid, row, col):
    """Extract a fixed 5x5 view centered on an actual board cell, without wraparound."""
    rows = validate_grid(grid)
    if (isinstance(row, bool) or isinstance(col, bool)
            or not isinstance(row, Integral) or not isinstance(col, Integral)
            or not 0 <= row < len(rows) or not 0 <= col < len(rows[0])):
        raise ValueError("Window center must be inside the actual board")
    return _extract(rows, int(row), int(col))


def iter_windows(grid):
    """Cover every actual cell, including corners and edges; no windows are skipped."""
    rows = validate_grid(grid)
    for row in range(len(rows)):
        for col in range(len(rows[0])):
            yield _extract(rows, row, col)


def format_grid(grid, *, show_legend=True):
    rows = validate_grid(grid)
    chars = {EMPTY: ".", BLACK: "X", WHITE: "O", FORBIDDEN: "#"}
    text = "\n".join(" ".join(chars[cell] for cell in row) for row in rows)
    if show_legend:
        text += "\n图例：. 可落子；X 黑棋；O 白棋；# 边界/禁下（2bit=11）"
    return text
