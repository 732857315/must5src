"""Shared 5x5 rules: 0 empty, 1 black, 2 white, 3 boundary/forbidden."""

N = 5
WIN = 5
CELL_COUNT = N * N
CENTER = (N // 2, N // 2)
EMPTY, BLACK, WHITE, FORBIDDEN = 0, 1, 2, 3


def cell_index(r, c):
    if not (0 <= r < N and 0 <= c < N):
        raise ValueError("Cell coordinates are outside the board")
    return r * N + c


CENTER_INDEX = cell_index(*CENTER)


def lines():
    for r in range(N):
        yield [(r, c) for c in range(N)]
    for c in range(N):
        yield [(r, c) for r in range(N)]
    yield [(i, i) for i in range(N)]
    yield [(N - 1 - i, i) for i in range(N)]


LINE_CELLS = list(lines())
LINE_INDICES = tuple(tuple(cell_index(r, c) for r, c in line) for line in LINE_CELLS)
_LINE_MASKS = tuple(sum(1 << (2 * i) for i in line) for line in LINE_INDICES)
_LINE_CHECKS = tuple((mask | (mask << 1), mask, mask << 1) for mask in _LINE_MASKS)
_CELL_LOW_BITS = sum(1 << (2 * i) for i in range(CELL_COUNT))
NEIGHBORS = tuple(
    tuple((r + dr) * N + c + dc
          for dr in (-1, 0, 1) for dc in (-1, 0, 1)
          if (dr, dc) != (0, 0) and 0 <= r + dr < N and 0 <= c + dc < N)
    for r in range(N) for c in range(N)
)
PERMS = tuple(
    tuple(fn(r, c)[0] * N + fn(r, c)[1] for r in range(N) for c in range(N))
    for fn in (
        lambda r, c: (r, c), lambda r, c: (c, N - 1 - r),
        lambda r, c: (N - 1 - r, N - 1 - c), lambda r, c: (N - 1 - c, r),
        lambda r, c: (N - 1 - r, c), lambda r, c: (r, N - 1 - c),
        lambda r, c: (c, r), lambda r, c: (N - 1 - c, N - 1 - r),
    )
)


def validate_state(s):
    """Validate encoding, without claiming the position has a legal game history."""
    if not isinstance(s, int) or isinstance(s, bool) or not 0 <= s < (1 << (2 * CELL_COUNT)):
        raise ValueError("State must be a nonnegative 50-bit integer")


def other_player(player):
    if player not in (BLACK, WHITE):
        raise ValueError("Player must be BLACK or WHITE")
    return BLACK + WHITE - player


def stone_count(s):
    """Count black/white stones; boundary and forbidden cells are not moves."""
    return ((s ^ (s >> 1)) & _CELL_LOW_BITS).bit_count()


def empty_count(s):
    """Count playable empty cells, excluding stones and forbidden cells."""
    return CELL_COUNT - ((s | (s >> 1)) & _CELL_LOW_BITS).bit_count()


def player_at(s):
    return BLACK if stone_count(s) % 2 == 0 else WHITE


def initial_state():
    """Black has already played the mandatory center opening; white is next."""
    return BLACK << (2 * CENTER_INDEX)


def legal_moves(s):
    """Return empty cells; search callers check terminal states separately."""
    return [i for i in range(CELL_COUNT) if ((s >> (2 * i)) & 3) == EMPTY]


def forbidden_moves(s):
    """Return cells marked as boundary/forbidden (3), in action-index order."""
    return [i for i in range(CELL_COUNT) if ((s >> (2 * i)) & 3) == FORBIDDEN]


def winner(s):
    for cells_mask, black_pattern, white_pattern in _LINE_CHECKS:
        pattern = s & cells_mask
        if pattern == black_pattern:
            return BLACK
        if pattern == white_pattern:
            return WHITE
    return EMPTY


def move_rejection_reason(s, idx):
    """Return a Chinese explanation for a prohibited move, or None if playable."""
    validate_state(s)
    if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < CELL_COUNT:
        return "禁下：落点越界或索引无效，落子索引必须是 0 到 24 的整数"
    cell = (s >> (2 * idx)) & 3
    if cell == FORBIDDEN:
        return "禁下：该格为边界或禁下格，不能落子"
    if cell != EMPTY:
        return "禁下：该格已有棋子，不能重复落子"
    if winner(s) != EMPTY:
        return "禁下：棋局已结束，不能继续落子"
    return None


def apply_move(s, idx, player):
    """Checked move at game boundaries; search can use its already checked moves."""
    other_player(player)
    reason = move_rejection_reason(s, idx)
    if reason is not None:
        raise ValueError(reason)
    return s | (player << (2 * idx))


def winning_moves(s, player):
    return [i for i in legal_moves(s) if winner(s | (player << (2 * i))) == player]


def ordered_moves(s, me):
    """Partition empty cells into immediate wins, blocks, and neighboring/rest moves."""
    opp = other_player(me)
    win, blocks, touch, rest = [], [], [], []
    for i in legal_moves(s):
        if winner(s | (me << (2 * i))) == me:
            win.append(i)
        elif winner(s | (opp << (2 * i))) == opp:
            blocks.append(i)
        elif any(((s >> (2 * j)) & 3) == me for j in NEIGHBORS[i]):
            touch.append(i)
        else:
            rest.append(i)
    return win, blocks, touch + rest


class Board:
    __slots__ = ("grid",)

    def __init__(self, grid=None):
        self.grid = bytearray(CELL_COUNT) if grid is None else bytearray(grid)
        if len(self.grid) != CELL_COUNT or any(v not in (EMPTY, BLACK, WHITE, FORBIDDEN) for v in self.grid):
            raise ValueError("Board requires 25 cells containing 0, 1, 2, or 3")

    def copy(self):
        return Board(self.grid)

    def get(self, r, c):
        return self.grid[cell_index(r, c)]

    def set(self, r, c, player):
        if player not in (EMPTY, BLACK, WHITE, FORBIDDEN):
            raise ValueError("Cell value must be 0, 1, 2, or 3")
        self.grid[cell_index(r, c)] = player

    def legal_moves(self):
        return [i for i, value in enumerate(self.grid) if value == EMPTY]

    def winner(self):
        return winner(self.pack())

    def is_terminal(self):
        return self.winner() != EMPTY or all(self.grid)

    def apply(self, idx, player):
        return Board.unpack(apply_move(self.pack(), idx, player))

    def pack(self):
        return sum(value << (2 * i) for i, value in enumerate(self.grid))

    @staticmethod
    def unpack(v):
        validate_state(v)
        return Board([(v >> (2 * i)) & 3 for i in range(CELL_COUNT)])

    def show(self, *, show_boundary=False, show_legend=False):
        """Render # for forbidden cells; optional outer # frame is display only."""
        chars = {EMPTY: ".", BLACK: "X", WHITE: "O", FORBIDDEN: "#"}
        rows = ["".join(chars[v] for v in self.grid[r * N:(r + 1) * N]) for r in range(N)]
        if show_boundary:
            frame = "#" * (N + 2)
            rows = [frame, *("#" + row + "#" for row in rows), frame]
        if show_legend:
            rows.append("图例：. 空位，X 黑棋，O 白棋，# 边界/禁下")
        return "\n".join(rows)
