"""Validate actual browser Gomoku histories independently of search estimates."""
import math
from numbers import Integral, Real
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web_match_runner import file_hash, trajectory_key, winner

N = 16
CELL_COUNT = N * N


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _side(value, name="ai_side"):
    return _integer(value, name, 1, 2)


def _seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _required(value, name):
    if name not in value:
        raise ValueError(f"Missing {name}")
    return value[name]


def _cells(value, *, flattened):
    if not isinstance(value, list):
        raise ValueError("board must be a list")
    if flattened:
        if len(value) != CELL_COUNT:
            raise ValueError("browser board must contain exactly 256 cells")
        cells = value
    else:
        if len(value) != N or any(not isinstance(row, list) or len(row) != N for row in value):
            raise ValueError("normalized board must be 16x16")
        cells = [cell for row in value for cell in row]
    checked = [_integer(cell, "board cell", 0, 2) for cell in cells]
    return [checked[start:start + N] for start in range(0, CELL_COUNT, N)]


def _five_through(board, row, col, side):
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < N and 0 <= c < N and board[r][c] == side:
                count += 1
                r, c = r + sign * dr, c + sign * dc
        if count >= 5:
            return True
    return False


def _replay(indices, ai_side):
    if not isinstance(indices, list) or len(indices) > CELL_COUNT:
        raise ValueError("history must be a list with at most 256 moves")
    board = [[0] * N for _ in range(N)]
    history, ended = [], False
    for index, point in enumerate(indices):
        point = _integer(point, "history index", 0, CELL_COUNT - 1)
        row, col = divmod(point, N)
        if ended:
            raise ValueError("History continues after the game ended")
        if board[row][col]:
            raise ValueError("History repeats an occupied cell")
        side = 1 + index % 2
        board[row][col] = side
        history.append(dict(row=row, col=col, side=side, by="ai" if side == ai_side else "human"))
        ended = _five_through(board, row, col, side) or index + 1 == CELL_COUNT
    won = int(winner(board))
    terminal = bool(won or len(history) == CELL_COUNT)
    if terminal != ended:
        raise ValueError("Replay terminal detection disagrees with the full-board rules")
    return dict(board=board, history=history, winner=won, terminal=terminal, plies=len(history), ai_side=ai_side)


def verify_browser_state(raw, ai_side, seconds=1.0, require_human_turn=False):
    """Validate a read-only __gomokuSnapshot(), including incomplete AI replies.

    A pre-start empty board is valid. A requested playable human turn also needs
    started/ready and not busy. Analysis contents never establish a game result.
    The returned board and history are fresh copies with no trusted claim fields.
    """
    raw = _object(raw, "snapshot")
    ai_side = _side(ai_side)
    if not isinstance(require_human_turn, bool):
        raise ValueError("require_human_turn must be boolean")
    if _integer(_required(raw, "n"), "n", N, N) != N:
        raise ValueError("Acceptance requires n=16")
    # Legacy square snapshots omit cols; explicit metadata must still name
    # exactly 16 columns, even if a contradictory payload has 256 cells.
    _integer(raw.get("cols", raw["n"]), "cols", N, N)
    human = _side(_required(raw, "human"), "human")
    if human != 3 - ai_side:
        raise ValueError("Browser player identity differs from the configured AI side")
    expected = _seconds(seconds, "expected seconds")
    actual = _seconds(_required(raw, "seconds"), "browser seconds")
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Browser search seconds differ from the configured budget")
    for field in ("started", "ready", "busy"):
        if not isinstance(_required(raw, field), bool):
            raise ValueError(f"{field} must be boolean")
    _integer(_required(raw, "revision"), "revision", 0, 2**53 - 1)
    analysis = _required(raw, "analysis")
    if analysis is not None and not isinstance(analysis, dict):
        raise ValueError("analysis must be an object or null")
    # A visible position is admissible only after its durable commit succeeds.
    # Older snapshots omitted these fields; absence remains backwards compatible.
    if "storageBlocked" in raw:
        if not isinstance(raw["storageBlocked"], bool):
            raise ValueError("storageBlocked must be boolean")
        if raw["storageBlocked"]:
            raise ValueError("Browser storage is blocked")
    if "storageError" in raw:
        if raw["storageError"] is not None and not isinstance(raw["storageError"], str):
            raise ValueError("storageError must be a string or null")
        if raw["storageError"]:
            raise ValueError("Browser storage has an unresolved error")
    if "pendingCommit" in raw:
        if raw["pendingCommit"] is not None and not isinstance(raw["pendingCommit"], dict):
            raise ValueError("pendingCommit must be an object or null")
        if raw["pendingCommit"] is not None:
            raise ValueError("Browser has an uncommitted pending action")
    expected_board = _cells(_required(raw, "board"), flattened=True)
    result = _replay(_required(raw, "history"), ai_side)
    if result["board"] != expected_board:
        raise ValueError("Browser board differs from replayed history")
    if not raw["started"] and result["plies"]:
        raise ValueError("An unstarted game cannot contain played moves")
    if require_human_turn and not result["terminal"]:
        if not raw["started"] or not raw["ready"] or raw["busy"] or 1 + result["plies"] % 2 != human:
            raise ValueError("Snapshot is not a ready, idle human turn")
    return result


def _verified_normalized(state):
    state = _object(state, "normalized state")
    ai_side = _side(_required(state, "ai_side"))
    board = _cells(_required(state, "board"), flattened=False)
    history = _required(state, "history")
    if not isinstance(history, list):
        raise ValueError("Normalized history must be a list")
    indices, checked_history = [], []
    for move in history:
        move = _object(move, "history move")
        row = _integer(_required(move, "row"), "row", 0, N - 1)
        col = _integer(_required(move, "col"), "col", 0, N - 1)
        side = _side(_required(move, "side"), "history side")
        by = _required(move, "by")
        if by not in ("ai", "human"):
            raise ValueError("History actor must be ai or human")
        indices.append(row * N + col)
        checked_history.append(dict(row=row, col=col, side=side, by=by))
    result = _replay(indices, ai_side)
    if result["board"] != board or result["history"] != checked_history:
        raise ValueError("Normalized history has a wrong turn, actor or board")
    claimed_winner = _integer(_required(state, "winner"), "winner", 0, 2)
    claimed_plies = _integer(_required(state, "plies"), "plies", 0, CELL_COUNT)
    claimed_terminal = _required(state, "terminal")
    if not isinstance(claimed_terminal, bool):
        raise ValueError("terminal must be boolean")
    if (claimed_winner, claimed_plies, claimed_terminal) != (result["winner"], result["plies"], result["terminal"]):
        raise ValueError("Claimed state result differs from legal replay")
    return result


def verify_extension(before, after, human_move=None):
    """Validate one observed round; return the independently verified after-state.

    With a supplied click, require its human move and optionally the immediate
    AI reply. Without a click, allow unchanged snapshots, an AI response, or a
    human/AI round according to the actor who was due. A second round is rejected.
    """
    before, after = _verified_normalized(before), _verified_normalized(after)
    if before["ai_side"] != after["ai_side"]:
        raise ValueError("AI identity changed within a game")
    prefix = before["history"]
    if after["history"][:len(prefix)] != prefix or after["plies"] < before["plies"]:
        raise ValueError("History rolled back or replaced its complete prefix")
    added = after["history"][len(prefix):]
    if before["terminal"]:
        if added or human_move is not None:
            raise ValueError("Cannot extend a terminal game")
        return after
    human = 3 - before["ai_side"]
    next_side = 1 + before["plies"] % 2
    if human_move is not None:
        if not isinstance(human_move, (list, tuple)) or len(human_move) != 2:
            raise ValueError("human_move must be a row/column pair")
        row = _integer(human_move[0], "human row", 0, N - 1)
        col = _integer(human_move[1], "human col", 0, N - 1)
        if next_side != human or len(added) not in (1, 2):
            raise ValueError("Expected one human click and at most its AI reply")
        if added[0]["by"] != "human" or (added[0]["row"], added[0]["col"]) != (row, col):
            raise ValueError("The observed first move differs from the human click")
        if len(added) == 2 and added[1]["by"] != "ai":
            raise ValueError("Only the immediate AI response may follow the click")
    else:
        allowed = 2 if next_side == human else 1
        if len(added) > allowed:
            raise ValueError("The extension contains more than one round")
    return after


def verify_terminal_game(state):
    """Return a scored terminal game only after actual five/full-board replay."""
    verified = _verified_normalized(state)
    if not verified["terminal"]:
        raise ValueError("An unfinished game cannot be scored as a terminal outcome")
    won = verified["winner"]
    outcome = "draw" if not won else "win" if won == verified["ai_side"] else "loss"
    return dict(verified, outcome=outcome,
                canonical_trajectory=trajectory_key(verified["history"], verified["ai_side"]),
                termination="five_or_more" if won else "board_full")
