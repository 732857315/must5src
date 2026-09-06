"""Global, overlap-deduplicated structural threats for either Gomoku color.

These are motif facts, not a proof of a forced win or a Renju legality rule.
Every core uses global coordinates, so overlapping neural windows cannot count
the same line pattern twice. Fixed forbidden cells and board edges break lines.
"""

from collections import Counter
from numbers import Integral

import numpy as np

from board_rules import BLACK, WHITE, EMPTY, DIRECTIONS, normalize_board, board_winner


def _side(side):
    if isinstance(side, bool) or not isinstance(side, Integral) or side not in (BLACK, WHITE):
        raise ValueError("行棋方必须为1黑棋或2白棋")
    return int(side)


def _windows(board, length):
    height, width = board.shape
    for direction in DIRECTIONS:
        dr, dc = direction
        for row in range(height):
            for col in range(width):
                end_r = row + (length - 1) * dr
                end_c = col + (length - 1) * dc
                if not (0 <= end_r < height and 0 <= end_c < width):
                    continue
                cells = tuple((row + step * dr, col + step * dc) for step in range(length))
                yield direction, cells, tuple(int(board[cell]) for cell in cells)


def _empty_result(side, winner, terminal):
    return dict(side=side, winner=winner, terminal=terminal, structural_only=True,
                fours=[], threes=[], four_count=0, three_count=0,
                double_four=False, double_three=False, open_four=False,
                winning_cells=[], double_kill=False,
                four_completion_groups=[], shared_completion_cells=[],
                completion_core_counts=[], four_defense_cells=[],
                three_extension_groups=[], three_defense_cells=[])


def analyze_threats(grid, side):
    """Return four/three cores and their concrete global continuation cells.

    Four: five-cell window with four own stones and one empty. Adjacent
    windows with the same direction and four stones merge their completion
    cells, so .XXXX. is one open four with two winning cells.

    Three: six-cell window with empty endpoints and three own stones plus one
    empty in its middle four cells. Filling that internal gap makes a straight
    four with both ends open. Identical three-stone cores merge extensions.

    double_four counts cores; double_kill counts distinct immediate winning
    cells. Different four cores can share a single winning cell.
    four_defense_cells lists single placements blocking every current
    immediate winning cell, excluding the defender's own winning replies.

    Each three's upgrade_paths records an empty endpoint pair and the extension
    gap. A defender blocks that route by occupying any of those three cells.
    Three defense cells intersect route blockers, first per core and then over
    every identified core. An empty intersection does NOT prove a forced loss.
    """
    side = _side(side)
    board = normalize_board(grid)
    result = board_winner(board)
    terminal = result != EMPTY or not bool(np.any(board == EMPTY))
    report = _empty_result(side, result, terminal)
    if terminal:
        return report

    fours = {}
    for direction, cells, values in _windows(board, 5):
        if values.count(side) != 4 or values.count(EMPTY) != 1:
            continue
        stones = tuple(cell for cell, value in zip(cells, values) if value == side)
        key = (direction, stones)
        fours.setdefault(key, set()).add(cells[values.index(EMPTY)])

    threes = {}
    for direction, cells, values in _windows(board, 6):
        if values[0] != EMPTY or values[-1] != EMPTY:
            continue
        middle = values[1:-1]
        if middle.count(side) != 3 or middle.count(EMPTY) != 1:
            continue
        stones = tuple(cell for cell, value in zip(cells, values) if value == side)
        key = (direction, stones)
        extension = cells[1 + middle.index(EMPTY)]
        endpoints = (cells[0], cells[-1])
        path_key = (extension, endpoints)
        threes.setdefault(key, {})[path_key] = dict(
            extension_cell=extension, empty_endpoints=list(endpoints),
            blockers=sorted((*endpoints, extension)))

    four_records = [
        dict(direction=direction, stones=list(stones), completion_cells=sorted(completions),
             open_four=len(completions) >= 2)
        for (direction, stones), completions in sorted(fours.items())
    ]
    three_records = []
    all_upgrade_paths = []
    for (direction, stones), paths in sorted(threes.items()):
        upgrades = [path for _, path in sorted(paths.items())]
        defenses = set(upgrades[0]["blockers"])
        for path in upgrades[1:]:
            defenses.intersection_update(path["blockers"])
        three_records.append(dict(
            direction=direction, stones=list(stones),
            extension_cells=sorted({path["extension_cell"] for path in upgrades}),
            upgrade_paths=upgrades, defense_cells=sorted(defenses)))
        all_upgrade_paths.extend(upgrades)
    three_defenses = set(all_upgrade_paths[0]["blockers"]) if all_upgrade_paths else set()
    for path in all_upgrade_paths[1:]:
        three_defenses.intersection_update(path["blockers"])
    completion_counts = Counter(cell for completions in fours.values() for cell in completions)
    winning = sorted(completion_counts)
    report.update(
        fours=four_records, threes=three_records,
        four_count=len(four_records), three_count=len(three_records),
        double_four=len(four_records) >= 2, double_three=len(three_records) >= 2,
        open_four=any(record["open_four"] for record in four_records),
        winning_cells=winning, double_kill=len(winning) >= 2,
        four_completion_groups=[record["completion_cells"] for record in four_records],
        shared_completion_cells=sorted(cell for cell, count in completion_counts.items() if count >= 2),
        completion_core_counts=[dict(cell=cell, core_count=completion_counts[cell]) for cell in winning],
        four_defense_cells=winning.copy() if len(winning) == 1 else [],
        three_extension_groups=[record["extension_cells"] for record in three_records],
        three_defense_cells=sorted(three_defenses),
    )
    return report
