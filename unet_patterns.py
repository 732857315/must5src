"""Finite five-cell motif catalogue, separate from compound-board enumeration.

All 4**5 encodings are inspected. Motifs with at least three stones of one
color are retained, normalized by color reversal and line reversal. A line
touching the five-cell window edge says nothing about cells outside that
window; _XXX_ is therefore only a LOCAL open-three motif.
"""

from collections import Counter
from itertools import product


def _describe(cells):
    stones = cells.count(1)
    has_blocker = 2 in cells or 3 in cells
    positions = [i for i, cell in enumerate(cells) if cell == 1]
    consecutive = positions[-1] - positions[0] + 1 == len(positions)
    subtype = None
    if stones == 5:
        kind = "five_terminal"
    elif stones == 4:
        if 0 in cells:
            kind = "four_one_empty"
            subtype = "edge_four" if cells.index(0) in (0, 4) else "internal_gap_four"
        else:
            kind = "blocked_four"
            subtype = "opponent_blocked" if 2 in cells else "boundary_blocked"
    elif cells == (0, 1, 1, 1, 0):
        kind = "open_three_local"
    elif has_blocker:
        kind = "closed_three_local" if consecutive else "broken_three_blocked"
    else:
        kind = "edge_three_local" if consecutive else "broken_three"
    return dict(id="p_" + "".join(map(str, cells)), cells=list(cells),
                kind=kind, subtype=subtype, policy_eligible=stones < 5,
                can_complete_five_in_this_window=not has_blocker,
                scope="five-cell local motif")


def pattern_catalogue(include_terminal=True):
    unique = set()
    for raw in product(range(4), repeat=5):
        if max(raw.count(1), raw.count(2)) < 3:
            continue
        owner = 1 if raw.count(1) >= 3 else 2
        cells = tuple(1 if cell == owner else 2 if cell in (1, 2) else cell for cell in raw)
        unique.add(min(cells, cells[::-1]))
    rows = [_describe(cells) for cells in sorted(unique)]
    return rows if include_terminal else [row for row in rows if row["policy_eligible"]]


def catalogue_report():
    rows = pattern_catalogue()
    return dict(enumerated_line_encodings=4 ** 5, retained_patterns=len(rows),
                equivalence="line reversal and simultaneous color reversal",
                scope="all five-cell motifs with at least three stones of one color; not all compound boards",
                classes=dict(Counter(row["kind"] for row in rows)),
                patterns=[row for row in rows if row["policy_eligible"]],
                terminal_archive=[row for row in rows if not row["policy_eligible"]],
                notes=["_XXX_ is only locally open; outside-window cells are unknown.",
                       "OXXXX and #XXXX cannot complete five inside this window.",
                       "Terminal five-in-a-row entries are never action-training samples."])
