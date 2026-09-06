"""Synthetic curriculum positions and auditable expert policies; no training.

Stages: 1 plain, 2 a fixed forbidden border, 3 arbitrary fixed forbidden cells.
Exact results are distinguished from depth-two heuristic estimates. Tactical
templates, sampled endgames and early/middle positions are not an enumeration
of the whole game. Colors/turns have legal counts; the side is stored explicitly.
"""

from collections import Counter
from numbers import Integral, Real
import math
import random

import numpy as np

from arena import PlayerAB
from game import BLACK, WHITE, EMPTY, FORBIDDEN, CELL_COUNT, CENTER_INDEX, LINE_INDICES
from game import empty_count, legal_moves, ordered_moves, player_at, winner
from unet_data import canonical_key
from unet_patterns import pattern_catalogue, catalogue_report


_STAGE_NAMES = {"plain": 1, "edge": 2, "random": 3}


def _stage_number(stage):
    stage = _STAGE_NAMES.get(stage, stage)
    if isinstance(stage, bool) or not isinstance(stage, Integral) or stage not in (1, 2, 3):
        raise ValueError("stage must be 1/plain, 2/edge, or 3/random")
    return int(stage)


def _mask(stage, rng, keep=(), required_free=1, count=None):
    keep = set(keep)
    if stage == 1:
        return set()
    if stage == 2:
        choices = []
        for thickness in (1, 2):
            for side in range(4):
                mask = {i for i in range(CELL_COUNT)
                        if (i // 5 < thickness if side == 0 else
                            i // 5 >= 5 - thickness if side == 1 else
                            i % 5 < thickness if side == 2 else
                            i % 5 >= 5 - thickness)}
                if not mask & keep and CELL_COUNT - len(mask) >= required_free:
                    choices.append(mask)
        return rng.choice(choices) if choices else None
    available = [i for i in range(CELL_COUNT) if i not in keep]
    maximum = min(len(available), CELL_COUNT - required_free)
    if maximum < 1:
        return None
    amount = rng.randint(1, maximum) if count is None else count
    if amount > maximum:
        return None
    return set(rng.sample(available, amount))


def _pack(black, white, forbidden):
    return (sum(BLACK << (2 * i) for i in black)
            | sum(WHITE << (2 * i) for i in white)
            | sum(FORBIDDEN << (2 * i) for i in forbidden))


def _random_position(stage, rng, ordinal, late, side, mask_count=None):
    mask = _mask(stage, rng, count=mask_count)
    if mask is None:
        return None
    free = [i for i in range(CELL_COUNT) if i not in mask]
    if late:
        count = len(free) - rng.randint(1, min(5, len(free)))
    else:
        count = rng.randint(0, len(free) - 1)
    parity = 0 if side == BLACK else 1
    if count % 2 != parity:
        count += 1 if count + 1 < len(free) else -1
    if count < 0:
        return None
    moves = rng.sample(free, count)
    if stage == 1 and count:
        # The unmasked complete-board random trajectory begins at center.
        moves = [CENTER_INDEX] + rng.sample([i for i in free if i != CENTER_INDEX], count - 1)
    state = _pack(moves[::2], moves[1::2], mask)
    if winner(state) != EMPTY:
        return None
    return state, side, "random_endgame" if late else "random_position"


def _tactical_position(stage, rng, ordinal, mode, side):
    line = LINE_INDICES[(ordinal // 2) % len(LINE_INDICES)]
    if mode == "loss":
        pairs = [(a, b) for a in LINE_INDICES for b in LINE_INDICES if not set(a) & set(b)]
        first, second = rng.choice(pairs)
        first_gap, second_gap = rng.choice(first), rng.choice(second)
        attack = set(first) - {first_gap} | (set(second) - {second_gap})
        keep = set(first) | set(second)
        attacker = 3 - side
        own_count = len(attack) if side == BLACK else len(attack) - 1
    else:
        gap = rng.choice(line)
        attack = set(line) - {gap}
        keep = set(line)
        attacker = side if mode == "win" else 3 - side
        # Counts at a BLACK turn are equal; at WHITE, black has one extra.
        own_count = (len(attack) if attacker == WHITE else len(attack) - 1)
        if side == BLACK:
            own_count = len(attack)
        elif mode == "win":
            own_count = len(attack) + 1
    required = len(attack) + own_count + (2 if mode == "loss" else 1)
    mask = _mask(stage, rng, keep, required_free=required)
    if mask is None:
        return None
    available = [i for i in range(CELL_COUNT) if i not in mask and i not in keep]
    if len(available) < own_count:
        return None
    filler = set(rng.sample(available, own_count))
    black, white = (attack, filler) if attacker == BLACK else (filler, attack)
    state = _pack(black, white, mask)
    if winner(state) != EMPTY or player_at(state) != side:
        return None
    wins, blocks, _ = ordered_moves(state, side)
    if mode == "win" and not wins:
        return None
    if mode == "block" and (wins or len(blocks) != 1):
        return None
    if mode == "loss" and (wins or len(blocks) < 2):
        return None
    return state, side, "tactical_" + mode


def _edge_masks():
    return [{i for i in range(CELL_COUNT)
             if (i // 5 < thickness if edge == 0 else
                 i // 5 >= 5 - thickness if edge == 1 else
                 i % 5 < thickness if edge == 2 else i % 5 >= 5 - thickness)}
            for thickness in (1, 2) for edge in range(4)]


def feasible_pattern_catalogue(stage):
    """Motifs compatible with the stage's fixed-mask geometry, before filling."""
    stage = _stage_number(stage)
    feasible = []
    for pattern in pattern_catalogue(include_terminal=False):
        cells = pattern["cells"]
        if stage == 1 and FORBIDDEN in cells:
            continue
        if stage == 2:
            fits = False
            for line in LINE_INDICES:
                forbidden = {i for i, cell in zip(line, cells) if cell == FORBIDDEN}
                unblocked = set(line) - forbidden
                if any(forbidden <= mask and not unblocked & mask for mask in _edge_masks()):
                    fits = True
                    break
            if not fits:
                continue
        feasible.append(pattern)
    return feasible


def _pattern_position(stage, rng, pattern, ordinal, side):
    attack_side = BLACK if ordinal % 2 == 0 else WHITE
    cells = [attack_side if cell == 1 else 3 - attack_side if cell == 2 else cell
             for cell in pattern["cells"]]
    black_count, white_count = cells.count(BLACK), cells.count(WHITE)
    if side == BLACK:
        black_total = white_total = max(black_count, white_count)
    else:
        white_total = max(white_count, black_count - 1)
        black_total = white_total + 1
    needed = black_total + white_total + max(1, cells.count(EMPTY))
    line_indices = list(range(len(LINE_INDICES)))
    rng.shuffle(line_indices)
    for line_index in line_indices:
        line = LINE_INDICES[line_index]
        required_mask = {i for i, cell in zip(line, cells) if cell == FORBIDDEN}
        unblocked = set(line) - required_mask
        if stage == 1:
            masks = [set()] if not required_mask else []
        elif stage == 2:
            masks = [mask for mask in _edge_masks()
                     if required_mask <= mask and not unblocked & mask
                     and CELL_COUNT - len(mask) >= needed]
            rng.shuffle(masks)
        else:
            maximum = CELL_COUNT - needed
            minimum = max(1, len(required_mask))
            if maximum < minimum:
                continue
            amount = rng.randint(minimum, maximum)
            available_mask = [i for i in range(CELL_COUNT) if i not in set(line)]
            masks = [required_mask | set(rng.sample(available_mask, amount - len(required_mask)))]
        for mask in masks:
            free = [i for i in range(CELL_COUNT) if i not in mask and i not in line]
            add_black, add_white = black_total - black_count, white_total - white_count
            if len(free) < add_black + add_white:
                continue
            for _ in range(8):
                filler = rng.sample(free, add_black + add_white)
                black = {i for i, cell in zip(line, cells) if cell == BLACK} | set(filler[:add_black])
                white = {i for i, cell in zip(line, cells) if cell == WHITE} | set(filler[add_black:])
                state = _pack(black, white, mask)
                if winner(state) != EMPTY or not legal_moves(state):
                    continue
                wins, blocks, _ = ordered_moves(state, side)
                # Extra count-balancing stones must not create another urgent
                # line that would hide the intended motif.
                if any(move not in line for move in wins + blocks):
                    continue
                return state, side, "pattern", pattern["id"], pattern["kind"], line_index
    return None


def _uniform(moves):
    target = np.zeros(CELL_COUNT, dtype=np.float32)
    target[moves] = 1.0 / len(moves)
    return target


def base_key(state):
    """One physical board identity storing separate labels for both actors."""
    return min(canonical_key(state, BLACK), canonical_key(state, WHITE))


def _exact_analysis(state, side):
    """Small endgame oracle with an explicit actor, including counterfactuals."""
    if empty_count(state) > 5:
        raise ValueError("the explicit endgame oracle is limited to five empty cells")
    table = {}

    def value(position, actor):
        key = (position, actor)
        if key in table:
            return table[key]
        result = winner(position)
        if result != EMPTY:
            answer = 1 if result == actor else -1
        else:
            moves = legal_moves(position)
            if not moves:
                answer = 0
            else:
                answer = -1
                for move in moves:
                    child = position | (actor << (2 * move))
                    answer = max(answer, -value(child, 3 - actor))
                    if answer == 1:
                        break
        table[key] = answer
        return answer

    best_value = value(state, side)
    if winner(state) != EMPTY:
        return best_value, []
    best_moves = [move for move in legal_moves(state)
                  if -value(state | (side << (2 * move)), 3 - side) == best_value]
    return best_value, best_moves


def _validate_teacher_temperature(teacher_temperature):
    if (isinstance(teacher_temperature, bool) or not isinstance(teacher_temperature, Real)
            or not math.isfinite(teacher_temperature) or teacher_temperature <= 0):
        raise ValueError("teacher_temperature must be a finite positive number")
    return float(teacher_temperature)


def _teach(state, side, searcher, teacher_temperature=0.025):
    teacher_temperature = _validate_teacher_temperature(teacher_temperature)
    moves = legal_moves(state)
    wins, blocks, _ = ordered_moves(state, side)
    if wins:
        return _uniform(wins), 1, "exact"
    if len(blocks) > 1:
        return _uniform(blocks), -1, "exact"
    if empty_count(state) <= 5:
        value, best = _exact_analysis(state, side)
        return _uniform(best), value, "exact"
    if state == 0:
        return _uniform([CENTER_INDEX]), 0.0, "heuristic"
    searcher.cache.clear()
    if blocks:
        move = blocks[0]
        value = -searcher._ab(state | (side << (2 * move)), 3 - side, 1)
        return _uniform(blocks), value, "exact" if abs(value) == 1 else "heuristic"
    values = np.array([-searcher._ab(state | (side << (2 * move)), 3 - side, 1)
                       for move in moves], dtype=np.float64)
    value = float(values.max())
    if value == 1:
        return _uniform([move for move, score in zip(moves, values) if score == 1]), 1, "exact"
    # The bounded line score ranks quieter moves without pretending to solve them.
    weights = np.exp((values - values.max()) / teacher_temperature)
    target = np.zeros(CELL_COUNT, dtype=np.float32)
    target[moves] = (weights / weights.sum()).astype(np.float32)
    return target, value, "exact" if value == -1 else "heuristic"


def build_curriculum_dataset(stage, sample_count, seed=0, exclude_keys=None, teacher_temperature=0.025):
    """Return unique expert examples, removing D4 and color-reversal duplicates.

    Each dictionary has state/side/target/value/value_kind/outcome/source/stage
    and key/mask_count plus counter_* labels recomputed for the other actor.
    The key identifies the base board, regardless of actor. Each target is
    supported only on legal actions of a
    nonterminal position. Exact +/-1 are proved results; heuristic values must
    not be counted as known wins, losses or draws. Temperature only sharpens
    quiet AB score distributions; their ranking, ties and exact labels stay
    unchanged. Pass 0.15 to reproduce the original softer teacher.
    """
    stage = _stage_number(stage)
    teacher_temperature = _validate_teacher_temperature(teacher_temperature)
    if isinstance(sample_count, bool) or not isinstance(sample_count, Integral) or sample_count < 0:
        raise ValueError("sample_count must be a nonnegative integer")
    if sample_count == 0:
        return []
    rng = random.Random(seed)
    seen = set() if exclude_keys is None else set(exclude_keys)
    rows = []
    searcher = PlayerAB(None, "curriculum-depth2", depth=2)
    modes = ("pattern", "pattern", "pattern", "win", "block", "loss", "late", "late", "general")
    patterns = feasible_pattern_catalogue(stage)
    pattern_ordinal = 0
    attempts = 0
    tactical_ordinals = {"win": 0, "block": 0, "loss": 0}
    # Stage 3 deliberately visits every feasible forbidden count before mixing.
    bootstrap = list(range(1, 25)) if stage == 3 else []
    while len(rows) < sample_count and attempts < max(1000, sample_count * 100):
        ordinal = attempts
        attempts += 1
        requested_side = BLACK if len(rows) % 2 == 0 else WHITE
        if bootstrap:
            amount = bootstrap[0]
            side = BLACK if amount == 24 else requested_side
            candidate = _random_position(stage, rng, ordinal, True, side, amount)
        else:
            mode = modes[ordinal % len(modes)]
            if mode == "pattern":
                pattern = patterns[pattern_ordinal % len(patterns)]
                candidate = _pattern_position(stage, rng, pattern, pattern_ordinal, requested_side)
                pattern_ordinal += 1
            elif mode in ("win", "block", "loss"):
                # Each kind independently cycles all 12 lines and both colors;
                # coupling line indices to the six-way mixture skips 8 lines.
                recipe = tactical_ordinals[mode]
                tactical_ordinals[mode] += 1
                tactical_side = BLACK if recipe % 2 == 0 else WHITE
                candidate = _tactical_position(stage, rng, recipe, mode, tactical_side)
            else:
                candidate = _random_position(stage, rng, ordinal, mode == "late", requested_side)
        if candidate is None:
            continue
        state, side, source, *pattern_details = candidate
        key = base_key(state)
        if key in seen:
            if bootstrap and attempts % 100 == 0:
                bootstrap.pop(0)
            continue
        target, value, kind = _teach(state, side, searcher, teacher_temperature)
        counter_target, counter_value, counter_kind = _teach(state, 3 - side, searcher, teacher_temperature)
        if source.startswith("random_") and kind == "exact":
            source = "exact_endgame" if empty_count(state) <= 5 else "tactical_random"
        outcome = ("win" if value == 1 else "loss" if value == -1 else "draw") if kind == "exact" else "unknown"
        counter_outcome = ("win" if counter_value == 1 else "loss" if counter_value == -1 else "draw") if counter_kind == "exact" else "unknown"
        rows.append(dict(state=state, side=side, target=target, value=value,
                         counter_target=counter_target, counter_value=counter_value,
                         counter_value_kind=counter_kind, counter_outcome=counter_outcome,
                         value_kind=kind, outcome=outcome, source=source, stage=stage,
                         pattern_id=pattern_details[0] if pattern_details else None,
                         pattern=pattern_details[1] if pattern_details else None,
                         pattern_line=pattern_details[2] if pattern_details else None,
                         key=key, mask_count=sum(((state >> (2 * i)) & 3) == FORBIDDEN for i in range(CELL_COUNT))))
        seen.add(key)
        if bootstrap:
            bootstrap.pop(0)
    if len(rows) != sample_count:
        raise RuntimeError(f"could generate only {len(rows)} distinct examples after {attempts} attempts")
    return rows


def dataset_report(rows):
    """JSON-friendly observed counts; heuristic zeros are never reported as draws."""
    report = {field: dict(Counter(str(row.get(field)) for row in rows))
              for field in ("side", "outcome", "value_kind", "counter_outcome", "counter_value_kind", "source", "mask_count", "pattern")}
    covered = sorted({row["pattern_id"] for row in rows if row.get("pattern_id")})
    stages = sorted({row["stage"] for row in rows})
    feasible = {pattern["id"] for stage in stages for pattern in feasible_pattern_catalogue(stage)}
    report["pattern_coverage"] = dict(covered=covered, covered_count=len(covered),
                                      feasible_count=len(feasible), missing=sorted(feasible - set(covered)))
    return report
