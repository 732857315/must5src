"""Posterior root defense guard around native search and exact VCF certificates.

A selected move is rejected only when the opponent's actual child position has
an exact forcing-win certificate. Unknown never means safe. All native/VCF
searches share one node/time allocation; atomic native/proof operations cannot
be interrupted in mid-call, and any resulting deadline overrun is reported.
The board's fixed forbidden cells are never used as a candidate exclusion mask.
"""
from functools import lru_cache
import math
from numbers import Integral, Real
import time

import numpy as np

from board_rules import normalize_board, board_winner, legal_cells, apply_board_move
from board_forcing import solve_forcing
from board_threat_search import solve_threat
from native_search import select_native_move


@lru_cache(maxsize=32)
def _segments(shape):
    rows, cols = shape
    lines = []
    for row in range(rows):
        for col in range(cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if 0 <= row + 4 * dr < rows and 0 <= col + 4 * dc < cols:
                    lines.append([(row + i * dr) * cols + col + i * dc for i in range(5)])
    return np.asarray(lines, dtype=np.intp).reshape(-1, 5)


def _ordered_legal(board, side, priors, legal):
    """Score every real empty cell; no frontier or beam removes candidates."""
    segments = _segments(board.shape)
    cells = board.reshape(-1)[segments]
    own, enemy = (cells == side).sum(1), (cells == 3 - side).sum(1)
    blocked = (cells == 3).any(1)
    weights = np.array([0., 1., 8., 64., 1024., 1000000.])
    attack = (weights[np.minimum(own + 1, 5)] - weights[own]) * ((enemy == 0) & ~blocked)
    defense = (weights[np.minimum(enemy + 1, 5)] - weights[enemy]) * ((own == 0) & ~blocked)
    scores = np.zeros(board.size, dtype=np.float64)
    np.add.at(scores, segments.reshape(-1), np.repeat(10 * attack + 11 * defense, 5))
    maximum = float(priors.max())
    if maximum > 0:
        scores += 240 * (priors.reshape(-1) / maximum)
    center = (np.array(board.shape) - 1) / 2
    return sorted(legal, key=lambda move: (-scores[move[0] * board.shape[1] + move[1]],
                                         float(np.square(np.array(move) - center).sum()), move))


def _native_work(result):
    forcing = result.get('forcing')
    # A root VCF certificate bypasses C and its nodes are already the top count.
    bypass = forcing is not None and forcing.get('proven_value') is not None and forcing.get('move') is not None
    return int(result['nodes']) + (int(forcing['nodes']) if forcing is not None and not bypass else 0)


def select_guarded_move(grid, side, priors=None, *, max_nodes=50000, time_limit=.5,
                        depth=9, candidate_width=16, native_fraction=.45,
                        probe_max_nodes=5000, probe_time_limit=.05, forcing_depth=32,
                        threat_time_limit=0.0, threat_max_nodes=20000, threat_width=16,
                        threat_quiet_plies=2, threat_total_plies=64,
                        attack_time_limit=0.0, attack_max_nodes=20000):
    """Prefer a move without a verified opponent VCF over a known losing move.

    The result retains initial_selection and every opponent probe. A changed
    move has completed_depth=0: the initial native depth does not describe that
    alternative. proven_value=-1 requires a certificate for every legal child;
    finding no VCF win is never converted to draw, safety, or a win claim.
    Optional attack_time_limit checks an active own-side forced winning plan.
    threat_time_limit checks opponent continuations with bounded quiet depth.
    Every search shares the original deadline and node allowance; unknown
    results and statistical network values never become proof certificates.
    """
    if isinstance(side, (bool, np.bool_)) or not isinstance(side, Integral) or side not in (1, 2):
        raise ValueError('side must be BLACK=1 or WHITE=2')
    for name, value, minimum, maximum in (('max_nodes', max_nodes, 0, 2**31 - 1),
            ('depth', depth, 1, 58), ('candidate_width', candidate_width, 1, 64),
            ('probe_max_nodes', probe_max_nodes, 1, 2**31 - 1), ('forcing_depth', forcing_depth, 0, 4096),
            ('threat_max_nodes', threat_max_nodes, 1, 2**31 - 1), ('threat_width', threat_width, 1, 4096),
            ('threat_quiet_plies', threat_quiet_plies, 0, 32), ('threat_total_plies', threat_total_plies, 1, 256),
            ('attack_max_nodes', attack_max_nodes, 1, 2**31 - 1)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not minimum <= value <= maximum:
            raise ValueError(f'{name} is outside its allowed integer range')
    for name, value in (('time_limit', time_limit), ('probe_time_limit', probe_time_limit), ('native_fraction', native_fraction), ('threat_time_limit', threat_time_limit), ('attack_time_limit', attack_time_limit)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if native_fraction > 1 or probe_time_limit == 0:
        raise ValueError('native_fraction must be at most 1 and probe_time_limit positive')
    board = normalize_board(grid)
    if max(board.shape) > 64:
        raise ValueError('native guarded search supports board dimensions up to 64')
    side = int(side)
    priors = np.zeros(board.shape, dtype=np.float64) if priors is None else np.asarray(priors, dtype=np.float64)
    if priors.shape != board.shape or not np.isfinite(priors).all() or np.any(priors < 0):
        raise ValueError('priors must be finite, nonnegative, and match the board')
    priors = np.where(board == 0, priors, 0.)
    started = time.monotonic()
    deadline = started + float(time_limit)
    legal = legal_cells(board)
    winner = board_winner(board)
    initial = None
    nodes = 0
    probes = []
    rejected = set()
    selected_probe = None
    original_move = None
    attack_result = None

    def finish(move, status, reason, value=None, line=None):
        elapsed = time.monotonic() - started
        changed = original_move is not None and move != original_move
        exhausted = nodes >= max_nodes or time.monotonic() >= deadline
        return dict(move=move, reason=reason, proven_value=value,
                    principal_variation=line or [], status=status,
                    nodes=nodes, elapsed_seconds=elapsed, budget_exhausted=exhausted,
                    time_limit=float(time_limit), max_nodes=int(max_nodes),
                    threat_time_limit=float(threat_time_limit), threat_max_nodes=int(threat_max_nodes),
                    threat_width=int(threat_width), threat_quiet_plies=int(threat_quiet_plies),
                    threat_total_plies=int(threat_total_plies), attack_time_limit=float(attack_time_limit),
                    attack_max_nodes=int(attack_max_nodes), attack_result=attack_result,
                    deadline_overrun_seconds=max(0., elapsed - float(time_limit)),
                    completed_depth=(initial.get('completed_depth', 0) if initial and not changed else 0),
                    initial_selection=initial, initial_move=original_move, guard_changed=changed,
                    guard_probes=probes, rejected_moves=sorted(rejected), selected_reply=selected_probe,
                    legal_count=len(legal), unexamined_count=len(legal) - len(probes),
                    engine='native_recursive_threat_guard_v2', value_evaluations=0,
                    interpretation='Unknown is not safe; a changed move was selected by root proof checks, not a completed native search of that move.')

    if winner or not legal:
        value = (1 if winner == side else -1) if winner else 0
        return finish(None, 'terminal', '棋局已结束', value)

    initial_time = max(0., deadline - time.monotonic()) * float(native_fraction)
    initial_nodes = int(max_nodes * float(native_fraction))
    root_forcing_nodes = min(int(probe_max_nodes), initial_nodes // 5)
    root_forcing_time = min(.03, initial_time * .2) if root_forcing_nodes else 0.
    initial = select_native_move(board, side, priors, max_nodes=initial_nodes - root_forcing_nodes,
                                 time_limit=max(0., initial_time - root_forcing_time), depth=int(depth),
                                 candidate_width=int(candidate_width), forcing_seconds=root_forcing_time,
                                 forcing_nodes=root_forcing_nodes, forcing_depth=int(forcing_depth))
    nodes = _native_work(initial)
    if nodes > initial_nodes:
        raise RuntimeError('Initial native/VCF search exceeded its allocated node cap')
    original_move = tuple(initial['move']) if initial.get('move') is not None else None
    if original_move not in legal:
        raise RuntimeError('Initial native search returned an illegal move')
    if initial.get('proven_value') == 1:
        line = (initial.get('forcing') or {}).get('principal_variation', [])
        return finish(original_move, 'native_forced_win', '已有全盘强制获胜证明', 1, line)

    # A whole-board strategy also has to create threats, rather than only
    # vetoing losing suggestions. A positive attacking certificate determines
    # a legal move directly; an unknown plan does not change the native move.
    remaining_time = deadline - time.monotonic()
    remaining_nodes = int(max_nodes) - nodes
    if attack_time_limit > 0 and remaining_time > 0 and remaining_nodes > 0:
        allowance = min(int(attack_max_nodes), remaining_nodes)
        attack_result = solve_threat(board, side, max_nodes=allowance,
            time_limit=min(float(attack_time_limit), remaining_time),
            candidate_width=int(threat_width), forcing_depth=int(forcing_depth),
            max_quiet_plies=int(threat_quiet_plies), max_total_plies=int(threat_total_plies))
        used = int(attack_result['nodes'])
        if not 0 <= used <= allowance:
            raise RuntimeError('Attacking threat proof exceeded its shared node allocation')
        nodes += used
        if attack_result['proven_value'] == 1:
            move = tuple(attack_result['move']) if attack_result.get('move') is not None else None
            if move not in legal:
                raise RuntimeError('Attacking threat proof returned an illegal root move')
            return finish(move, 'attack_forced_win', '已找到可验证的全盘主动进攻胜线',
                          1, attack_result['principal_variation'])

    ordered = None
    candidate = original_move
    while candidate is not None:
        remaining_time = deadline - time.monotonic()
        remaining_nodes = int(max_nodes) - nodes
        if remaining_time <= 0 or remaining_nodes <= 0:
            # Keep an unexamined move separate from a move whose child was probed.
            return finish(candidate, 'unexamined_budget_fallback', '预算已用尽，选择尚未证明失败的候选；该点未完成防守验证')
        child = apply_board_move(board, candidate, side)
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            return finish(candidate, 'unexamined_budget_fallback', '预算已用尽，选择尚未证明失败的候选；该点未完成防守验证')
        reply = solve_forcing(child, 3 - side, max_nodes=min(int(probe_max_nodes), remaining_nodes),
                              time_limit=min(float(probe_time_limit), remaining_time), max_depth=int(forcing_depth))
        nodes += int(reply['nodes'])
        if nodes > max_nodes:
            raise RuntimeError('VCF guard exceeded its shared node cap')
        probe = dict(move=candidate, opponent_result=reply, opponent_threat_result=None, rejection_source=None)
        probes.append(probe)
        if reply['proven_value'] == -1:
            selected_probe = probe
            line = [dict(side=side, move=candidate, kind='chosen_root_move')] + reply['principal_variation']
            return finish(candidate, 'child_forced_win', '此着后的对手局面已证明失败', 1, line)
        if reply['proven_value'] == 1:
            probe['rejection_source'] = 'vcf'
        elif reply['proven_value'] is None and threat_time_limit > 0:
            remaining_time = deadline - time.monotonic()
            remaining_nodes = int(max_nodes) - nodes
            if remaining_time > 0 and remaining_nodes > 0:
                allowance = min(int(threat_max_nodes), remaining_nodes)
                threat = solve_threat(child, 3 - side, max_nodes=allowance,
                                      time_limit=min(float(threat_time_limit), remaining_time),
                                      candidate_width=int(threat_width), forcing_depth=int(forcing_depth),
                                      max_quiet_plies=int(threat_quiet_plies), max_total_plies=int(threat_total_plies))
                used = int(threat['nodes'])
                if not 0 <= used <= allowance:
                    raise RuntimeError('Quiet threat search exceeded its shared node allocation')
                # solve_threat.nodes already includes all of its nested VCF work.
                nodes += used
                probe['opponent_threat_result'] = threat
                if threat['proven_value'] == 1:
                    probe['rejection_source'] = 'quiet'
        if probe['rejection_source'] is None:
            selected_probe = probe
            return finish(candidate, 'opponent_not_proved_winning', '在本次预算内未证明对手强制获胜；尚不能保证此着安全')
        rejected.add(candidate)
        if len(rejected) == len(legal):
            chosen = original_move
            selected_probe = next(item for item in probes if item['move'] == chosen)
            certificate = (selected_probe['opponent_threat_result'] if selected_probe['rejection_source'] == 'quiet'
                           else selected_probe['opponent_result'])
            line = [dict(side=side, move=chosen, kind='representative_root_defense')] + certificate['principal_variation']
            return finish(chosen, 'all_moves_proved_losing', '全部合法落点都已验证存在对手强制胜线', -1, line)
        if ordered is None:
            ordered = _ordered_legal(board, side, priors, legal)
        candidate = next((move for move in ordered if move not in rejected), None)
    raise RuntimeError('Guard exhausted candidates without accounting for every legal move')
