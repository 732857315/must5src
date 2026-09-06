"""Bounded recursive quiet-threat certificates against every real legal reply.

Root candidates may be trimmed by a heuristic. Opponent replies never are:
every proposed move needs a positive continuation for every actual legal reply.
Forcing attacks and mandatory defenses may connect bounded quiet setups.
Unknown is not safety, draw, or loss. No neural value participates in proof.
"""
from functools import lru_cache
import math
from numbers import Integral, Real
import time

import numpy as np

from board_rules import normalize_board, board_winner, apply_board_move
from board_forcing import solve_forcing


@lru_cache(maxsize=32)
def _segments(shape):
    rows, cols = shape
    lines = []
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if 0 <= r + 4 * dr < rows and 0 <= c + 4 * dc < cols:
                    lines.append([(r + i * dr) * cols + c + i * dc for i in range(5)])
    result = np.asarray(lines, dtype=np.intp).reshape(-1, 5)
    result.setflags(write=False)
    return result


def _facts(board, side, lines):
    cells = board.reshape(-1)[lines]
    own, enemy = (cells == side).sum(1), (cells == 3 - side).sum(1)
    empties = cells == 0
    counts = empties.sum(1)
    def points(mask):
        return [tuple(map(int, divmod(i, board.shape[1]))) for i in np.unique(lines[mask][empties[mask]])]
    return dict(winner=side if np.any(own == 5) else 3 - side if np.any(enemy == 5) else 0,
                wins=points((own == 4) & (counts == 1)),
                forcing=points((own == 3) & (counts == 2)),
                enemy_wins=points((enemy == 4) & (counts == 1)))


def _ordered_moves(board, side, lines):
    cells = board.reshape(-1)[lines]
    own, enemy = (cells == side).sum(1), (cells == 3 - side).sum(1)
    blocked = (cells == 3).any(1)
    weights = np.array([0., 1., 8., 64., 1024., 1000000.])
    attack = (weights[np.minimum(own + 1, 5)] - weights[own]) * ((enemy == 0) & ~blocked)
    defense = (weights[np.minimum(enemy + 1, 5)] - weights[enemy]) * ((own == 0) & ~blocked)
    scores = np.zeros(board.size)
    np.add.at(scores, lines.reshape(-1), np.repeat(10 * attack + 11 * defense, 5))
    center = (np.array(board.shape) - 1) / 2
    moves = [tuple(map(int, p)) for p in np.argwhere(board == 0)]
    return sorted(moves, key=lambda p: (-scores[p[0] * board.shape[1] + p[1]],
                                       float(np.square(np.array(p) - center).sum()), p))


class _BudgetExceeded(Exception):
    pass


class _ThreatSearch:
    def __init__(self, board, side, *, max_nodes, time_limit, candidate_width,
                 forcing_depth, vcf_max_nodes, vcf_time_limit, max_total_plies):
        self.side = side
        self.max_nodes = max_nodes
        self.started = time.monotonic()
        self.deadline = self.started + time_limit
        self.lines = _segments(board.shape)
        self.candidate_width = candidate_width
        self.forcing_depth = forcing_depth
        self.vcf_max_nodes = vcf_max_nodes
        self.vcf_time_limit = vcf_time_limit
        self.max_total_plies = max_total_plies
        self.nodes = self.vcf_nodes = self.cache_hits = self.recursive_calls = 0
        self.max_ply = 0
        self.active_quiet_limit = self.max_quiet_searched = 0
        self.total_ply_limited = False
        self.quiet_limited = False
        self.cache = {}

    def check(self):
        if self.nodes >= self.max_nodes or time.monotonic() >= self.deadline:
            raise _BudgetExceeded

    def visit(self, ply):
        self.check()
        self.nodes += 1
        self.max_ply = max(self.max_ply, ply)

    def new_frame(self, board, quiet_remaining, ply):
        return dict(status='unknown', proven_value=None, move=None, winner=None,
                    principal_variation=[], proof_plies=0, reason='searching',
                    quiet_plies_used=0, quiet_remaining=quiet_remaining, ply=ply,
                    total_plies_remaining=self.max_total_plies-ply,
                    root_legal_count=int(np.count_nonzero(board == 0)),
                    examined_root_moves=0, skipped_forcing_moves=0,
                    candidates_checked=0, quiet_candidates_checked=0, candidate_limit_reached=False,
                    candidate_attempts=[], winning_candidate=None, cache_hit=False)

    def positive(self, frame, reason, move=None, line=None, evidence=None, quiet_used=0):
        frame.update(status='win', proven_value=1, winner=self.side, move=move,
                     principal_variation=line or [], proof_plies=len(line or []), reason=reason,
                     winning_candidate=evidence, quiet_plies_used=quiet_used)
        return frame

    def search(self, board, quiet_remaining, ply=0, frame=None):
        if frame is None:
            frame = self.new_frame(board, quiet_remaining, ply)
        if ply == 0:
            self.active_quiet_limit = quiet_remaining
        start_nodes, start_vcf, start_time = self.nodes, self.vcf_nodes, time.monotonic()
        key = (board.tobytes(), self.side, quiet_remaining, self.max_total_plies-ply)
        try:
            self.visit(ply)  # Cache hits consume budget too.
            cached = self.cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                frame.update(cached)
                frame['cache_hit'] = True
                return frame
            self.explore(board, quiet_remaining, ply, frame)
            if frame['proven_value'] == 1:
                # Only completed positive certificates enter the per-call cache.
                self.cache[key] = frame.copy()
            return frame
        except _BudgetExceeded:
            frame.update(status='unknown', proven_value=None, move=None, winner=None,
                         principal_variation=[], proof_plies=0, reason='budget_exhausted')
            if frame['candidate_attempts'] and frame['candidate_attempts'][-1]['status'] == 'checking':
                frame['candidate_attempts'][-1]['status'] = 'budget_exhausted'
            raise
        finally:
            frame.update(nodes=self.nodes-start_nodes, vcf_nodes=self.vcf_nodes-start_vcf,
                         elapsed_seconds=time.monotonic()-start_time)

    def explore(self, board, quiet_remaining, ply, frame):
        side = self.side
        facts = _facts(board, side, self.lines)
        winner = board_winner(board)
        if winner:
            frame['reason'] = 'terminal_opponent_win'
            if winner == side:
                self.positive(frame, 'terminal_own_win')
            return
        if not frame['root_legal_count']:
            frame['reason'] = 'terminal_draw_no_winning_move'
            return
        remaining = self.max_total_plies - ply
        if remaining <= 0:
            self.total_ply_limited = True
            frame['reason'] = 'total_ply_limit'
            return
        if facts['wins']:
            move = facts['wins'][0]
            self.positive(frame, 'immediate_win', move,
                          [dict(side=side, move=move, kind='immediate_win')])
            self.max_ply = max(self.max_ply, ply+1)
            return
        if len(facts['enemy_wins']) >= 2:
            frame.update(reason='opponent_immediate_wins', opponent_winning_cells=facts['enemy_wins'])
            return
        mandatory = bool(facts['enemy_wins'])
        if not mandatory and quiet_remaining <= 0:
            self.quiet_limited = True
            frame['reason'] = 'quiet_ply_limit'
            return
        if remaining < 2:
            self.total_ply_limited = True
            frame['reason'] = 'total_ply_limit'
            return
        # Only quiet setup choices spend the width/quiet allowance. Active
        # fours and mandatory defenses can connect two quiet phases.
        forcing_moves = set(facts['forcing'])
        ranked = facts['enemy_wins'] if mandatory else _ordered_moves(board, side, self.lines)
        candidates = ranked if mandatory else ([move for move in ranked if move not in forcing_moves]
                                               + [move for move in ranked if move in forcing_moves])
        for move in candidates:
            if not mandatory and move not in forcing_moves:
                if frame['quiet_candidates_checked'] >= self.candidate_width:
                    frame['candidate_limit_reached'] = True
                    continue
                frame['quiet_candidates_checked'] += 1
            self.visit(ply+1)
            frame['examined_root_moves'] += 1
            child = apply_board_move(board, move, side)
            child_facts = _facts(child, side, self.lines)
            forcing_attack = bool(child_facts['wins'])
            quiet_cost = 0 if mandatory or forcing_attack else 1
            next_quiet = quiet_remaining - quiet_cost
            self.max_quiet_searched = max(self.max_quiet_searched, self.active_quiet_limit-next_quiet)
            expected = {tuple(map(int, p)) for p in np.argwhere(child == 0)}
            attempt = dict(move=move, quiet=bool(quiet_cost), mandatory_defense=mandatory,
                           forcing_attack=forcing_attack,
                           quiet_cost=quiet_cost, quiet_remaining_after_move=next_quiet,
                           legal_reply_count=len(expected), checked_replies=0,
                           certified_replies=0, reply_proofs=[], status='checking')
            frame['candidate_attempts'].append(attempt)
            frame['candidates_checked'] += 1
            if child_facts['enemy_wins']:
                attempt.update(status='opponent_immediate_win', refutation_move=child_facts['enemy_wins'][0])
                continue
            if not expected:
                attempt['status'] = 'terminal_draw'
                continue
            responses = _ordered_moves(child, 3-side, self.lines)
            if len(responses) != len(expected) or set(responses) != expected:
                # An ordering defect must not silently remove a real defense.
                attempt['status'] = 'incomplete_reply_coverage'
                continue
            complete = True
            for response in responses:
                self.visit(ply+2)
                after = apply_board_move(child, response, 3-side)
                attempt['checked_replies'] += 1
                if board_winner(after) == 3-side:
                    attempt.update(status='opponent_immediate_win', refutation_move=response)
                    complete = False
                    break
                continuation_remaining = self.max_total_plies - (ply+2)
                record = dict(move=response, continuation=None, vcf_result=None,
                              recursive_result=None, proof_source=None)
                attempt['reply_proofs'].append(record)
                if continuation_remaining <= 0:
                    self.total_ply_limited = True
                    attempt['status'] = 'total_ply_limit'
                    complete = False
                    break
                self.check()
                allowance = min(self.vcf_max_nodes, self.max_nodes-self.nodes)
                seconds = min(self.vcf_time_limit, max(0., self.deadline-time.monotonic()))
                if seconds <= 0:
                    raise _BudgetExceeded
                # VCF leaves may append up to two immediate tactical plies.
                # Also check the returned certificate length before accepting it.
                depth = min(self.forcing_depth, max(0, continuation_remaining-2))
                vcf = solve_forcing(after, side, max_nodes=allowance, time_limit=seconds, max_depth=depth)
                used = int(vcf['nodes'])
                if not 0 <= used <= allowance:
                    raise RuntimeError('VCF continuation exceeded its allocated node cap')
                self.nodes += used
                self.vcf_nodes += used
                self.max_ply = max(self.max_ply, ply+2+int(vcf.get("max_ply", 0)))
                record['vcf_result'] = vcf
                record['continuation'] = vcf
                if vcf['proven_value'] == 1 and len(vcf['principal_variation']) <= continuation_remaining:
                    record['proof_source'] = 'vcf'
                    self.max_ply = max(self.max_ply, ply+2+len(vcf['principal_variation']))
                elif vcf['proven_value'] == 1:
                    # Do not turn a deeper leaf certificate into an in-budget win.
                    self.total_ply_limited = True
                    attempt['status'] = 'total_ply_limit'
                    complete = False
                    break
                elif vcf['proven_value'] is None and (next_quiet > 0 or
                        (next_quiet == 0 and len(_facts(after, side, self.lines)['enemy_wins']) == 1)):
                    nested = self.new_frame(after, next_quiet, ply+2)
                    record['recursive_result'] = nested
                    self.recursive_calls += 1
                    self.search(after, next_quiet, ply+2, nested)
                    record['continuation'] = nested
                    if nested['proven_value'] == 1:
                        record['proof_source'] = 'recursive'
                if record['proof_source'] is None:
                    complete = False
                    attempt['status'] = ('continuation_unknown' if vcf['proven_value'] is None
                                         else 'continuation_not_winning')
                    if next_quiet <= 0 and vcf['proven_value'] is None:
                        self.quiet_limited = True
                    break
                attempt['certified_replies'] += 1
            covered = {tuple(p['move']) for p in attempt['reply_proofs'] if p['proof_source'] is not None}
            if complete and covered == expected and attempt['certified_replies'] == len(expected):
                attempt['status'] = 'win_all_legal_replies'
                representative = max(attempt['reply_proofs'], key=lambda r: len(r['continuation']['principal_variation']))
                line = [dict(side=side, move=move, kind='mandatory_defense' if mandatory else 'forcing_attack' if forcing_attack else 'quiet_move'),
                        dict(side=3-side, move=representative['move'], kind='representative_defense')]
                line += representative['continuation']['principal_variation']
                quiet_used = quiet_cost + max(p['continuation'].get('quiet_plies_used', 0)
                                              for p in attempt['reply_proofs'])
                self.positive(frame, 'mandatory_defense_all_replies_have_win' if mandatory
                              else 'forcing_attack_all_replies_have_win' if forcing_attack
                              else 'quiet_move_all_replies_have_vcf_win', move, line, attempt, quiet_used)
                return
            if complete:
                attempt['status'] = 'incomplete_reply_coverage'
        frame['candidate_limit_reached'] |= frame['quiet_candidates_checked'] >= self.candidate_width
        frame['reason'] = 'no_quiet_win_certificate'


def solve_threat(grid, side, *, max_nodes=20000, time_limit=.5, candidate_width=16,
                 forcing_depth=32, vcf_max_nodes=5000, vcf_time_limit=.05,
                 max_quiet_plies=1, max_total_plies=64, staged_vcf=True):
    """Prove an actual win through bounded quiet setups and mandatory defenses.

    Fast VCF leaves first scan the candidates; unresolved work can then use
    the configured leaf caps. Only completed proofs are cached between stages.
    All recursive work shares one node counter/deadline. The active quiet cap
    increases from 1 to max_quiet_plies, retaining fast one-setup certificates.
    Mandatory blocks and active fours do not spend a quiet allowance. Width
    bounds quiet candidates, not forcing transitions. A zero remaining quiet
    allowance still permits recursive mandatory defense. Every proposed
    move still needs positive continuations for all real legal opponent replies.
    Unknown, candidate trimming and ply cutoffs can never establish loss/draw.
    max_total_plies bounds every accepted proof line, including VCF leaf moves.
    """
    if isinstance(side, (bool, np.bool_)) or not isinstance(side, Integral) or side not in (1, 2):
        raise ValueError('side must be explicit BLACK=1 or WHITE=2')
    for name, value, minimum in (('max_nodes', max_nodes, 0), ('candidate_width', candidate_width, 1),
                                 ('forcing_depth', forcing_depth, 0), ('vcf_max_nodes', vcf_max_nodes, 1),
                                 ('max_quiet_plies', max_quiet_plies, 0), ('max_total_plies', max_total_plies, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} must be an integer at least {minimum}')
    # Two board plies per recursive frame; this also avoids Python stack limits.
    if max_total_plies > 256:
        raise ValueError('max_total_plies must not exceed 256')
    for name, value in (('time_limit', time_limit), ('vcf_time_limit', vcf_time_limit)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if vcf_time_limit == 0:
        raise ValueError('vcf_time_limit must be positive')
    if not isinstance(staged_vcf, (bool, np.bool_)):
        raise ValueError('staged_vcf must be a boolean')
    board = normalize_board(grid)
    search = _ThreatSearch(board, int(side), max_nodes=int(max_nodes), time_limit=float(time_limit),
                           candidate_width=int(candidate_width), forcing_depth=int(forcing_depth),
                           vcf_max_nodes=int(vcf_max_nodes), vcf_time_limit=float(vcf_time_limit),
                           max_total_plies=int(max_total_plies))
    configured_caps = (int(vcf_max_nodes), float(vcf_time_limit))
    fast_caps = (min(configured_caps[0], 500), min(configured_caps[1], .005))
    stages = [('configured', configured_caps)]
    if staged_vcf and fast_caps != configured_caps:
        stages.insert(0, ('fast', fast_caps))
    iterations = []
    first_quiet = 0 if max_quiet_plies == 0 else 1
    frame = search.new_frame(board, first_quiet, 0)
    stop = False
    for stage, caps in stages:
        search.vcf_max_nodes, search.vcf_time_limit = caps
        for limit in range(first_quiet, int(max_quiet_plies)+1):
            frame = search.new_frame(board, limit, 0)
            try:
                search.search(board, limit, frame=frame)
            except _BudgetExceeded:
                pass
            iterations.append(dict(quiet_limit=limit, vcf_stage=stage,
                                   vcf_max_nodes=caps[0], vcf_time_limit=caps[1],
                                   proven_value=frame['proven_value'], reason=frame['reason'],
                                   nodes=frame['nodes'], elapsed_seconds=frame['elapsed_seconds'],
                                   candidates_checked=frame['candidates_checked']))
            stop = frame['proven_value'] == 1 or frame['reason'] in (
                'budget_exhausted', 'terminal_opponent_win',
                'terminal_draw_no_winning_move', 'total_ply_limit')
            if stop:
                break
        if stop:
            break
    elapsed = time.monotonic()-search.started
    frame.update(nodes=search.nodes, vcf_nodes=search.vcf_nodes, elapsed_seconds=elapsed,
                 max_nodes=int(max_nodes), time_limit=float(time_limit),
                 deadline_overrun_seconds=max(0., elapsed-float(time_limit)),
                 budget_exhausted=search.nodes >= max_nodes or elapsed >= time_limit,
                 candidate_width=int(candidate_width), forcing_depth=int(forcing_depth),
                 vcf_max_nodes=int(vcf_max_nodes), vcf_time_limit=float(vcf_time_limit),
                 max_quiet_plies=int(max_quiet_plies), max_total_plies=int(max_total_plies),
                 max_ply=search.max_ply, total_ply_limited=search.total_ply_limited,
                 quiet_limited=search.quiet_limited, recursive_calls=search.recursive_calls,
                 max_quiet_plies_searched=search.max_quiet_searched,
                 cache_hits=search.cache_hits, quiet_iterations=iterations,
                 staged_vcf=bool(staged_vcf),
                 proof_scope='Actual attacker win only. Every legal defender reply has a positive VCF or recursive certificate; all work shares node, time, quiet and total-ply bounds. Unknown is never a loss or draw.')
    return frame
