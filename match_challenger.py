"""Serious, reproducible browser challenger using full-board search and VCF.

All played actions come from bounded search. Seed changes root tie ordering;
there is no random-move branch, deliberate blunder, outcome-dependent weakening,
or access to the candidate's weights, hidden state, or desired acceptance count.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from board_rules import normalize_board, board_winner
from native_search import select_native_move, native_fingerprint

DEFAULT_OPTIONS = dict(time_limit=1.0, max_nodes=100000, depth=9, candidate_width=24,
                       forcing_seconds=.15, forcing_nodes=20000, forcing_depth=32)


def configuration_fingerprint(options=None):
    resolved = dict(DEFAULT_OPTIONS)
    if options:
        if set(options) - set(resolved):
            raise ValueError('Unknown challenger option')
        resolved.update(options)
    root = Path(__file__).resolve().parent
    return dict(name='full_board_native_vcf_challenger_v1', options=resolved,
                policy='search every move; seeded root ordering, no random action sampling',
                native=native_fingerprint(),
                source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                               for name in ('match_challenger.py','native_search.py','native_board.c','board_forcing.py','board_rules.py')})


def choose_move(board, side, *, context):
    board = normalize_board(board)
    if board_winner(board) or not np.any(board==0):
        raise ValueError('A challenger action was requested on a terminal board')
    options = dict(DEFAULT_OPTIONS)
    supplied = context.get('opponent_options', {})
    if not isinstance(supplied, dict) or set(supplied)-set(options):
        raise ValueError('Invalid challenger options')
    options.update(supplied)
    identity = json.dumps({name:context.get(name) for name in ('seed','game_index','ai_side','ply')},
                           sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(identity+bytes([int(side)])+board.tobytes()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    priors = rng.random(board.shape) * (board==0)
    priors /= priors.sum()
    result = select_native_move(board, side, priors, **options)
    if result['move'] is None or board[tuple(result['move'])]!=0:
        raise AssertionError('Challenger search failed to return a legal action')
    return dict(move=list(result['move']), reason=result['reason'], search=result,
                challenger='full_board_native_vcf_challenger_v1', root_order_seed=digest.hex(),
                options=options, neural_value_used=False)
