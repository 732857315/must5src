"""Diverse reachable full-board tactical positions with checked VCF labels.

These are constructed positions, not fabricated completed games. Side counts
and the initial black center are valid, no five is present, and every action
and value target is verified on the complete board by the forcing solver.
"""
import random
import time
from collections import Counter
from numbers import Integral

import numpy as np
from board_rules import board_winner, tactical_candidates
from board_forcing import solve_forcing
from global_data import canonical_board_key, physical_board_key


def _candidate(rng, shape, actor, kind):
    height, width = shape
    board = np.zeros(shape, dtype=np.uint8)
    direction = rng.choice(((0,1),(1,0),(1,1),(1,-1)))
    for _ in range(100):
        start = (rng.randrange(height), rng.randrange(width))
        line = [(start[0]+i*direction[0], start[1]+i*direction[1]) for i in range(6)]
        if all(0<=r<height and 0<=c<width for r,c in line):
            break
    else:
        return None
    owner = 3-actor if kind=='enemy_open_four' else actor
    pattern = [0,owner,owner,owner,owner,0]
    if kind=='own_open_three':
        pattern[rng.randrange(1,5)] = 0
    protected = set(line)
    for point,value in zip(line,pattern):
        board[point]=value
    center = (height//2,width//2)
    if board[center]==2 or (center in protected and board[center]==0):
        return None
    board[center]=1
    protected.add(center)
    available = [(r,c) for r in range(height) for c in range(width) if (r,c) not in protected]
    rng.shuffle(available)
    # Include clear, one-edge and arbitrary-blocked contexts without changing
    # the six-cell tactical core or the mandatory initial black center.
    mask_kind = rng.choice(('none','edge','random'))
    if mask_kind=='edge':
        edge = rng.randrange(4)
        for r,c in available:
            if (edge==0 and r==0) or (edge==1 and r==height-1) or (edge==2 and c==0) or (edge==3 and c==width-1):
                board[r,c]=3
    elif mask_kind=='random':
        for point in available[:rng.randrange(1,max(2,board.size//8))]:
            board[point]=3
    if not np.any(board == 3):
        mask_kind = 'none'
    available = [point for point in available if board[point]==0]
    target_each = rng.randrange(5,min(32,len(available)//2))
    desired = {1:target_each+(actor==2),2:target_each}
    for color in (1,2):
        needed = desired[color]-int(np.count_nonzero(board==color))
        if needed<0 or needed>len(available):
            return None
        for _ in range(needed):
            board[available.pop()]=color
    if board_winner(board):
        return None
    assert int((board==1).sum())-int((board==2).sum())==int(actor==2)
    return board,mask_kind


def build_tactical_positions(count=2048, seed=20260907, sizes=(16,), *, seen_physical=(), max_attempts=None,
                             progress_callback=None):
    if isinstance(count,(bool,np.bool_)) or not isinstance(count,Integral) or count<0:
        raise ValueError('count must be a nonnegative integer')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,Integral):
        raise ValueError('seed must be an integer')
    count,seed=int(count),int(seed)
    if max_attempts is None:
        max_attempts=max(100,count*30)
    if isinstance(max_attempts,(bool,np.bool_)) or not isinstance(max_attempts,Integral) or max_attempts<0:
        raise ValueError('max_attempts must be a nonnegative integer')
    max_attempts=int(max_attempts)
    if isinstance(sizes,Integral) and not isinstance(sizes,(bool,np.bool_)):
        sizes=(sizes,)
    try:
        values=list(sizes)
    except TypeError as exc:
        raise ValueError('sizes must contain square sizes or integer (rows,cols) pairs') from exc
    shapes=[]
    for value in values:
        if isinstance(value,Integral) and not isinstance(value,(bool,np.bool_)):
            shape=(value,value)
        else:
            try:
                shape=tuple(value)
            except TypeError as exc:
                raise ValueError('sizes must contain square sizes or integer (rows,cols) pairs') from exc
        if len(shape)!=2 or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Integral) or x<6 for x in shape):
            raise ValueError('each board dimension must be an integer at least six')
        shapes.append(tuple(map(int,shape)))
    if not shapes:
        raise ValueError('sizes cannot be empty')
    if isinstance(seen_physical,(str,bytes)):
        raise ValueError('seen_physical must be a collection of physical board keys')
    try:
        excluded=list(seen_physical)
    except TypeError as exc:
        raise ValueError('seen_physical must be a collection of physical board keys') from exc
    if any(not isinstance(key,str) or not key for key in excluded):
        raise ValueError('seen_physical entries must be nonempty string keys')
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError('progress_callback must be callable or None')
    rng=random.Random(seed)
    seen=set(excluded)
    records=[]
    attempts=0
    started=time.monotonic()
    while len(records)<count and attempts<max_attempts:
        attempts+=1
        actor=rng.choice((1,2))
        expected = 1 if len(records)%2==0 else -1
        kind = rng.choice(('own_open_four','own_open_three')) if expected==1 else 'enemy_open_four'
        candidate=_candidate(rng,shapes[len(records)%len(shapes)],actor,kind)
        if candidate is None:
            continue
        board,mask_kind=candidate
        physical=physical_board_key(board)
        if physical in seen:
            continue
        proof=solve_forcing(board,actor,max_depth=16,max_nodes=5000,time_limit=.03)
        if proof['proven_value']!=expected or proof['move'] is None:
            continue
        wins,blocks,_=tactical_candidates(board,actor)
        target=np.zeros(board.shape,dtype=np.float32)
        move=tuple(proof['move'])
        if expected == -1:
            # A certified loss means every legal move has minimax value -1.
            # Do not invent a unique optimal action from a representative PV.
            target[board==0]=1/int(np.count_nonzero(board==0))
            policy_source='all_legal_proved_loss'
        elif wins:
            for point in wins:
                target[point]=1/len(wins)
            policy_source='immediate_win_set'
        elif len(blocks)==1:
            assert move==blocks[0]
            target[move]=1
            policy_source='forced_block'
        else:
            target[move]=1
            policy_source='vcf_proof_move' if expected==1 else 'proved_loss_resistance'
        seen.add(physical)
        case_id=f'tactical-case-{seed}-{len(records):06d}'
        records.append(dict(board=board,side=actor,target_policy=target,action=move,
             board_key=canonical_board_key(board,actor),physical_key=physical,game_id=case_id,
             group_kind='constructed_position',case_id=case_id,ply=int(np.count_nonzero((board==1)|(board==2))),
             opening=[],mask_kind=mask_kind,source='constructed_verified_full_board',tactical_kind=kind,
             policy_source=policy_source,explored=False,search_engine='continuous_four_proof',
             search_requested_depth=16,search_completed_depth=proof['max_ply'],search_nodes=proof['nodes'],
             search_budget_exhausted=proof['budget_exhausted'],search_proven_value=expected,
             search_reason=proof['reason'],search_principal_variation=proof['principal_variation'],
             value=float(expected),value_valid=True,value_source='search_proof',terminal_value=None,
             game_terminal=False,game_winner=None,game_termination='constructed_not_played',game_length=None))
        if progress_callback is not None and (len(records)%200==0 or len(records)==count):
            progress_callback(dict(records=len(records),requested=count,attempts=attempts,
                                   elapsed_seconds=time.monotonic()-started))
    if len(records)!=count:
        raise RuntimeError(f'Could only construct {len(records)} of {count} tactical cases after {attempts} attempts')
    report=dict(constructed_cases=len(records),requested_count=count,attempts=attempts,max_attempts=max_attempts,
                excluded_physical_keys=len(set(excluded)),seed=seed,
                elapsed_seconds=time.monotonic()-started,
                values=dict(Counter(str(int(r['value'])) for r in records)),
                actors=dict(Counter(str(r['side']) for r in records)),
                kinds=dict(Counter(r['tactical_kind'] for r in records)),
                masks=dict(Counter(r['mask_kind'] for r in records)),
                interpretation='Independent constructed, reachable, full-board positions; not completed games.')
    return records,report
