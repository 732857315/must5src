"""Freestanding native search with validated Python inputs and optional VCF."""
import ctypes
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np
from board_rules import normalize_board, board_winner

_ROOT = Path(__file__).resolve().parent
_LOCAL = threading.local()
_BUILD_LOCK = threading.Lock()
_LIBRARY = None
_STOP = ctypes.CFUNCTYPE(ctypes.c_int)


class NativeOutput(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in ('move', 'nodes', 'completed_depth', 'score',
                                               'proof', 'budget_exhausted', 'status')]


def native_library():
    global _LIBRARY
    with _BUILD_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        source = _ROOT / 'native_board.c'
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = _ROOT / 'exports' / 'native_search' / source_hash[:16]
        destination.mkdir(parents=True, exist_ok=True)
        binary = destination / 'native_board.dll'
        if not binary.exists():
            clang, linker = shutil.which('clang'), shutil.which('lld-link')
            if not clang or not linker:
                raise RuntimeError('Native search requires clang and lld-link; choose the Python engine otherwise')
            obj = destination / 'native_board.obj'
            subprocess.run([clang, '-c', str(source), '-o', str(obj), '-O3', '-ffreestanding', '-fno-builtin'],
                           check=True, capture_output=True, text=True)
            subprocess.run([linker, '/dll', '/noentry', '/nodefaultlib', '/out:' + str(binary), str(obj)],
                           check=True, capture_output=True, text=True)
            (destination / 'build.json').write_text(json.dumps(dict(source_sha256=source_hash, clang=clang,
                  linker=linker, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()), indent=2), encoding='utf-8')
        library = ctypes.CDLL(str(binary))
        library.native_context_size.argtypes = []
        library.native_context_size.restype = ctypes.c_ulonglong
        library.native_select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
            ctypes.c_int, _STOP, ctypes.POINTER(NativeOutput)]
        library.native_select.restype = ctypes.c_int
        library.source_sha256 = source_hash
        library.binary_path = str(binary)
        library.binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
        _LIBRARY = library
        return library


def native_fingerprint():
    library = native_library()
    return dict(engine='native_freestanding_v1', source_sha256=library.source_sha256,
                binary_sha256=library.binary_sha256, binary_path=library.binary_path)


def select_native_move(grid, side, priors=None, *, max_nodes=50000, time_limit=1., depth=9,
                       candidate_width=16, forcing_seconds=0., forcing_nodes=10000, forcing_depth=24):
    board = normalize_board(grid)
    if isinstance(side, (bool, np.bool_)) or not isinstance(side, Integral) or side not in (1, 2):
        raise ValueError('side must be BLACK=1 or WHITE=2')
    if max(board.shape)>64:
        raise ValueError('native board dimensions must be at most 64')
    for value, name, low, high in ((max_nodes, 'max_nodes', 0, 2**31-1), (depth, 'depth', 1, 58),
                                   (candidate_width, 'candidate_width', 1, 64),
                                   (forcing_nodes, 'forcing_nodes', 0, 2**31-1),
                                   (forcing_depth, 'forcing_depth', 0, 4096)):
        if isinstance(value, bool) or not isinstance(value, Integral) or not low<=value<=high:
            raise ValueError(f'{name} outside allowed integer range')
    for value in (time_limit, forcing_seconds):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value<0:
            raise ValueError('time budgets must be finite and nonnegative')
    if priors is None:
        priors = np.zeros(board.shape, dtype=np.float64)
    priors = np.asarray(priors, dtype=np.float64)
    if priors.shape!=board.shape or not np.isfinite(priors).all() or np.any(priors<0):
        raise ValueError('priors must match the board and be finite nonnegative')
    priors = np.ascontiguousarray(np.where(board==0, priors, 0))
    library = native_library()
    if not hasattr(_LOCAL, 'context'):
        _LOCAL.context = ctypes.create_string_buffer(library.native_context_size())
    started = time.monotonic()
    forcing = None
    if forcing_seconds>0:
        from board_forcing import solve_forcing
        forcing = solve_forcing(board, int(side), max_nodes=forcing_nodes, time_limit=forcing_seconds, max_depth=forcing_depth)
        if forcing['proven_value'] is not None and forcing.get('move') is not None:
            return dict(move=tuple(forcing['move']), reason='全盘连续冲四搜索已证明强制胜' if forcing['proven_value']==1 else '全盘连续冲四搜索已证明当前必防链失败',
                        nodes=forcing['nodes'], completed_depth=0, proven_value=forcing['proven_value'],
                        elapsed_seconds=time.monotonic()-started, budget_exhausted=forcing['budget_exhausted'],
                        forcing=forcing, engine='native_freestanding_v1', value_evaluations=0)
    deadline = time.monotonic() + time_limit
    callback_errors = []
    @_STOP
    def stop():
        try:
            return int(time.monotonic()>=deadline)
        except BaseException as exc:
            callback_errors.append(exc)
            return 1
    output = NativeOutput()
    code = library.native_select(_LOCAL.context, board.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                  board.shape[0], board.shape[1], int(side), priors.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  int(depth), int(candidate_width), int(max_nodes), stop, ctypes.byref(output))
    if callback_errors:
        raise callback_errors[0]
    if code:
        raise RuntimeError(f'Native search rejected request: {code}')
    if (not -1 <= output.move < board.size or output.proof not in (-1, 0, 1, 2)
            or not 0 <= output.nodes <= max_nodes or not 0 <= output.completed_depth <= depth):
        raise RuntimeError('Native search returned invalid result fields')
    move = None if output.move<0 else tuple(map(int, np.unravel_index(output.move, board.shape)))
    if move is not None and (board[move]!=0 or board_winner(board)):
        raise RuntimeError('Native search returned an illegal action')
    reasons = {1:'棋局已结束', 2:'全盘立即成五', 3:'对手有多个独立成五点，当前无法一手全堵',
               4:'形成活四并验证两个独立成五点', 5:'原生全盘搜索已证明此着可强制获胜'}
    return dict(move=move, reason=reasons.get(output.status, f'原生全盘搜索完成 {output.completed_depth} 层'),
                nodes=output.nodes, completed_depth=output.completed_depth, proven_value=None if output.proof==2 else output.proof,
                heuristic_score=output.score, elapsed_seconds=time.monotonic()-started,
                budget_exhausted=bool(output.budget_exhausted), engine='native_freestanding_v1',
                value_evaluations=0, forcing=forcing)
