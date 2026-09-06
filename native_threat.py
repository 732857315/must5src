"""Isolated ctypes bridge for the bounded C all-replies threat prover.

Only completed positive proofs become value=1. Unknown/capacity/budget results
are never losses. Search, context setup, and certificate decoding are timed
separately; optional independent verification is deliberately a separate call.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid

import numpy as np

_ROOT = Path(__file__).resolve().parent
_BUILD_LOCK = threading.RLock()
_LOCAL = threading.local()
_LIBRARIES = {}
_NOW = ctypes.CFUNCTYPE(ctypes.c_double)
_INT_MAX = 2**31 - 1
_MAX_CONTEXT_BYTES = 512 * 1024 * 1024
_CAPACITIES = dict(node_capacity=2048, edge_capacity=65536, line_capacity=262144)


class NativeThreatOutput(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in (
        'value', 'move', 'nodes', 'exhausted', 'root_id', 'certificate_nodes',
        'certificate_edges', 'pv_length', 'certified_replies',
        'completed_quiet', 'status', 'iteration_count')]


class NativeThreatNode(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in ('move', 'side', 'first_edge', 'edge_count')]
    _fields_ += [('board', ctypes.c_ubyte * 1024)]


class NativeThreatEdge(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in ('reply', 'child_id', 'line_offset', 'line_length')]


class NativeThreatIteration(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in ('ordering', 'quiet', 'leaf_ms', 'nodes', 'completed')]


def _clock_ms():
    """Host clock hook, also used by small callback-failure regression tests."""
    return time.perf_counter() * 1000.0


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer between {low} and {high}')
    return int(value)


def _finite_milliseconds(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError('milliseconds must be finite and nonnegative')
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError('milliseconds must fit a finite C double') from exc
    if not math.isfinite(converted) or converted < 0:
        raise ValueError('milliseconds must be finite and nonnegative')
    return converted


def _validated_board(grid):
    # dtype=object preserves a bool mixed with ints so it cannot silently become 1.
    try:
        values = np.asarray(grid, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError('board must be a rectangular two-dimensional integer array') from exc
    if values.ndim != 2 or any(not 5 <= n <= 32 for n in values.shape):
        raise ValueError('board dimensions must each be between 5 and 32')
    for cell in values.flat:
        _integer(cell, 'board cell', 0, 3)
    return np.array(values, dtype=np.uint8, order='C', copy=True)


@contextmanager
def _process_build_lock(path):
    """Serialize compilers in separate Python processes without stale lock files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        deadline = time.monotonic() + 30
        while True:
            try:
                stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (PermissionError, BlockingIOError, OSError) as exc:
                if getattr(exc, 'errno', None) not in (11, 13, 35, 36) and getattr(exc, 'winerror', None) not in (5, 32, 33):
                    raise
                if time.monotonic() >= deadline:
                    raise RuntimeError('Timed out waiting for native threat compilation') from exc
                time.sleep(0.05)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _configure_library(library):
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    library.native_threat_storage_version.argtypes = []
    library.native_threat_storage_version.restype = integer
    library.native_threat_context_size.argtypes = []
    library.native_threat_context_size.restype = ctypes.c_ulonglong
    library.native_threat_base_context.argtypes = [ptr]
    library.native_threat_base_context.restype = ptr
    library.native_threat_solve.argtypes = [ptr, ctypes.POINTER(ctypes.c_ubyte), integer,
        integer, integer, ctypes.c_double, integer, integer, integer, integer,
        integer, integer, integer, _NOW, ctypes.POINTER(NativeThreatOutput)]
    library.native_threat_solve.restype = integer
    library.native_threat_solve_range.argtypes = library.native_threat_solve.argtypes + [integer]
    library.native_threat_solve_range.restype = integer
    for name, restype, indexed in (
        ('node', ctypes.POINTER(NativeThreatNode), True),
        ('edge', ctypes.POINTER(NativeThreatEdge), True),
        ('lines', ctypes.POINTER(integer), False),
        ('pv', ctypes.POINTER(integer), False),
        ('iteration', ctypes.POINTER(NativeThreatIteration), True),
        ('board', ctypes.POINTER(ctypes.c_ubyte), False),
        ('line_count', integer, False),
        ('group_count', integer, False),
        ('group_capacity', integer, False),
    ):
        function = getattr(library, 'native_threat_' + name)
        function.argtypes = [ptr, integer] if indexed else [ptr]
        function.restype = restype
    return library


def native_threat_library():
    """Compile/load the exact pair of C sources into a separate hash directory."""
    with _BUILD_LOCK:
        sources = {name: (_ROOT / name).read_bytes()
                   for name in ('native_threat.c', 'native_board.c')}
        source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
        identity = hashlib.sha256(json.dumps(source_hashes, sort_keys=True,
            separators=(',', ':')).encode('ascii')).hexdigest()
        if identity in _LIBRARIES:
            return _LIBRARIES[identity]
        destination = _ROOT / 'exports' / 'native_threat' / identity
        destination.mkdir(parents=True, exist_ok=True)
        binary, manifest = destination / 'native_threat.dll', destination / 'build.json'
        with _process_build_lock(destination / '.build.lock'):
            for name, data in sources.items():
                target = destination / name
                if target.exists() and target.read_bytes() != data:
                    raise RuntimeError('Frozen native threat source hash mismatch')
                if not target.exists():
                    target.write_bytes(data)
            if binary.exists() and manifest.exists():
                metadata = json.loads(manifest.read_text(encoding='utf-8'))
                if (metadata.get('source_sha256') != source_hashes or
                        metadata.get('binary_sha256') != hashlib.sha256(binary.read_bytes()).hexdigest()):
                    raise RuntimeError('Existing native threat binary manifest mismatch')
            else:
                clang, linker = shutil.which('clang'), shutil.which('lld-link')
                if not clang or not linker:
                    raise RuntimeError('Native threat compilation requires clang and lld-link')
                suffix = uuid.uuid4().hex
                obj = destination / f'build_{suffix}.obj'
                temporary = destination / f'build_{suffix}.dll'
                temporary_manifest = destination / f'build_{suffix}.json'
                compile_command = [clang, '-c', str(destination / 'native_threat.c'), '-o', str(obj),
                                   '-O3', '-ffreestanding', '-fno-builtin']
                link_command = [linker, '/dll', '/noentry', '/nodefaultlib', '/out:' + str(temporary), str(obj)]
                try:
                    for command in (compile_command, link_command):
                        try:
                            subprocess.run(command, check=True, capture_output=True, text=True,
                                           encoding='utf-8', errors='replace', timeout=30)
                        except subprocess.CalledProcessError as exc:
                            raise RuntimeError(f'Native threat build failed: {exc.stderr.strip()}') from exc
                    metadata = dict(format='native_threat_build_v1', source_sha256=source_hashes,
                        combined_sha256=identity, binary_sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(),
                        compile_command=compile_command, link_command=link_command)
                    temporary_manifest.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
                    os.replace(temporary, binary)
                    os.replace(temporary_manifest, manifest)
                finally:
                    for artifact in (obj, temporary, temporary_manifest, temporary.with_suffix('.lib'), temporary.with_suffix('.exp')):
                        artifact.unlink(missing_ok=True)
        library = _configure_library(ctypes.CDLL(str(binary)))
        if library.native_threat_storage_version() != 2:
            raise RuntimeError('Native threat certificate storage version must be 2')
        context_size = int(library.native_threat_context_size())
        if not 1 <= context_size <= _MAX_CONTEXT_BYTES:
            raise RuntimeError('Native threat context size is invalid')
        library.context_size = context_size
        library.combined_sha256 = identity
        library.source_sha256 = source_hashes
        library.binary_path = str(binary)
        library.binary_sha256 = metadata['binary_sha256']
        _LIBRARIES[identity] = library
        return library


def native_threat_fingerprint():
    library = native_threat_library()
    return dict(engine='native_threat_v1', source_sha256=dict(library.source_sha256),
                storage_version=int(library.native_threat_storage_version()),
                combined_sha256=library.combined_sha256, binary_path=library.binary_path,
                binary_sha256=library.binary_sha256)


def _checked_pointer(pointer, context, ctype, count=1):
    if count < 0:
        raise RuntimeError('Negative native pointer element count')
    if count == 0:
        return pointer
    address = ctypes.cast(pointer, ctypes.c_void_p).value
    start, size = ctypes.addressof(context), ctypes.sizeof(context)
    if (address is None or address < start or address + ctypes.sizeof(ctype) * count > start + size
            or address % ctypes.alignment(ctype)):
        raise RuntimeError('Native threat returned an invalid context pointer')
    return pointer


def _won_at(flat, cols, point, side):
    rows = len(flat) // cols
    row, col = divmod(point, cols)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < rows and 0 <= c < cols and flat[r * cols + c] == side:
                count += 1
                r, c = r + sign * dr, c + sign * dc
        if count >= 5:
            return True
    return False


def _decode_line(points, board, side, total):
    if not isinstance(board, np.ndarray) or board.ndim != 2:
        raise RuntimeError('Internal line board must retain its dimensions')
    if not 1 <= len(points) <= min(total, board.size):
        raise RuntimeError('Native threat returned an invalid winning line length')
    flat, columns = bytearray(board.tobytes()), board.shape[1]
    result, actor, won = [], side, False
    for point in points:
        if won or not 0 <= point < len(flat) or flat[point] != 0:
            raise RuntimeError('Native threat returned an illegal continuation action')
        flat[point] = actor
        won = _won_at(flat, columns, point, actor)
        result.append(dict(side=actor, move=int(point)))
        actor = 3 - actor
    if not won or result[-1]['side'] != side:
        raise RuntimeError('Native threat continuation does not end at its claimed winner')
    return result


def _decode_proof(library, context, output, board, side, total, line_count, include_certificate):
    cells, columns = board.size, board.shape[1]
    pv_pointer = _checked_pointer(library.native_threat_pv(context), context, ctypes.c_int, output.pv_length)
    pv = [int(pv_pointer[i]) for i in range(output.pv_length)]
    line = _decode_line(pv, board, side, total)
    if pv[0] != output.move:
        raise RuntimeError('Native root move and PV disagree')
    if not include_certificate:
        return line, None
    if output.root_id == -1:
        return line, dict(kind='forcing_line_certificate', result=dict(value=1, move=output.move,
            line=line, certifiedReplies=0), board=board.tolist(), side=side)
    lines = _checked_pointer(library.native_threat_lines(context), context, ctypes.c_int, line_count)
    nodes, edges, visiting, remap, records = {}, {}, set(), {}, []

    def node_at(index):
        if not 0 <= index < output.certificate_nodes:
            raise RuntimeError('Native certificate references an invalid node')
        if index not in nodes:
            pointer = _checked_pointer(library.native_threat_node(context, index), context, NativeThreatNode)
            node = pointer.contents
            if (node.side != side or not 0 <= node.move < cells or node.first_edge < 0 or
                    node.edge_count < 0 or node.first_edge + node.edge_count > output.certificate_edges):
                raise RuntimeError('Native certificate node fields are invalid')
            flat = bytes(node.board[:cells])
            if any(cell > 3 for cell in flat):
                raise RuntimeError('Native certificate board has invalid cells')
            nodes[index] = (node.move, node.side, node.first_edge, node.edge_count, flat)
        return nodes[index]

    def edge_at(index):
        if index not in edges:
            pointer = _checked_pointer(library.native_threat_edge(context, index), context, NativeThreatEdge)
            # Storage v2 reuses one scratch edge. Copy fields before any getter
            # or recursive visit; a ctypes contents view remains mutable.
            reply, child_id, offset, length = (int(getattr(pointer.contents, name))
                for name in ('reply', 'child_id', 'line_offset', 'line_length'))
            if (not 0 <= reply < cells or not -1 <= child_id < output.certificate_nodes or
                    offset < 0 or length < 1 or offset + length > line_count):
                raise RuntimeError('Native certificate edge fields are invalid')
            edges[index] = (reply, child_id, offset, length)
        return edges[index]

    def visit(index, before, depth):
        if index in visiting or depth > total:
            raise RuntimeError('Native certificate contains a cycle or exceeds its ply horizon')
        move, actor, first, count, flat = node_at(index)
        attacked = before.copy()
        if attacked.flat[move] != 0:
            raise RuntimeError('Native certificate attack is not legal')
        attacked.flat[move] = actor
        if attacked.tobytes() != flat:
            raise RuntimeError('Native certificate board differs from its parent action')
        if index in remap:
            return remap[index]
        visiting.add(index)
        terminal = _won_at(flat, columns, move, actor)
        legal = set() if terminal else {i for i, cell in enumerate(flat) if cell == 0}
        supplied, replies = set(), []
        for edge_index in range(first, first + count):
            reply, child_id, offset, length = edge_at(edge_index)
            if reply in supplied or reply not in legal:
                raise RuntimeError('Native certificate has a duplicate or illegal defender reply')
            supplied.add(reply)
            child = attacked.copy()
            child.flat[reply] = 3 - side
            if _won_at(child.ravel(), columns, reply, 3 - side) or not np.any(child == 0):
                raise RuntimeError('Native certificate includes a defender terminal or draw')
            continuation = _decode_line([int(lines[k]) for k in range(offset, offset + length)], child, side, total - depth - 2)
            nested = None if child_id < 0 else visit(child_id, child, depth + 2)
            if nested is not None and continuation[0]['move'] != records[nested]['move']:
                raise RuntimeError('Native child PV differs from its certificate attack')
            replies.append(dict(move=reply, value=1, line=continuation, auditId=nested))
        if supplied != legal or (not legal and not terminal):
            raise RuntimeError('Native certificate omitted legal replies or claimed a full-board draw')
        visiting.remove(index)
        remap[index] = len(records)
        records.append(dict(move=move, side=side, board=list(flat), replies=replies,
                            certifiedReplies=len(replies)))
        return remap[index]

    root = visit(output.root_id, board, 0)
    if records[root]['move'] != output.move or len(records[root]['replies']) != output.certified_replies:
        raise RuntimeError('Native root identity or certified reply count disagrees')
    if len(records) != output.certificate_nodes or len(edges) != output.certificate_edges:
        raise RuntimeError('Native certificate contains unreachable nodes or logical edges')
    return line, dict(kind='complete_all_replies_certificate', result=dict(value=1,
        move=output.move, line=line, auditId=root, certifiedReplies=output.certified_replies), certificates=records)


def solve_native_threat(grid, side, *, milliseconds=200, max_nodes=20000, quiet=2,
                        total=64, width=16, node_capacity=2048, edge_capacity=65536,
                        line_capacity=262144, certificate=True, minimum_quiet=None):
    """Return a completed C proof or explicit unknown, using a private context.

    ``move`` is a row-major flat index. ``elapsedMs`` covers only the C solve
    (including host clock callbacks), while ``decodeMs`` covers result decoding.
    Independent proof verification is available via verify_native_threat_result.
    ``edge_capacity`` bounds physical groups; ``certificate_edges`` counts all
    defender replies. ``minimum_quiet`` starts at a later quiet iteration within
    this call, without retaining evidence from any previous call.
    """
    board = _validated_board(grid)
    side = _integer(side, 'side', 1, 2)
    milliseconds = _finite_milliseconds(milliseconds)
    max_nodes = _integer(max_nodes, 'max_nodes', 0, _INT_MAX)
    quiet = _integer(quiet, 'quiet', 0, 4)
    minimum = (0 if quiet == 0 else 1) if minimum_quiet is None else _integer(
        minimum_quiet, 'minimum_quiet', 0 if quiet == 0 else 1, quiet)
    total = _integer(total, 'total', 1, 64)
    width = _integer(width, 'width', 1, 64)
    capacities = {name: _integer(value, name, 0, _CAPACITIES[name]) for name, value in
        [('node_capacity', node_capacity), ('edge_capacity', edge_capacity), ('line_capacity', line_capacity)]}
    if type(certificate) is not bool:
        raise ValueError('certificate must be a bool')
    if getattr(_LOCAL, 'busy', False):
        raise RuntimeError('Native threat context cannot be re-entered on the same thread')
    _LOCAL.busy = True
    try:
        setup_started = time.perf_counter()
        library = native_threat_library()
        storage_version = int(library.native_threat_storage_version())
        if storage_version != 2:
            raise RuntimeError('Native threat certificate storage version must be 2')
        key = library.combined_sha256
        if getattr(_LOCAL, 'context_key', None) != key:
            _LOCAL.context = ctypes.create_string_buffer(library.context_size)
            _LOCAL.context_key = key
        context = _LOCAL.context
        ctypes.memset(ctypes.addressof(context), 0, ctypes.sizeof(context))
        setup_ms = (time.perf_counter() - setup_started) * 1000
        callback_errors = []

        @_NOW
        def now():
            if callback_errors:
                return math.inf
            try:
                value = float(_clock_ms())
                if not math.isfinite(value):
                    raise ValueError('Host clock returned a non-finite value')
                return value
            except BaseException as exc:
                callback_errors.append(exc)
                return math.inf

        output = NativeThreatOutput()
        input_bytes = board.tobytes()
        started = time.perf_counter()
        arguments = (context, board.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            board.shape[0], board.shape[1], side, milliseconds, max_nodes, quiet, total, width,
            capacities['node_capacity'], capacities['edge_capacity'], capacities['line_capacity'], now, ctypes.byref(output))
        code = (library.native_threat_solve(*arguments) if minimum_quiet is None else
                library.native_threat_solve_range(*arguments, minimum))
        search_ms = (time.perf_counter() - started) * 1000
        if callback_errors:
            raise RuntimeError('Native threat host clock callback failed; result discarded') from callback_errors[0]
        if code != 0:
            raise RuntimeError(f'Native threat rejected request: {code}')
        decode_started = time.perf_counter()
        if (output.value not in (1, 2) or output.exhausted not in (0, 1) or
            not 0 <= output.nodes <= max_nodes or not -1 <= output.move < board.size or
            not 0 <= output.certificate_nodes <= capacities['node_capacity'] or
            not 0 <= output.certificate_edges <= output.certificate_nodes * board.size or
            not -1 <= output.root_id < output.certificate_nodes or
            not 0 <= output.pv_length <= min(total, board.size) or
            not 0 <= output.certified_replies <= board.size or
            not 0 <= output.completed_quiet <= quiet or not 0 <= output.iteration_count <= 32 or
            output.status not in (0, 1, 2, 3, 4)):
            raise RuntimeError('Native threat output fields are out of bounds')
        if output.completed_quiet and output.completed_quiet < minimum:
            raise RuntimeError('Native threat completed an iteration below the requested minimum')
        if (output.value == 1 and output.status not in (1, 4) or
            output.value == 2 and output.status == 1 or
            bool(output.exhausted) != (output.status in (2, 3))):
            raise RuntimeError('Native threat status disagrees with its proof or exhaustion flag')
        if board.tobytes() != input_bytes:
            raise RuntimeError('Native threat changed the supplied input buffer')
        restored = _checked_pointer(library.native_threat_board(context), context, ctypes.c_ubyte, board.size)
        if bytes(restored[:board.size]) != input_bytes:
            raise RuntimeError('Native threat did not restore its context board')
        line_count = int(library.native_threat_line_count(context))
        if not 0 <= line_count <= capacities['line_capacity']:
            raise RuntimeError('Native certificate line count exceeds capacity')
        group_count = int(library.native_threat_group_count(context))
        group_capacity = int(library.native_threat_group_capacity(context))
        if (group_capacity != capacities['edge_capacity'] or
                not 0 <= group_count <= min(group_capacity, output.certificate_edges) or
                bool(group_count) != bool(output.certificate_edges)):
            raise RuntimeError('Native certificate physical group counts are invalid')
        iterations = []
        for index in range(output.iteration_count):
            pointer = _checked_pointer(library.native_threat_iteration(context, index), context, NativeThreatIteration)
            iteration = pointer.contents
            if (iteration.ordering not in (0, 1) or not minimum <= iteration.quiet <= quiet or
                iteration.leaf_ms not in (5, 50) or not 0 <= iteration.nodes <= max_nodes or iteration.completed not in (0, 1)):
                raise RuntimeError('Native threat iteration fields are invalid')
            iterations.append(dict(ordering=('quiet_first', 'natural')[iteration.ordering],
                quiet=iteration.quiet, leafMs=iteration.leaf_ms, nodes=iteration.nodes, completed=bool(iteration.completed)))
        if iterations and sum(iteration['nodes'] for iteration in iterations) != output.nodes:
            raise RuntimeError('Native threat iteration nodes disagree with the total node count')
        line, proof = [], None
        if output.status == 4:
            from board_rules import board_winner
            winner = board_winner(board)
            if (output.move != -1 or output.pv_length or output.root_id != -1 or output.certified_replies or
                output.certificate_nodes or output.certificate_edges or line_count or output.iteration_count or
                (not winner and np.any(board == 0)) or (output.value == 1) != (winner == side)):
                raise RuntimeError('Native threat terminal result disagrees with the actual board')
            if output.value == 1 and certificate:
                proof = dict(kind='terminal_state', board=board.tolist(), winner=side)
        elif output.value == 1:
            from board_rules import board_winner
            if output.move < 0 or board.flat[output.move] != 0 or board_winner(board):
                raise RuntimeError('Native threat claimed an illegal root action')
            if output.root_id == -1 and (output.certified_replies or output.certificate_nodes or
                    output.certificate_edges or line_count or group_count):
                raise RuntimeError('Native line-only proof cannot claim a complete reply graph')
            line, proof = _decode_proof(library, context, output, board, side, total, line_count, certificate)
        elif (output.move != -1 or output.pv_length or output.root_id != -1 or output.certified_replies or
                output.certificate_nodes or output.certificate_edges or line_count or group_count):
            raise RuntimeError('Native unknown result contains an uncompleted root proof')
        result = dict(value=1 if output.value == 1 else None, move=None if output.move < 0 else output.move,
            coordinate=None if output.move < 0 else divmod(output.move, board.shape[1]), line=line,
            nodes=output.nodes, elapsedMs=search_ms, decodeMs=(time.perf_counter() - decode_started) * 1000,
            setupMs=setup_ms, exhausted=bool(output.exhausted), certifiedReplies=output.certified_replies,
            completedQuiet=output.completed_quiet, iterations=iterations, status=output.status,
            root_id=output.root_id, proof=proof, certificateDecoded=certificate and output.value == 1,
            certificate_nodes=output.certificate_nodes, certificate_edges=output.certificate_edges,
            certificate_lines=line_count, storage_version=storage_version,
            certificate_groups=group_count, group_capacity=group_capacity, minimumQuiet=minimum,
            engine='native_threat_v1', source_sha256=dict(library.source_sha256),
            binary_sha256=library.binary_sha256)
        return result
    finally:
        _LOCAL.busy = False


def verify_native_threat_result(grid, side, result):
    """Independently replay a returned proof; never run native/Python search.

    An already-won board is a ``terminal_state`` fact and has no root action.
    It must not be passed off as a complete all-replies action certificate.
    """
    from board_rules import board_winner
    from global_policy_constraints import _replay_line, verify_action_certificate
    board = _validated_board(grid)
    side = _integer(side, 'side', 1, 2)
    if not isinstance(result, dict):
        raise ValueError('Native result must be an object')
    _integer(result.get('value'), 'completed positive value', 1, 1)
    proof = result.get('proof')
    if not isinstance(proof, dict):
        raise ValueError('Result has no decoded certificate')
    counts = dict(attacker_nodes=0, defender_branches=0, moves_replayed=0,
                  forced_lines=0, double_win_facts=0)
    if proof.get('kind') == 'terminal_state':
        winner = _integer(proof.get('winner'), 'terminal winner', 1, 2)
        if (result.get('move') is not None or result.get('coordinate') is not None or
            result.get('line') != [] or winner != side or board_winner(board) != side or
            not np.array_equal(_validated_board(proof.get('board')), board)):
            raise ValueError('Terminal certificate differs from the already-won input board')
        return dict(kind='terminal_state', **counts)
    point = _integer(result.get('move'), 'result move', 0, board.size - 1)
    coordinate = divmod(point, board.shape[1])
    if 'coordinate' in result:
        supplied = result['coordinate']
        if not isinstance(supplied, (tuple, list)) or len(supplied) != 2:
            raise ValueError('Result coordinate does not match its flat move')
        converted = tuple(_integer(cell, 'coordinate', 0, board.shape[axis] - 1)
                          for axis, cell in enumerate(supplied))
        if converted != coordinate:
            raise ValueError('Result coordinate does not match its flat move')
    proof_result = proof.get('result')
    if not isinstance(proof_result, dict):
        raise ValueError('Certificate result must be an object')
    if proof.get('kind') == 'complete_all_replies_certificate':
        if proof_result.get('line') != result.get('line'):
            raise ValueError('Result line differs from certificate line')
        return verify_action_certificate(board, side, coordinate, 1, proof)
    if proof.get('kind') != 'forcing_line_certificate':
        raise ValueError('Unsupported native proof kind')
    proof_side = _integer(proof.get('side'), 'proof side', 1, 2)
    proof_move = _integer(proof_result.get('move'), 'proof move', 0, board.size - 1)
    _integer(proof_result.get('value'), 'proof positive value', 1, 1)
    if (proof_side != side or not np.array_equal(_validated_board(proof.get('board')), board) or
        proof_result.get('line') != result.get('line') or proof_move != point):
        raise ValueError('Forcing certificate does not match its requested board or root')
    line = result.get('line')
    if not isinstance(line, list) or not line or not isinstance(line[0], dict) or line[0].get('move') != point:
        raise ValueError('Forcing result has an invalid first action')
    _replay_line(board, side, line, side, counts, forcing=True)
    return dict(kind=proof['kind'], **counts)
