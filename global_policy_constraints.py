"""Import action sets proved by complete, independently replayed strategies.

This module never searches, infers from a heuristic, or trains. JSONL[.gz] rows
use format ``global_policy_constraints_v1`` and require board, side, game_id,
source_file/source_sha256, prefix_plies, and action_evidence. Each evidence item
has move=[row,col], actor_value=+1/-1, proof_actor, proof_file/proof_file_sha256.
Paths are relative to the JSONL file. The source must replay to the exact input
prefix; output game_id is derived from its SHA, never its caller-supplied name.

Supported proofs are browser ``complete_all_replies_certificate`` DAGs (flat
indices use the input board width) and ``terminal_after_action`` with board and
winner. DAG nodes store the board AFTER their attacker move. A negative action
starts before the root attack; a positive action IS the root attack. All real
defender replies must be present. Unlinked continuation lines are accepted only
when independently forced: a defender has one mandatory block, or two distinct
attacker winning points and no counter-win. A merely legal PV is insufficient.
"""
import gzip
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from board_rules import apply_board_move, board_winner, legal_cells, normalize_board, winning_cells

FORMAT = 'global_policy_constraints_v1'


def _int(value, name, low=0, high=None):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < low or (high is not None and value > high):
        raise ValueError(f'{name} must be an integer in the permitted range')
    return int(value)


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be an object')
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value.strip()


def _board(value, shape=None):
    # numpy otherwise silently coerces a bool mixed with integers to an int.
    if isinstance(value, (list, tuple)):
        for row in value:
            if not isinstance(row, (list, tuple, np.ndarray)):
                raise ValueError('board must have two dimensions')
            for cell in row:
                _int(cell, 'board cell', 0, 3)
    board = normalize_board(value)
    if min(board.shape) < 5 or max(board.shape) > 32:
        raise ValueError('board dimensions must be between 5 and 32')
    if shape is not None and board.shape != shape:
        raise ValueError('proof board shape differs from its parent')
    return board


def _move(value, shape):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('move must contain row and column')
    return (_int(value[0], 'row', 0, shape[0] - 1), _int(value[1], 'col', 0, shape[1] - 1))


def _point(value, shape):
    return divmod(_int(value, 'flat move', 0, shape[0] * shape[1] - 1), shape[1])


def _transform(array, turns, reflected):
    transformed = np.rot90(array, turns)
    return np.fliplr(transformed) if reflected else transformed


def _actor_key(board, side):
    normalized = np.array([0, 2, 1, 3], dtype=np.uint8)[board] if side == 2 else board
    shape, cells = min((tuple(a.shape), a.tobytes()) for turns in range(4)
                       for a in (_transform(normalized, turns, False), _transform(normalized, turns, True)))
    return f'{shape[0]}x{shape[1]}:' + cells.hex()


def _physical_key(board):
    return min(_actor_key(board, 1), _actor_key(board, 2))


def _mask(value, board, name):
    mask = np.asarray(value)
    if mask.shape != board.shape or mask.dtype.kind != 'b':
        raise ValueError(f'{name} must be a boolean board-shaped mask')
    if np.any(mask & (board != 0)):
        raise ValueError(f'{name} contains occupied or forbidden actions')
    return mask.copy()


def align_constraint(record, board, side):
    """Map verified action masks to an equivalent board+actor, else return None.

    All exact D4 matches are unioned, including symmetries of the same board.
    Color reversal always reverses side too and leaves 0/3 unchanged. Metadata
    describes every valid transformation from the original evidence frame;
    proof files remain in their original coordinates and are not rewritten.
    """
    source = _board(record['board'])
    target = _board(board)
    source_side = _int(record['side'], 'source side', 1, 2)
    side = _int(side, 'side', 1, 2)
    losing = _mask(record['losing_mask'], source, 'losing_mask')
    winning = _mask(record['winning_mask'], source, 'winning_mask')
    if np.any(losing & winning):
        raise ValueError('winning and losing action sets conflict')
    color_swap = source_side != side
    colored = np.array([0, 2, 1, 3], dtype=np.uint8)[source] if color_swap else source
    bad, good = np.zeros(target.shape, bool), np.zeros(target.shape, bool)
    transforms = []
    for turns in range(4):
        for reflected in (False, True):
            candidate = _transform(colored, turns, reflected)
            if candidate.shape != target.shape or not np.array_equal(candidate, target):
                continue
            bad |= _transform(losing, turns, reflected)
            good |= _transform(winning, turns, reflected)
            transforms.append(dict(quarter_turns_ccw=turns, reflect_left_right=reflected,
                                   swap_colors_and_side=color_swap))
    if not transforms:
        return None
    if np.any(bad & good):
        raise ValueError('equivalent proof actions conflict under a board symmetry')
    return dict(losing_mask=bad, winning_mask=good, evidence_transforms=transforms,
                source_board_key=_actor_key(source, source_side), target_board_key=_actor_key(target, side))


def _replay_line(board, first_side, line, winner, counts, *, forcing):
    if not isinstance(line, list) or not line or len(line) > board.size:
        raise ValueError('winning continuation must be a nonempty bounded line')
    side = first_side
    forced_double = False
    for entry in line:
        entry = _object(entry, 'continuation move')
        if _int(entry.get('side'), 'line side', 1, 2) != side:
            raise ValueError('continuation side does not alternate')
        point = _point(entry.get('move'), board.shape)
        if forcing and not forced_double and side != winner:
            if winning_cells(board, side):
                raise ValueError('defender has an immediate counter-win')
            threats = winning_cells(board, winner)
            if not threats:
                raise ValueError('a free defender choice cannot be certified by a PV')
            if len(threats) == 1 and point != threats[0]:
                raise ValueError('PV omitted the unique mandatory defense')
            # Every defense loses at two distinct completion points. The rest
            # remains a legal representative line, not the proof of all choices.
            if len(threats) >= 2:
                forced_double = True
                counts['double_win_facts'] += 1
        board = apply_board_move(board, point, side)
        counts['moves_replayed'] += 1
        side = 3 - side
    if board_winner(board) != winner:
        raise ValueError('continuation does not end at the certified winner')
    if forcing:
        counts['forced_lines'] += 1


def verify_action_certificate(board, side, move, actor_value, proof):
    """Independently prove one legal action from the actual parent position.

    Returns deterministic counts. Full-reply DAGs are checked against actual
    rules and identity, not trusted ``value``, ``verified`` or count fields.
    """
    board = _board(board)
    side = _int(side, 'side', 1, 2)
    move = _move(move, board.shape)
    if isinstance(actor_value, bool) or not isinstance(actor_value, Integral) or actor_value not in (-1, 1):
        raise ValueError('actor_value must be -1 or +1')
    after = apply_board_move(board, move, side)
    winner = side if actor_value == 1 else 3 - side
    proof = _object(proof, 'proof')
    counts = dict(attacker_nodes=0, defender_branches=0, moves_replayed=0,
                  forced_lines=0, double_win_facts=0)
    if proof.get('kind') == 'terminal_after_action':
        if actor_value != 1 or board_winner(after) != side:
            raise ValueError('terminal action is not an actual immediate win')
        if _int(proof.get('winner'), 'winner', 1, 2) != side or not np.array_equal(_board(proof.get('board'), board.shape), after):
            raise ValueError('terminal certificate identity mismatch')
        return dict(kind=proof['kind'], **counts)
    if proof.get('kind') != 'complete_all_replies_certificate':
        raise ValueError('proof must contain a complete all-replies certificate, not just a PV')
    result = _object(proof.get('result'), 'proof result')
    records = proof.get('certificates')
    if not isinstance(records, list) or not records:
        raise ValueError('complete certificate needs its defender nodes')
    if _int(result.get('value'), 'proof value', 1, 1) != 1:
        raise ValueError('proof is not a positive strategy')
    root_id = _int(result.get('auditId'), 'root auditId', 0, len(records) - 1)
    root = _object(records[root_id], 'root node')
    root_move = _point(root.get('move'), board.shape)
    if _point(result.get('move'), board.shape) != root_move:
        raise ValueError('result move differs from root node')
    parsed = {}
    visited = set()

    def node_board(index):
        if index not in parsed:
            node = _object(records[index], 'certificate node')
            flat = node.get('board')
            if not isinstance(flat, list) or len(flat) != board.size:
                raise ValueError('node board must contain every flat cell')
            for cell in flat:
                _int(cell, 'node board cell', 0, 3)
            parsed[index] = _board(np.array(flat, dtype=np.uint8).reshape(board.shape))
        return parsed[index]

    root_after = node_board(root_id)
    if actor_value == 1:
        if root_move != move or not np.array_equal(root_after, after):
            raise ValueError('positive certificate must start after exactly the labeled action')
        root_before = board
    else:
        root_before = after
        if not np.array_equal(apply_board_move(after, root_move, winner), root_after):
            raise ValueError('negative certificate must start at the labeled action successor')

    def visit(index, expected_before):
        node = _object(records[index], 'certificate node')
        if _int(node.get('side'), 'attacker side', 1, 2) != winner:
            raise ValueError('certificate attacker color mismatch')
        point = _point(node.get('move'), board.shape)
        attacked = apply_board_move(expected_before, point, winner)
        if not np.array_equal(attacked, node_board(index)):
            raise ValueError('nested certificate board identity mismatch')
        if index in visited:
            return
        visited.add(index)
        counts['attacker_nodes'] += 1
        counts['moves_replayed'] += 1
        if node.get('mandatory', False) is not False:
            if node['mandatory'] is not True or winning_cells(expected_before, 3 - winner) != [point]:
                raise ValueError('false mandatory-defense claim')
        replies = node.get('replies')
        if not isinstance(replies, list):
            raise ValueError('defender replies must be a list')
        actual = set(legal_cells(attacked))
        supplied = [_point(_object(reply, 'reply').get('move'), board.shape) for reply in replies]
        if len(supplied) != len(set(supplied)) or set(supplied) != actual:
            raise ValueError('missing, duplicate or illegal defender reply')
        if _int(node.get('certifiedReplies'), 'certified reply count') != len(actual):
            raise ValueError('certified reply count differs from actual legal set')
        if not actual and board_winner(attacked) != winner:
            raise ValueError('a full draw board is not a winning certificate')
        for reply, response in zip(replies, supplied):
            counts['defender_branches'] += 1
            if _int(reply.get('value'), 'reply value', 1, 1) != 1:
                raise ValueError('defender branch is not proved')
            child = apply_board_move(attacked, response, 3 - winner)
            if board_winner(child) or not np.any(child == 0):
                raise ValueError('defender terminal or draw refutes the winning strategy')
            nested = reply.get('auditId')
            if nested is None:
                _replay_line(child, winner, reply.get('line'), winner, counts, forcing=True)
            else:
                nested = _int(nested, 'nested auditId', 0, index - 1)
                visit(nested, child)
                line = reply.get('line')
                if not line or _point(_object(line[0], 'nested PV first move').get('move'), board.shape) != _point(records[nested]['move'], board.shape):
                    raise ValueError('nested representative line has the wrong first attack')
                _replay_line(child, winner, line, winner, counts, forcing=False)

    visit(root_id, root_before)
    if len(visited) != len(records):
        raise ValueError('certificate contains detached or unreachable nodes')
    if _int(result.get('certifiedReplies'), 'root reply count') != len(root['replies']):
        raise ValueError('result reply count differs from root')
    line = result.get('line')
    if not line or _point(_object(line[0], 'PV first move').get('move'), board.shape) != root_move:
        raise ValueError('representative PV does not begin at the certified root')
    _replay_line(root_before, winner, line, winner, counts, forcing=False)
    return dict(kind=proof['kind'], **counts)


def _read_json(path, expected_sha):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or expected_sha.lower() != digest:
        raise ValueError(f'{path.name}: SHA256 mismatch')
    if path.name.lower().endswith('.gz'):
        data = gzip.decompress(data)
    return json.loads(data.decode('utf-8-sig')), digest


def _resolve(base, value, name):
    path = Path(_text(value, name))
    return (path if path.is_absolute() else base / path).resolve()


def _source_prefix(source, prefix, expected_board, side):
    source = _object(source, 'source game')
    legacy = None
    if 'board' not in source:
        # Older Python result files retain the same actual history and identity
        # twice, with the terminal board only in final_state. Do not create a
        # replacement source file or silently choose between contradictory data.
        legacy = _object(source.get('final_state'), 'legacy final_state')
        if not isinstance(source.get('history'), list) or legacy.get('history') != source['history']:
            raise ValueError('legacy root and final_state histories differ')
        if _int(source.get('plies'), 'source plies') != _int(legacy.get('move_count'), 'legacy move_count'):
            raise ValueError('legacy move_count differs from source plies')
        if _int(source.get('winner'), 'source winner', 0, 2) != _int(legacy.get('winner'), 'legacy winner', 0, 2):
            raise ValueError('legacy root and final_state winners differ')
        ai = _int(source.get('ai_side'), 'source ai_side', 1, 2)
        if _int(legacy.get('ai_side'), 'legacy ai_side', 1, 2) != ai:
            raise ValueError('legacy root and final_state AI sides differ')
        if 'human_side' in legacy and _int(legacy['human_side'], 'legacy human_side', 1, 2) != 3 - ai:
            raise ValueError('legacy human side must be opposite the AI')
        if not isinstance(legacy.get('finished'), bool):
            raise ValueError('legacy finished must be boolean')
        _int(legacy.get('turn'), 'legacy turn', 0, 2)
    final = _board(legacy.get('board') if legacy is not None else source.get('board'), expected_board.shape)
    history = source.get('history')
    if not isinstance(history, list) or prefix > len(history):
        raise ValueError('source prefix is beyond the saved history')
    # Fixed gray cells are the only allowed stones before the black first move.
    replay = np.where(final == 3, 3, 0).astype(np.uint8)
    prefix_board = replay.copy() if prefix == 0 else None
    for i, entry in enumerate(history):
        entry = _object(entry, 'source history move')
        actor = _int(entry.get('side'), 'source history side', 1, 2)
        if actor != 1 + i % 2:
            raise ValueError('source game must alternate from black')
        point = _move([entry.get('row'), entry.get('col')], replay.shape)
        replay = apply_board_move(replay, point, actor)
        if i + 1 == prefix:
            prefix_board = replay.copy()
    if not np.array_equal(replay, final):
        raise ValueError('source final board differs from its legal history')
    if 'plies' in source and _int(source['plies'], 'source plies') != len(history):
        raise ValueError('source plies differs from history length')
    actual_winner = board_winner(final)
    if 'winner' in source and _int(source['winner'], 'source winner', 0, 2) != actual_winner:
        raise ValueError('source winner differs from board rules')
    if legacy is not None:
        terminal = bool(actual_winner or not np.any(final == 0))
        if legacy['finished'] != terminal:
            raise ValueError('legacy finished disagrees with actual terminal rules')
        expected_turn = 0 if terminal else 1 + len(history) % 2
        if legacy['turn'] != expected_turn:
            raise ValueError('legacy turn disagrees with terminal or actual next actor')
    if side != 1 + prefix % 2 or not np.array_equal(prefix_board, expected_board):
        raise ValueError('input board/actor does not match the actual source prefix')


def load_policy_constraints(paths):
    """Return (records, report); fail closed on any unverified action or identity.

    Records contain zero HxW float32 target_policy, policy_mask=False, value=0,
    value_valid=False and search_proven_value=None. Legal bool action masks are
    partial information, never a claim that their complement is safe. Same
    board+actor evidence is unioned with exact coordinate transforms. Physical
    duplicates from different source games are rejected for explicit group
    reconciliation by the caller, never silently split or discarded.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records, files = [], []
    by_key, owners, source_cache, proof_cache = {}, {}, {}, {}
    totals = dict(input_rows=0, merged_rows=0, action_evidence=0)
    for filename in paths:
        path = Path(filename).resolve()
        if not path.name.lower().endswith(('.jsonl', '.jsonl.gz')):
            raise ValueError('policy constraints require .jsonl or .jsonl.gz')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        content = gzip.decompress(raw) if path.name.lower().endswith('.gz') else raw
        file_report = dict(path=str(path), sha256=digest, rows=0)
        for line_number, line in enumerate(content.decode('utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = _object(json.loads(line), 'constraint row')
                if row.get('format') != FORMAT:
                    raise ValueError(f'constraint format must be {FORMAT}')
                board = _board(row.get('board'))
                if board_winner(board) or not np.any(board == 0):
                    raise ValueError('policy constraint parent must be nonterminal')
                side = _int(row.get('side'), 'side', 1, 2)
                prefix = _int(row.get('prefix_plies'), 'prefix_plies')
                supplied_game_id = _text(row.get('game_id'), 'game_id')
                source_path = _resolve(path.parent, row.get('source_file'), 'source_file')
                source_sha = row.get('source_sha256')
                # Each referenced file is hashed from the same bytes parsed;
                # a different declared hash never hits a prior cache entry.
                source_key = (source_path, source_sha)
                if source_key not in source_cache:
                    source_cache[source_key] = _read_json(source_path, source_sha)
                source, source_sha = source_cache[source_key]
                _source_prefix(source, prefix, board, side)
                evidence = row.get('action_evidence')
                if not isinstance(evidence, list) or not evidence:
                    raise ValueError('at least one independently verifiable action is required')
                bad, good = np.zeros(board.shape, bool), np.zeros(board.shape, bool)
                provenance = []
                for item in evidence:
                    item = _object(item, 'action evidence')
                    move = _move(item.get('move'), board.shape)
                    value = item.get('actor_value')
                    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (-1, 1):
                        raise ValueError('actor_value must be -1 or +1')
                    proof_actor = _int(item.get('proof_actor'), 'proof_actor', 1, 2)
                    if proof_actor != (side if value == 1 else 3 - side):
                        raise ValueError('proof_actor disagrees with the labeled action value')
                    proof_path = _resolve(path.parent, item.get('proof_file'), 'proof_file')
                    proof_sha = item.get('proof_file_sha256')
                    proof_key = (proof_path, proof_sha)
                    if proof_key not in proof_cache:
                        proof_cache[proof_key] = _read_json(proof_path, proof_sha)
                    proof, proof_sha = proof_cache[proof_key]
                    verified = verify_action_certificate(board, side, move, value, proof)
                    (bad if value == -1 else good)[move] = True
                    provenance.append(dict(move=list(move), actor_value=int(value), proof_actor=proof_actor,
                                           proof_file=str(proof_path), proof_file_sha256=proof_sha,
                                           verification=verified))
                if np.any(bad & good):
                    raise ValueError('one action cannot be both proved winning and losing')
                key, physical = _actor_key(board, side), _physical_key(board)
                game_id = 'source-game-' + source_sha
                if physical in owners and owners[physical] != game_id:
                    raise ValueError('equivalent physical board crosses source groups; reconcile owners before importing')
                owners[physical] = game_id
                entry = dict(board=board, side=side, losing_mask=bad, winning_mask=good,
                             target_policy=np.zeros(board.shape, dtype=np.float32), policy_mask=False,
                             policy_valid=False, value=0., value_valid=False, search_proven_value=None,
                             terminal_value=None, terminal_value_valid=False, game_terminal=False,
                             game_winner=None, game_termination='action_constraints',
                             game_id=game_id, group_id=game_id, group_kind='constructed_position',
                             source='verified_action_constraints', policy_source='verified_action_sets',
                             search_requested_depth=0, search_completed_depth=0, value_source='unknown',
                             source_file=str(source_path), source_sha256=source_sha, prefix_plies=prefix,
                             source_game_id=supplied_game_id, board_key=key, physical_key=physical,
                             action_evidence=provenance,
                             constraint_imports=[dict(path=str(path), sha256=digest, line=line_number)],
                             evidence_alignments=[])
                # Include board automorphisms as valid transformed certificates.
                aligned = align_constraint(entry, board, side)
                entry['losing_mask'], entry['winning_mask'] = aligned['losing_mask'], aligned['winning_mask']
                entry['evidence_alignments'].append(aligned['evidence_transforms'])
                if key in by_key:
                    existing = by_key[key]
                    aligned = align_constraint(entry, existing['board'], existing['side'])
                    bad = existing['losing_mask'] | aligned['losing_mask']
                    good = existing['winning_mask'] | aligned['winning_mask']
                    if np.any(bad & good):
                        raise ValueError('merged action certificates conflict')
                    existing['losing_mask'], existing['winning_mask'] = bad, good
                    # Each evidence retains its input frame; explicit transforms
                    # tell consumers how it maps onto the retained board frame.
                    existing['action_evidence'].extend(provenance)
                    existing['constraint_imports'].extend(entry['constraint_imports'])
                    existing['evidence_alignments'].append(aligned['evidence_transforms'])
                    totals['merged_rows'] += 1
                else:
                    by_key[key] = entry
                    records.append(entry)
                totals['input_rows'] += 1
                totals['action_evidence'] += len(evidence)
                file_report['rows'] += 1
            except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc
        files.append(file_report)
    report = dict(records=len(records), groups=len({r['game_id'] for r in records}),
                  losing_actions=sum(int(r['losing_mask'].sum()) for r in records),
                  winning_actions=sum(int(r['winning_mask'].sum()) for r in records),
                  whole_position_values=0, files=files, **totals,
                  proof_verification='Independent legal source replay and complete defender-branch verification; no search and no trusted verified flags.')
    return records, report
