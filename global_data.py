"""Whole-board search trajectories for a global policy/value model.

Only actually searched actions are labeled. Randomized setup moves are opening
context, not imitation targets. Each board keeps its actual acting color;
counterfactual labels are never invented. Seed controls openings and exploration,
while a wall-clock search limit can change completed depths across machines.
"""

from collections import Counter
import gzip
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import random
import time

import numpy as np

from board_rules import BLACK, WHITE, EMPTY, FORBIDDEN
from board_rules import normalize_board, board_winner, legal_cells, tactical_candidates, apply_board_move
from board_search import select_move


TEACHER_ENGINES = ("python", "native")
VALUE_SUPERVISION_MODES = ("proven_or_terminal", "proven")


def record_group(record):
    """Separate played trajectories from independently constructed positions."""
    return ("constructed" if record.get("group_kind") == "constructed_position"
            or record.get("source") in ("constructed_verified_full_board", "extra_verified_full_board")
            else "real_games")


def teacher_fingerprint(engine):
    """Identify the implementation actually used; native includes its loaded DLL."""
    if engine not in TEACHER_ENGINES:
        raise ValueError("teacher_engine must be python or native")
    root = Path(__file__).resolve().parent
    if engine == "native":
        from native_search import native_fingerprint
        fingerprint = dict(native_fingerprint())
        fingerprint["wrapper_sha256"] = hashlib.sha256((root / "native_search.py").read_bytes()).hexdigest()
        fingerprint["forcing_source_sha256"] = hashlib.sha256((root / "board_forcing.py").read_bytes()).hexdigest()
        return fingerprint
    return dict(engine="python", source_sha256=hashlib.sha256((root / "board_search.py").read_bytes()).hexdigest(),
                binary_sha256=None, binary_path=None)


def apply_value_supervision(records, value_supervision="proven_or_terminal"):
    """Update labels in memory while preserving actual terminal outcomes.

    The legacy default prefers the played terminal outcome, falling back to a
    search proof on truncations. proven uses only explicit search certificates.
    Unknown value placeholders remain zero with value_valid=False. This function
    also supports old cached records; no serialized file is changed.
    """
    if value_supervision not in VALUE_SUPERVISION_MODES:
        raise ValueError("value_supervision must be proven_or_terminal or proven")
    for record in records:
        side = record["side"]
        if isinstance(side, bool) or not isinstance(side, Integral) or side not in (BLACK, WHITE):
            raise ValueError("record side must be BLACK=1 or WHITE=2")
        proof = record.get("search_proven_value")
        if proof is not None and (isinstance(proof, bool) or proof not in (-1, 0, 1)):
            raise ValueError("search returned an invalid proven value")
        # Old full caches carry game_winner; compact exports may retain only
        # the explicit actor-relative terminal label. Neither may contradict
        # the other, and an unknown zero is never promoted to a draw.
        terminal_flag = record.get("game_terminal", False)
        retained_flag = record.get("terminal_value_valid", False)
        if not isinstance(terminal_flag, bool) or not isinstance(retained_flag, bool):
            raise ValueError("terminal validity flags must be boolean")
        terminal = terminal_flag
        outcome = None
        if retained_flag:
            retained = record.get("terminal_value")
            if isinstance(retained, bool) or retained not in (-1, 0, 1):
                raise ValueError("a retained terminal label must be -1, 0 or 1")
            if "game_terminal" in record and not terminal:
                raise ValueError("terminal label contradicts nonterminal game metadata")
            outcome, terminal = float(retained), True
        if terminal_flag:
            winner = record.get("game_winner")
            if isinstance(winner, bool) or winner not in (EMPTY, BLACK, WHITE):
                raise ValueError("terminal game must have an explicit winner or draw")
            played = 0.0 if winner == EMPTY else 1.0 if winner == side else -1.0
            if retained_flag and played != outcome:
                raise ValueError("retained terminal label contradicts winner and actor")
            outcome = played
        record.update(terminal_value=outcome, terminal_value_valid=terminal,
                      value_supervision=value_supervision, value=0.0, value_valid=False,
                      value_source="unknown")
        if value_supervision == "proven_or_terminal" and terminal:
            record.update(value=outcome, value_valid=True, value_source="terminal")
        elif proof is not None:
            record.update(value=float(proof), value_valid=True, value_source="search_proof")
    return records


class GlobalDataset(list):
    """A list of records carrying generation/game statistics for reporting."""
    def __init__(self, records=(), *, games=None, generation=None):
        super().__init__(records)
        self.games = [] if games is None else games
        self.generation = {} if generation is None else generation


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))
    return int(value)


def _real(value, name, minimum=0.0, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be finite and within its allowed range")
    return float(value)


def _shapes(sizes):
    if isinstance(sizes, Integral) and not isinstance(sizes, bool):
        sizes = (sizes,)
    try:
        values = list(sizes)
    except TypeError as exc:
        raise ValueError("sizes must contain square sizes or (height, width) pairs") from exc
    result = []
    for value in values:
        if isinstance(value, Integral) and not isinstance(value, bool):
            size = _integer(value, "board size", 6)
            result.append((size, size))
        else:
            try:
                height, width = value
            except (TypeError, ValueError) as exc:
                raise ValueError("each rectangular size needs two dimensions") from exc
            result.append((_integer(height, "height", 6), _integer(width, "width", 6)))
    if not result:
        raise ValueError("sizes must not be empty")
    return tuple(result)


def canonical_board_key(grid, side):
    """Exact board+actor identity under D4 and simultaneous color/side reversal."""
    side = _integer(side, "side", 1)
    if side not in (BLACK, WHITE):
        raise ValueError("side must be BLACK=1 or WHITE=2")
    board = normalize_board(grid)
    if side == WHITE:
        board = np.array([EMPTY, WHITE, BLACK, FORBIDDEN], dtype=np.uint8)[board]
    alternatives = []
    for turns in range(4):
        rotated = np.rot90(board, turns)
        for transformed in (rotated, np.fliplr(rotated)):
            alternatives.append((tuple(transformed.shape), transformed.tobytes(order="C")))
    shape, cells = min(alternatives)
    return f"{shape[0]}x{shape[1]}:" + cells.hex()


def physical_board_key(grid):
    """A grouping identity only; it does not imply the two actors share a label."""
    return min(canonical_board_key(grid, BLACK), canonical_board_key(grid, WHITE))



def load_extra_positions(paths, *, seen_physical=()):
    """Import externally proved positions, never count them as played games.

    Each JSONL/JSONL.GZ row requires board, explicit side, target_policy,
    search_proven_value (-1/0/1), value_source="search_proof", proof_source and
    group_id. All related nodes of one proof tree must share group_id, including
    across files. That caller-supplied identity defines the held-out unit.
    Optional proof metadata is retained; referenced proof/source files require
    matching SHA256 hashes. Proof logic is NOT independently verified here.
    The importer checks structure and label consistency, not proof soundness.
    Actor is explicit; simultaneous color/actor augmentation remains valid.
    Physical D4/color duplicates retain only their first record and actor label.
    Imported feature maps and caller-supplied canonical keys are never trusted.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    seen = set(seen_physical)
    actor_values, records, files = {}, [], []
    verified_files = {}
    source_counts, skipped = Counter(), 0
    for filename in paths:
        path = Path(filename)
        if not (path.name.lower().endswith('.jsonl') or path.name.lower().endswith('.jsonl.gz')):
            raise ValueError(f"{path}: extra positions require .jsonl or .jsonl.gz")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        file_report = dict(path=str(path.resolve()), sha256=digest, rows=0, accepted=0, duplicates=0)
        opener = gzip.open if path.name.lower().endswith('.gz') else open
        with opener(path, 'rt', encoding='utf-8-sig') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                file_report['rows'] += 1
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("each row must be a JSON object")
                    board = normalize_board(row['board'])
                    if min(board.shape) < 6:
                        raise ValueError("extra full-board positions require both dimensions >= 6")
                    if board_winner(board) or not np.any(board == EMPTY):
                        raise ValueError("policy samples must be nonterminal with legal empty cells")
                    side = _integer(row['side'], 'side', 1)
                    if side not in (BLACK, WHITE):
                        raise ValueError("side must be BLACK=1 or WHITE=2")
                    for name in ('group_id', 'proof_source'):
                        if not isinstance(row.get(name), str) or not row[name].strip():
                            raise ValueError(f"{name} must be a nonempty string")
                    proof = row['search_proven_value']
                    if isinstance(proof, bool) or not isinstance(proof, Real) or proof not in (-1, 0, 1):
                        raise ValueError("search_proven_value must be -1, 0 or 1")
                    if row.get('value_source') != 'search_proof':
                        raise ValueError("extra value_source must be search_proof")
                    if 'value' in row and (isinstance(row['value'], bool) or row['value'] != proof):
                        raise ValueError("value must equal search_proven_value")
                    if 'value_valid' in row and row['value_valid'] is not True:
                        raise ValueError("extra proof value_valid must be true")
                    if row.get('game_terminal', False) is not False or row.get('terminal_value_valid', False) is not False:
                        raise ValueError("extra proof positions must not claim a played terminal outcome")
                    if row.get('terminal_value') is not None or row.get('game_winner') is not None:
                        raise ValueError("extra positions cannot supply played-game outcomes")
                    target = np.asarray(row['target_policy'])
                    if target.dtype.kind not in 'iuf' or target.shape not in (board.shape, (board.size,)):
                        raise ValueError("target_policy must be a numeric HxW or H*W array")
                    target = target.astype(np.float64).reshape(board.shape)
                    if not np.all(np.isfinite(target)) or np.any(target < 0):
                        raise ValueError("target_policy must be finite and nonnegative")
                    if np.any(target[board != EMPTY] != 0):
                        raise ValueError("target_policy assigns probability to occupied or forbidden cells")
                    if not np.isclose(target.sum(), 1.0, rtol=0, atol=1e-6):
                        raise ValueError("target_policy must sum to one")
                    target /= target.sum()
                    action = row.get('action', list(np.unravel_index(int(target.argmax()), board.shape)))
                    if (not isinstance(action, (list, tuple)) or len(action) != 2
                            or any(isinstance(v, bool) or not isinstance(v, Integral) for v in action)):
                        raise ValueError("action must contain two integer coordinates")
                    action = tuple(map(int, action))
                    if not (0 <= action[0] < board.shape[0] and 0 <= action[1] < board.shape[1]) or target[action] <= 0:
                        raise ValueError("action must be in the legal target support")
                    policy_source = row.get('policy_source', 'external_proof_policy')
                    if not isinstance(policy_source, str) or not policy_source.strip():
                        raise ValueError("policy_source must be a nonempty string")
                    uniform_loss = proof == -1 and np.allclose(target[board == EMPTY], 1 / np.count_nonzero(board == EMPTY),
                                                               rtol=0, atol=1e-7)
                    if policy_source == 'all_legal_proved_loss' and not uniform_loss:
                        raise ValueError("all_legal_proved_loss requires a uniform legal policy and proved loss")
                    if uniform_loss:
                        policy_source = 'all_legal_proved_loss'
                    provenance = {}
                    for name in ('source_kind',):
                        if name in row:
                            if not isinstance(row[name], str) or not row[name].strip():
                                raise ValueError(f"{name} must be a nonempty string")
                            provenance[name] = row[name]
                    for name in ('prefix_plies', 'variant_index'):
                        if name in row:
                            provenance[name] = _integer(row[name], name)
                    for name, hash_name in (('proof_file', 'proof_file_sha256'), ('source_file', 'source_sha256')):
                        if name not in row and hash_name not in row:
                            continue
                        if not isinstance(row.get(name), str) or not row[name].strip():
                            raise ValueError(f"{name} must accompany {hash_name}")
                        expected_hash = row.get(hash_name)
                        if (not isinstance(expected_hash, str) or len(expected_hash) != 64
                                or any(char not in '0123456789abcdefABCDEF' for char in expected_hash)):
                            raise ValueError(f"{hash_name} must be a SHA256 hex digest")
                        referenced = Path(row[name])
                        if not referenced.is_absolute():
                            referenced = path.parent / referenced
                        referenced = referenced.resolve()
                        if referenced not in verified_files:
                            verified_files[referenced] = hashlib.sha256(referenced.read_bytes()).hexdigest()
                        if verified_files[referenced] != expected_hash.lower():
                            raise ValueError(f"{name} SHA256 mismatch")
                        provenance.update({name: row[name], hash_name: expected_hash.lower(),
                                           name + '_resolved': str(referenced)})
                    key, physical = canonical_board_key(board, side), physical_board_key(board)
                    if key in actor_values and actor_values[key] != proof:
                        raise ValueError("equivalent board+actor rows have conflicting proof values")
                    actor_values[key] = proof
                    if physical in seen:
                        file_report['duplicates'] += 1
                        skipped += 1
                        continue
                    seen.add(physical)
                    # Namespace group IDs, not individual rows/files: one proof
                    # tree supplied in several files must remain in one split.
                    group_id = 'extra-proof:' + row['group_id'].strip()
                    item = dict(board=board, side=side, target_policy=target.astype(np.float32),
                                action=action, board_key=key, physical_key=physical,
                                game_id=group_id, group_kind='constructed_position',
                                source='extra_verified_full_board', policy_source=policy_source,
                                proof_source=row['proof_source'].strip(), proof=row.get('proof'),
                                extra_import=dict(path=str(path.resolve()), sha256=digest, line=line_number),
                                teacher_engine='external_proof', search_engine=row['proof_source'].strip(),
                                teacher_fingerprint=dict(engine='external_proof', source_sha256=digest,
                                                         proof_source=row['proof_source'].strip()),
                                search_requested_depth=None, search_completed_depth=None,
                                search_proven_value=int(proof), game_terminal=False, game_winner=None,
                                game_termination='constructed_position')
                    item.update(provenance)
                    apply_value_supervision([item], 'proven_or_terminal')
                    records.append(item)
                    source_counts[item['proof_source']] += 1
                    file_report['accepted'] += 1
                except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        files.append(file_report)
    return records, dict(records=len(records), groups=len({r['game_id'] for r in records}),
                         duplicates_skipped=skipped, proof_sources=dict(source_counts), files=files,
                         proof_verification='External proof provenance and label consistency checked; certificates not replayed.')


def _start_game(shape, rng, game_index, max_moves):
    board = np.zeros(shape, dtype=np.uint8)
    height, width = shape
    center = (height // 2, width // 2)
    mask_kind = ("none", "edge", "random", "none")[game_index % 4]
    if mask_kind == "edge":
        edge = rng.randrange(4)
        if edge == 0:
            board[0, :] = FORBIDDEN
        elif edge == 1:
            board[-1, :] = FORBIDDEN
        elif edge == 2:
            board[:, 0] = FORBIDDEN
        else:
            board[:, -1] = FORBIDDEN
    elif mask_kind == "random":
        allowed = [(r, c) for r in range(height) for c in range(width) if (r, c) != center]
        count = rng.randint(1, max(1, int(board.size * 0.08)))
        for point in rng.sample(allowed, count):
            board[point] = FORBIDDEN
    board = apply_board_move(board, center, BLACK)
    opening = [(BLACK, center)]
    side = WHITE
    opening_length = 1 if game_index % 4 == 0 else rng.randint(2, 4)
    opening_length = min(opening_length, max_moves - 1)
    while len(opening) < opening_length:
        near = [move for move in legal_cells(board)
                if max(abs(move[0] - center[0]), abs(move[1] - center[1])) <= 2]
        moves = near or legal_cells(board)
        move = rng.choice(moves)
        board = apply_board_move(board, move, side)
        opening.append((side, move))
        side = BLACK + WHITE - side
    return board, side, opening, mask_kind


def _search_policy(board, side, result, wins, blocks):
    move = result.get("move")
    try:
        move = tuple(move)
    except (TypeError, ValueError) as exc:
        raise ValueError("whole-board search did not return a valid action") from exc
    # Validate before converting: a fractional coordinate must never truncate.
    child = apply_board_move(board, move, side)
    move = tuple(int(value) for value in move)
    policy = np.zeros(board.shape, dtype=np.float32)
    if wins:
        if move not in wins:
            raise ValueError("whole-board search missed an already known immediate win")
        for point in wins:
            policy[point] = 1.0 / len(wins)
        source = "immediate_win_set"
    elif len(blocks) == 1:
        if move != blocks[0]:
            raise ValueError("whole-board search ignored the unique immediate defense")
        policy[move] = 1.0
        source = "forced_block"
    else:
        policy[move] = 1.0
        source = "searched_move"
    return move, policy, source, child


def build_global_dataset(games=80, seed=20260906, sizes=(6, 10, 16), *,
                         max_moves=64, max_positions=None, search_depth=3,
                         search_time_limit=0.02, search_max_nodes=256,
                         candidate_width=8, exploration_rate=0.12, progress_callback=None,
                         teacher_engine="python", forcing_seconds=0.0, forcing_nodes=10000,
                         forcing_depth=24, value_supervision="proven_or_terminal"):
    """Collect real whole-board trajectories with bounded search for both colors.

    max_moves counts all moves including opening setup. max_positions caps stored
    examples; the current game still finishes or reaches its move limit so its
    retained labels can use a real terminal result. Truncation is never a draw.
    Native search is opt-in; its optional forcing budget is additional to the
    regular search budget. proven supervision masks unproven values even after
    a played terminal result; terminal_value always preserves that true outcome.

    board_key includes the actor. physical_key reserves a physical board for its
    first game owner, preventing mirrored/color-reversed boards from crossing
    game-group splits. No local 5x5 models or closed-window labels are used.

    If provided, progress_callback(event) is called once after each game has
    finished or reached its move limit and its labels have been finalized.
    event contains game (a copied summary), games_completed, records (the
    cumulative stored count), and elapsed_seconds. Callback errors propagate.
    """
    if teacher_engine not in TEACHER_ENGINES:
        raise ValueError("teacher_engine must be python or native")
    if value_supervision not in VALUE_SUPERVISION_MODES:
        raise ValueError("value_supervision must be proven_or_terminal or proven")
    games = _integer(games, "games")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    seed = _integer(seed, "seed", None)
    shapes = _shapes(sizes)
    max_moves = _integer(max_moves, "max_moves", 2)
    if max_positions is not None:
        max_positions = _integer(max_positions, "max_positions")
    search_depth = _integer(search_depth, "search_depth", 3)
    search_max_nodes = _integer(search_max_nodes, "search_max_nodes")
    candidate_width = _integer(candidate_width, "candidate_width", 1)
    search_time_limit = _real(search_time_limit, "search_time_limit")
    exploration_rate = _real(exploration_rate, "exploration_rate", maximum=1.0)
    forcing_seconds = _real(forcing_seconds, "forcing_seconds")
    forcing_nodes = _integer(forcing_nodes, "forcing_nodes")
    forcing_depth = _integer(forcing_depth, "forcing_depth")
    if teacher_engine == "python" and forcing_seconds:
        raise ValueError("forcing_seconds requires teacher_engine='native'")
    if teacher_engine == "native":
        if (any(max(shape) > 64 for shape in shapes) or search_depth > 58 or candidate_width > 64
                or search_max_nodes > 2**31-1 or forcing_nodes > 2**31-1 or forcing_depth > 4096):
            raise ValueError("native board shape or search budget exceeds the supported range")
    started = time.monotonic()
    rng = random.Random(seed)
    dataset = GlobalDataset(generation=dict(requested_games=games, seed=seed, duplicate_actor=0,
                                           duplicate_physical_owner=0, considered_positions=0,
                                           teacher_engine=teacher_engine, forcing_seconds=forcing_seconds,
                                           forcing_nodes=forcing_nodes, forcing_depth=forcing_depth,
                                           value_supervision=value_supervision))
    if games == 0 or max_positions == 0:
        dataset.generation["elapsed_seconds"] = time.monotonic() - started
        return dataset
    searcher = select_move
    search_options = dict(max_nodes=search_max_nodes, time_limit=search_time_limit,
                          depth=search_depth, candidate_width=candidate_width)
    if teacher_engine == "native":
        from native_search import select_native_move
        searcher = select_native_move
        search_options.update(forcing_seconds=forcing_seconds, forcing_nodes=forcing_nodes,
                              forcing_depth=forcing_depth)
    fingerprint = teacher_fingerprint(teacher_engine)
    dataset.generation["teacher_fingerprint"] = fingerprint
    seen = set()
    physical_owners = {}
    for game_index in range(games):
        if max_positions is not None and len(dataset) >= max_positions:
            break
        game_id = f"global-{seed}-{game_index:05d}"
        board, side, opening, mask_kind = _start_game(shapes[game_index % len(shapes)], rng, game_index, max_moves)
        pending = []
        ply = len(opening)
        calls = total_nodes = 0
        while ply < max_moves and board_winner(board) == EMPTY and bool(np.any(board == EMPTY)):
            wins, blocks, _ = tactical_candidates(board, side)
            priors = np.zeros(board.shape, dtype=np.float32)
            explored = not wins and not blocks and rng.random() < exploration_rate
            if explored:
                # Jitter influences ordering, but the final action is still the
                # bounded whole-board search result, never an unsearched detour.
                for point in legal_cells(board):
                    priors[point] = rng.random()
            result = searcher(board, side, priors, **search_options)
            calls += 1
            total_nodes += int(result.get("nodes", 0))
            move, target, policy_source, child = _search_policy(board, side, result, wins, blocks)
            proof = result.get("proven_value")
            if proof is not None:
                if isinstance(proof, bool) or proof not in (-1, 0, 1):
                    raise ValueError("search returned an invalid proven value")
                proof = int(proof)
            dataset.generation["considered_positions"] += 1
            if max_positions is None or len(dataset) < max_positions:
                key = canonical_board_key(board, side)
                physical = physical_board_key(board)
                owner = physical_owners.get(physical)
                if owner is not None and owner != game_id:
                    dataset.generation["duplicate_physical_owner"] += 1
                elif key in seen:
                    dataset.generation["duplicate_actor"] += 1
                else:
                    seen.add(key)
                    physical_owners[physical] = game_id
                    record = dict(board=board.copy(), side=side, target_policy=target,
                                  action=move, board_key=key, physical_key=physical, game_id=game_id,
                                  ply=ply, opening=opening.copy(), mask_kind=mask_kind,
                                  source="whole_board_search", policy_source=policy_source,
                                  teacher_engine=teacher_engine,
                                  search_engine=result.get("engine", fingerprint["engine"]),
                                  teacher_fingerprint=fingerprint.copy(),
                                  search_forcing=result.get("forcing"),
                                  explored=bool(explored), search_requested_depth=search_depth,
                                  search_completed_depth=int(result.get("completed_depth", 0)),
                                  search_nodes=int(result.get("nodes", 0)),
                                  search_budget_exhausted=bool(result.get("budget_exhausted", False)),
                                  search_proven_value=proof, search_reason=result.get("reason", ""),
                                  value=0.0, value_valid=False, value_source="unknown")
                    dataset.append(record)
                    pending.append(record)
            board, side = child, BLACK + WHITE - side
            ply += 1
        result = board_winner(board)
        terminal = result != EMPTY or not bool(np.any(board == EMPTY))
        termination = "win" if result else "draw" if terminal else "move_limit"
        for record in pending:
            record.update(game_terminal=terminal, game_winner=int(result) if terminal else None,
                          game_termination=termination, game_length=ply)
        apply_value_supervision(pending, value_supervision)
        game_summary = dict(game_id=game_id, shape=list(board.shape), mask_kind=mask_kind,
                            terminal=terminal, winner=int(result) if terminal else None,
                            termination=termination, moves=ply, opening_moves=len(opening),
                            recorded_positions=len(pending), search_calls=calls, search_nodes=total_nodes,
                            teacher_engine=teacher_engine)
        dataset.games.append(game_summary)
        if progress_callback is not None:
            progress_callback(dict(game=dict(game_summary, shape=game_summary["shape"].copy()),
                                   games_completed=len(dataset.games), records=len(dataset),
                                   elapsed_seconds=time.monotonic() - started))
    dataset.generation["elapsed_seconds"] = time.monotonic() - started
    return dataset


def split_global_records(records, validation_fraction=0.2, seed=0):
    """Split complete game groups; reject inputs that violate physical ownership."""
    fraction = _real(validation_fraction, "validation_fraction")
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    owners = {}
    game_ids = set()
    groups_by_kind = {}
    group_kinds = {}
    for record in records:
        game_id = record["game_id"]
        physical = record.get("physical_key")
        if physical is None:
            physical = physical_board_key(record["board"])
        if physical in owners and owners[physical] != game_id:
            raise ValueError("a canonical physical board belongs to multiple game groups")
        owners[physical] = game_id
        game_ids.add(game_id)
        kind = record_group(record)
        if game_id in group_kinds and group_kinds[game_id] != kind:
            raise ValueError("one group cannot mix played games and constructed cases")
        group_kinds[game_id] = kind
        # Each constructed source gets an independent grouped holdout. Without
        # this, two source-game proof trees could both land in the training set
        # among thousands of unrelated one-position tactical groups.
        stratum = (kind, record.get("source", "unrecorded_legacy")) if kind == "constructed" else (kind,)
        groups_by_kind.setdefault(stratum, set()).add(game_id)
    if len(game_ids) < 2:
        raise ValueError("at least two independent games are required for a held-out split")
    def ordered_groups(group_ids):
        return sorted(group_ids, key=lambda game: hashlib.blake2b(f"{seed}:{game}".encode(), digest_size=16).digest())
    validation_games = set()
    for group_ids in groups_by_kind.values():
        if len(group_ids) < 2:
            continue
        ordered = ordered_groups(group_ids)
        count = max(1, min(len(ordered) - 1, round(len(ordered) * fraction)))
        validation_games.update(ordered[:count])
    if not validation_games:
        # Two singleton categories still need a nonempty held-out partition.
        validation_games.add(ordered_groups(game_ids)[0])
    return ([record for record in records if record["game_id"] not in validation_games],
            [record for record in records if record["game_id"] in validation_games])


def global_dataset_report(records):
    """Summarize actual outcomes; invalid zero placeholders are never draws."""
    games = getattr(records, "games", None)
    if games is None:
        grouped = {}
        for record in records:
            if record_group(record) == "constructed":
                continue
            grouped.setdefault(record["game_id"], dict(game_id=record["game_id"],
                               terminal=record.get("game_terminal", False),
                               winner=record.get("game_winner"), termination=record.get("game_termination", "unknown")))
        games = list(grouped.values())
    valid = [record for record in records if record["value_valid"]]
    constructed = [record for record in records if record_group(record) == "constructed"]
    return dict(records=len(records), unique_board_actor_keys=len({r["board_key"] for r in records}),
                unique_physical_boards=len({r["physical_key"] for r in records}),
                constructed_cases=len({r["game_id"] for r in constructed}),
                constructed_records=len(constructed), real_game_records=len(records)-len(constructed),
                record_groups=dict(Counter(record_group(r) for r in records)),
                sources=dict(Counter(r.get("source", "unrecorded_legacy") for r in records)),
                extra_position_records=sum(r.get("source") == "extra_verified_full_board" for r in records),
                extra_position_groups=len({r["game_id"] for r in records if r.get("source") == "extra_verified_full_board"}),
                games_started=len(games), games_terminal=sum(bool(g["terminal"]) for g in games),
                games_truncated=sum(not bool(g["terminal"]) for g in games),
                sizes=dict(Counter("x".join(map(str, np.asarray(r["board"]).shape)) for r in records)),
                sides=dict(Counter(str(r["side"]) for r in records)),
                policy_sources=dict(Counter(r["policy_source"] for r in records)),
                value_sources=dict(Counter(r["value_source"] for r in records)),
                value_supervision_modes=dict(Counter(r.get("value_supervision", "proven_or_terminal") for r in records)),
                terminal_values=dict(Counter(str(int(r["terminal_value"])) for r in records
                                             if r.get("terminal_value_valid", False))),
                teacher_engines=dict(Counter(r.get("teacher_engine", "unrecorded_legacy") for r in records)),
                search_engines=dict(Counter(r.get("search_engine", "unrecorded_legacy") for r in records)),
                teacher_fingerprints=list({tuple(sorted(r["teacher_fingerprint"].items())): r["teacher_fingerprint"]
                                          for r in records if "teacher_fingerprint" in r}.values()),
                valid_values=dict(Counter(str(int(r["value"])) for r in valid)),
                invalid_values=len(records) - len(valid),
                completed_depths=dict(Counter(str(r["search_completed_depth"]) for r in records)),
                requested_depths=dict(Counter(str(r["search_requested_depth"]) for r in records)),
                generation=dict(getattr(records, "generation", {})), games=games)
