"""Mine proved whole-board plans and lost successor positions for global training.

Templates come from actual match prefixes. Every noise variant is re-proved;
D4/color-equivalent boards are deduplicated. A positive move and its lost
successor share a source-game group, so a proof tree cannot cross the split.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from board_rules import apply_board_move, board_winner, normalize_board
from board_threat_search import solve_threat
from global_data import physical_board_key


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_templates(specifications):
    """Read path:plies prefixes of completed true games and replay their rules."""
    templates = []
    for spec in specifications:
        path_text, plies_text = spec.rsplit(':', 1)
        path = Path(path_text).resolve()
        plies = int(plies_text)
        result = json.loads(path.read_text(encoding='utf-8-sig'))
        history = result['history']
        if not 0 <= plies < len(history):
            raise ValueError('template prefix must precede the end of the real game')
        board = np.zeros((16, 16), dtype=np.uint8)
        for index, step in enumerate(history[:plies]):
            if step['side'] != 1 + index % 2:
                raise ValueError('template history is not alternating from black')
            board = apply_board_move(board, (step['row'], step['col']), step['side'])
        if board_winner(board):
            raise ValueError('template must be nonterminal')
        source_hash = file_sha(path)
        templates.append(dict(board=board, side=1 + plies % 2, prefix_plies=plies,
                              source_file=str(path), source_sha256=source_hash,
                              group_id='match-proof-' + source_hash[:20]))
    return templates


def varied_position(template, rng, *, max_pairs=6):
    board = normalize_board(template['board'])
    stones = np.argwhere((board == 1) | (board == 2))
    if not len(stones):
        return None
    low = np.maximum(stones.min(0) - 2, 0)
    high = np.minimum(stones.max(0) + 2, np.array(board.shape) - 1)
    available = [tuple(map(int, p)) for p in np.argwhere(board == 0)
                 if np.any(p < low) or np.any(p > high)]
    rng.shuffle(available)
    pairs = rng.randrange(min(max_pairs, len(available) // 2) + 1)
    for _ in range(pairs):
        board[available.pop()] = 1
        board[available.pop()] = 2
    mask = rng.choice(('none', 'none', 'edge', 'random'))
    if mask == 'random':
        for point in available[:rng.randrange(min(12, len(available)) + 1)]:
            board[point] = 3
    elif mask == 'edge':
        edge = rng.randrange(4)
        for r, c in available:
            if (edge == 0 and r == 0) or (edge == 1 and r == board.shape[0] - 1) or (edge == 2 and c == 0) or (edge == 3 and c == board.shape[1] - 1):
                board[r, c] = 3
    if board_winner(board):
        return None
    return board


def proof_pair(board, side, proof, provenance):
    """Convert a positive full-response certificate into two supervised positions.

    The negative successor is certified by the same complete strategy for the
    attacker: every defender move has a winning continuation. No terminal is
    generated here and no arbitrary continuation result is called minimax.
    """
    board = normalize_board(board)
    if board_winner(board) or proof.get('proven_value') != 1:
        raise ValueError('expected a proved positive nonterminal position')
    move = tuple(proof['move'])
    child = apply_board_move(board, move, side)
    if board_winner(child) or not np.any(child == 0):
        raise ValueError('strategic pair must leave a live successor')
    line = proof.get('principal_variation', [])
    if not line or tuple(line[0]['move']) != move or line[0]['side'] != side:
        raise ValueError('certificate first move does not match root')
    winning_candidate = proof.get('winning_candidate')
    if not isinstance(winning_candidate, dict):
        raise ValueError('a non-immediate strategic win needs a complete defender certificate')
    expected = int(np.count_nonzero(child == 0))
    if winning_candidate.get('certified_replies') != expected or winning_candidate.get('legal_reply_count') != expected:
        raise ValueError('winning move certificate must cover every actual defender reply')
    replies = winning_candidate.get('reply_proofs', [])
    points = [tuple(r['move']) for r in replies]
    if len(points) != expected or len(set(points)) != expected or any(child[p] != 0 for p in points):
        raise ValueError('defender proof points do not cover the actual legal set')
    for reply in replies:
        continuation = reply['continuation']
        if continuation.get('proven_value') != 1:
            raise ValueError('every defender branch must have a positive continuation')
    positive = np.zeros(board.shape, dtype=np.float32)
    positive[move] = 1
    negative = np.zeros(board.shape, dtype=np.float32)
    negative[child == 0] = 1 / expected
    common = {**provenance, 'value_source': 'search_proof',
              'proof_source': 'recursive_whole_board_threat_certificate',
              'source_kind': 'reproved_match_prefix_with_remote_context'}
    return [dict(common, board=board, side=int(side), target_policy=positive,
                 search_proven_value=1, policy_source='strategic_proof_move'),
            dict(common, board=child, side=3-int(side), target_policy=negative,
                 search_proven_value=-1, policy_source='all_legal_proved_loss')]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', action='append', required=True, help='actual result.json:prefix_plies')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--max-attempts', type=int, default=4096)
    parser.add_argument('--seconds', type=float, default=.8)
    parser.add_argument('--nodes', type=int, default=30000)
    parser.add_argument('--quiet-plies', type=int, default=2)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--exclude-dataset', type=Path)
    args = parser.parse_args(argv)
    if args.pairs < 1 or args.max_attempts < args.pairs or args.seconds <= 0 or args.nodes < 1:
        parser.error('positive budgets and max-attempts >= pairs required')
    if args.output_dir.exists():
        parser.error('output directory must be new')
    templates = load_templates(args.template)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'proofs').mkdir()
    seen = set()
    if args.exclude_dataset:
        with gzip.open(args.exclude_dataset, 'rt', encoding='utf-8') as stream:
            for line in stream:
                seen.add(physical_board_key(json.loads(line)['board']))
    rng = random.Random(args.seed)
    accepted = 0
    attempts = 0
    started = time.monotonic()
    configuration = vars(args).copy()
    configuration.update(solver_sha256=file_sha(Path(__file__).with_name('board_threat_search.py')),
                         forcing_sha256=file_sha(Path(__file__).with_name('board_forcing.py')))
    (args.output_dir / 'config.json').write_text(json.dumps(configuration, ensure_ascii=False, default=json_value, indent=2), encoding='utf-8')
    with gzip.open(args.output_dir / 'positions.jsonl.gz', 'wt', encoding='utf-8') as output:
        while accepted < args.pairs and attempts < args.max_attempts:
            template = templates[attempts % len(templates)]
            attempts += 1
            board = varied_position(template, rng)
            if board is None or physical_board_key(board) in seen:
                continue
            proof = solve_threat(board, template['side'], max_nodes=args.nodes, time_limit=args.seconds,
                candidate_width=args.width, max_quiet_plies=args.quiet_plies, max_total_plies=64, forcing_depth=40)
            if proof['proven_value'] != 1 or proof.get('winning_candidate') is None:
                continue
            provenance = {k: v for k, v in template.items() if k != 'board'}
            provenance['variant_index'] = accepted
            proof_name = f'proofs/{accepted:06d}.json.gz'
            provenance['proof_file'] = proof_name
            try:
                rows = proof_pair(board, template['side'], proof, provenance)
            except ValueError:
                continue
            keys = [physical_board_key(row['board']) for row in rows]
            if keys[0] == keys[1] or any(k in seen for k in keys):
                continue
            data = gzip.compress(json.dumps(dict(board=board,side=template['side'],result=proof),ensure_ascii=False,default=json_value).encode('utf-8'),mtime=0)
            (args.output_dir / proof_name).write_bytes(data)
            proof_hash = hashlib.sha256(data).hexdigest()
            for row, key in zip(rows, keys):
                row['proof_file_sha256'] = proof_hash
                output.write(json.dumps(row,ensure_ascii=False,default=json_value) + '\n')
                seen.add(key)
            output.flush()
            accepted += 1
            if accepted % 16 == 0 or accepted == args.pairs:
                status = dict(pairs=accepted,positions=2*accepted,attempts=attempts,elapsed_seconds=time.monotonic()-started)
                print(json.dumps(status),flush=True)
                (args.output_dir / 'progress.json').write_text(json.dumps(status),encoding='utf-8')
    summary = dict(pairs=accepted, positions=2*accepted, attempts=attempts, complete=accepted==args.pairs,
                   source_games=len({t['group_id'] for t in templates}), elapsed_seconds=time.monotonic()-started,
                   model_training_started=False, interpretation='Constructed re-proved plans; not played games.')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
    return 0 if summary['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
