"""Merge proved action evidence and retain immutable source holdout ownership."""
from collections import defaultdict
import copy
import hashlib
import numpy as np
from global_data import canonical_board_key, physical_board_key


def has_informative_constraints(record):
    board = np.asarray(record['board'])
    bad = np.asarray(record.get('losing_mask', np.zeros_like(board, dtype=bool)))
    good = np.asarray(record.get('winning_mask', np.zeros_like(board, dtype=bool)))
    return bool(good.any() or 0 < bad.sum() < np.count_nonzero(board == 0))


def merge_constraints(records, constraints):
    """Merge equivalent actor states in the old orientation, keeping its owner.

    Source links survive deduplication and are later split as whole connected
    groups. They cannot silently move a previously held-out source into training.
    """
    from global_policy_constraints import align_constraint
    by_actor = {canonical_board_key(r['board'], r['side']): r for r in records}
    owners = {physical_board_key(r['board']): r['game_id'] for r in records}
    report = dict(appended=0, merged=0, ce_disabled=0)
    for constraint in constraints:
        key = canonical_board_key(constraint['board'], constraint['side'])
        existing = by_actor.get(key)
        source_group = constraint['game_id']
        if existing is None:
            existing = copy.deepcopy(constraint)
            physical = physical_board_key(existing['board'])
            existing['game_id'] = owners.get(physical, source_group)
            existing['physical_key'] = physical
            existing['board_key'] = key
            records.append(existing)
            owners[physical] = existing['game_id']
            by_actor[key] = existing
            report['appended'] += 1
            aligned = constraint
        else:
            aligned = align_constraint(constraint, existing['board'], existing['side'])
            if aligned is None:
                raise ValueError('Actor key equivalence did not align proof actions')
            report['merged'] += 1
        board = np.asarray(existing['board'])
        for name in ('losing_mask', 'winning_mask'):
            previous = np.asarray(existing.get(name, np.zeros_like(board, dtype=bool)))
            incoming = np.asarray(aligned[name])
            if previous.dtype != np.bool_ or incoming.dtype != np.bool_ or incoming.shape != board.shape:
                raise ValueError('Aligned action masks must be boolean and match the board')
            existing[name] = previous | incoming
        if np.any(existing['losing_mask'] & existing['winning_mask']):
            raise ValueError('Merged proofs disagree about an action outcome')
        if any(np.any(existing[name] & (board != 0)) for name in ('losing_mask', 'winning_mask')):
            raise ValueError('Merged proof targets an occupied or forbidden cell')
        groups = set(existing.get('constraint_source_groups', []))
        groups.add(source_group)
        existing['constraint_source_groups'] = sorted(groups)
        existing.setdefault('constraint_provenance', []).append(dict(
            source_group=source_group, source_file=constraint.get('source_file'),
            source_sha256=constraint.get('source_sha256'),
            prefix_plies=constraint.get('prefix_plies'),
            board=np.asarray(constraint['board']).tolist(), side=constraint['side'],
            action_evidence=copy.deepcopy(constraint.get('action_evidence', [])),
            import_alignments=copy.deepcopy(constraint.get('evidence_alignments', [])),
            target_alignments=copy.deepcopy(aligned.get('evidence_transforms', []))))
        if existing.get('policy_mask', True):
            existing['previous_policy_supervision'] = dict(source=existing['policy_source'],
                target_policy=np.asarray(existing['target_policy']).tolist())
            report['ce_disabled'] += 1
        existing['policy_mask'] = False
        existing['target_policy'] = np.zeros(board.shape, dtype=np.float32)
        # A partial action proof does not relabel a parent's previously unknown
        # game value. Explicit existing played outcomes retain their provenance.
    report['sources'] = sorted({g for r in records for g in r.get('constraint_source_groups', [])})
    return report


def preserved_policy_split(records, source_split, *, seed=0):
    """Extend a frozen split by source groups without redrawing old membership."""
    train_old = set(source_split['train_groups'])
    valid_old = set(source_split['validation_groups'])
    if train_old & valid_old:
        raise ValueError('Frozen source split overlaps')
    parent = {}
    def find(value):
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]
    def join(a, b):
        parent[find(b)] = find(a)
    physical_owners = {}
    for record in records:
        owner = record['game_id'];find(owner)
        physical = physical_board_key(record['board'])
        if physical in physical_owners:
            join(owner, physical_owners[physical])
        physical_owners[physical] = owner
        for source in record.get('constraint_source_groups', []):
            join(owner, source)
    actual_groups = {r['game_id'] for r in records}
    if not (train_old | valid_old) <= actual_groups:
        raise ValueError('Frozen source groups disappeared while appending constraints')
    components = defaultdict(set)
    for group in list(parent):
        components[find(group)].add(group)
    assignment = {};unassigned = []
    for root, groups in components.items():
        train, valid = bool(groups & train_old), bool(groups & valid_old)
        if train and valid:
            raise ValueError('Constraint source connects frozen training and validation groups')
        if train or valid:
            assignment[root] = 'train' if train else 'validation'
        else:
            unassigned.append(root)
    unassigned.sort(key=lambda key: hashlib.sha256(
        (str(seed) + ':' + '|'.join(sorted(components[key]))).encode()).digest())
    # A single new source is development-only, never both train and validation.
    count = max(1, round(.2 * len(unassigned))) if len(unassigned) >= 2 else 0
    for index, root in enumerate(unassigned):
        assignment[root] = 'validation' if index < count else 'train'
    train = [r for r in records if assignment[find(r['game_id'])] == 'train']
    valid = [r for r in records if assignment[find(r['game_id'])] == 'validation']
    if not train or not valid:
        raise ValueError('Both train and validation partitions must be nonempty')
    if {physical_board_key(r['board']) for r in train} & {physical_board_key(r['board']) for r in valid}:
        raise ValueError('Physical board leaked across preserved holdout')
    return train, valid


def constraint_metrics(probabilities, records):
    """Known-action mass and hits, averaged by original evidence source group.

    Choosing an unproved action is counted as unknown, never as a safe move.
    """
    groups = defaultdict(list);uninformative = 0
    for probability, record in zip(probabilities, records):
        if 'losing_mask' not in record and 'winning_mask' not in record:
            continue
        board = np.asarray(record['board'])
        bad = np.asarray(record.get('losing_mask', np.zeros_like(board, dtype=bool)))
        good = np.asarray(record.get('winning_mask', np.zeros_like(board, dtype=bool)))
        if not has_informative_constraints(record):
            uninformative += 1;continue
        p = np.asarray(probability).reshape(board.shape)
        selected = np.unravel_index(int(p.argmax()), board.shape)
        has_bad, has_good = bool(bad.any()), bool(good.any())
        bad_mass, good_mass = float(p[bad].sum()), float(p[good].sum())
        row = dict(losing_mass=bad_mass if has_bad else None,
                   winning_mass=good_mass if has_good else None,
                   chose_proved_loss=int(bad[selected]), chose_proved_win=int(good[selected]),
                   chose_unknown=int(not (bad[selected] or good[selected])),
                   quality=((1-bad_mass if has_bad else 0)+(good_mass if has_good else 0))/(has_bad+has_good))
        for source in record.get('constraint_source_groups', [record['game_id']]):
            groups[source].append(row)
    def summarize(rows):
        return {key: float(np.mean([r[key] for r in rows if r[key] is not None]))
                if any(r[key] is not None for r in rows) else None
                for key in ('losing_mass','winning_mass','chose_proved_loss','chose_proved_win','chose_unknown','quality')}
    by_source = {source: dict(records=len(rows), **summarize(rows)) for source, rows in groups.items()}
    return dict(source_groups=len(groups), records=sum(len(rows) for rows in groups.values()),
                uninformative_records=uninformative, by_source=by_source,
                **summarize(list(by_source.values()))) if by_source else dict(
                    source_groups=0,records=0,uninformative_records=uninformative,by_source={},
                    losing_mass=None,winning_mass=None,chose_proved_loss=None,chose_proved_win=None,
                    chose_unknown=None,quality=None)
