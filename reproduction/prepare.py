"""Verify fixed inputs and relocate constraint/cache references; no training."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / 'reproduction' / 'manifest.json'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def project_path(value):
    """Only manifest paths rooted in this checkout; original D: paths are metadata."""
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Invalid relative manifest path')
    parts = value.split('/')
    if PurePosixPath(value).is_absolute() or any(p in ('', '.', '..') for p in parts):
        raise ValueError('Manifest path must not escape the checkout')
    result = (PROJECT_ROOT / value).resolve()
    if not result.is_relative_to(PROJECT_ROOT):
        raise ValueError('Manifest path resolves outside the checkout')
    return result


def verify_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    if manifest.get('format') != 'must5_reproduction_v1':
        raise ValueError('Unsupported reproduction manifest format')
    entries = manifest.get('files')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Manifest must list the fixed inputs')
    seen = set()
    for entry in entries:
        name, expected = entry['path'], entry['sha256']
        if name in seen or not name.startswith('reproduction/inputs/'):
            raise ValueError('Duplicate or unexpected input path: ' + name)
        seen.add(name)
        path = project_path(name)
        if not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Malformed input SHA-256: ' + name)
        if not path.is_file() or path.stat().st_size != entry['bytes'] or digest(path) != expected:
            raise ValueError('Fixed input missing or changed: ' + name)
    sources = manifest.get('training_source_sha256')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Manifest must identify the training source closure')
    for name, expected in sources.items():
        path = project_path(name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Training source differs from the original experiment: ' + name)
    training = manifest['training']
    required = [training['initial_checkpoint'], *training['constraints'],
        training['data_source'] + '/dataset.jsonl.gz', training['data_source'] + '/dataset_report.json',
        training['data_source'] + '/split.json', training['local_models'] + '/opponent.pt',
        training['local_models'] + '/play.pt', *manifest['published_checkpoints'].values()]
    if any(name not in seen for name in required):
        raise ValueError('Training or export input is absent from the verified manifest')
    return manifest


def write_same_or_new(path, data):
    """Preparation is repeatable, but a different existing file is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError('Refusing to replace an existing preparation file: ' + str(path))
        return
    with path.open('xb') as stream:
        stream.write(data)


def copy_same_or_new(source, destination):
    """Copy large immutable inputs once; repeat preparations only verify them."""
    checksum, size = digest(source), source.stat().st_size
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != size or digest(destination) != checksum:
            raise ValueError('Refusing to replace an existing preparation file: ' + str(destination))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as reader, destination.open('xb') as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
        if destination.stat().st_size != size or digest(destination) != checksum:
            raise ValueError('Prepared copy differs from its fixed input: ' + str(destination))
    return dict(original_path=source.relative_to(PROJECT_ROOT).as_posix(),
        path=destination.relative_to(PROJECT_ROOT).as_posix(), bytes=size,
        original_sha256=checksum, sha256=checksum)


def relocate_cache(manifest, output):
    """Change only model-path keys; all cached features, labels and splits stay fixed."""
    training = manifest['training']
    original_dir = project_path(training['data_source'])
    original_report = original_dir / 'dataset_report.json'
    source = json.loads(original_report.read_text(encoding='utf-8'))
    hashes = source.get('local_model_sha256')
    if not isinstance(hashes, dict) or len(hashes) != 2:
        raise ValueError('Cache must identify exactly opponent.pt and play.pt')
    roles = {}
    for name, checksum in hashes.items():
        if not isinstance(name, str):
            raise ValueError('Cache checkpoint path must be a string')
        role = name.replace('\\', '/').rsplit('/', 1)[-1]
        if role not in ('opponent.pt', 'play.pt') or role in roles:
            raise ValueError('Cache checkpoint role is missing, duplicated or unknown')
        if not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum):
            raise ValueError('Cache checkpoint SHA-256 is malformed')
        roles[role] = (name, checksum)
    relocated, models = {}, []
    for role in ('opponent.pt', 'play.pt'):
        name, checksum = roles[role]
        actual = project_path(training['local_models'] + '/' + role)
        if digest(actual) != checksum:
            raise ValueError('Cached features belong to different ' + role + ' weights')
        relocated[str(actual)] = checksum
        models.append(dict(role=role, original_path=name, path=str(actual), sha256=checksum))
    current = deepcopy(source)
    current['local_model_sha256'] = relocated
    restored = deepcopy(current)
    restored['local_model_sha256'] = hashes
    if restored != source:
        raise ValueError('Cache relocation changed more than checkpoint path keys')
    destination_dir = output / 'data_source'
    copies = [copy_same_or_new(original_dir / name, destination_dir / name)
              for name in ('dataset.jsonl.gz', 'split.json')]
    destination_report = destination_dir / 'dataset_report.json'
    write_same_or_new(destination_report,
        (json.dumps(current, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    return dict(data_source=destination_dir.relative_to(PROJECT_ROOT).as_posix(),
        original_report_path=original_report.relative_to(PROJECT_ROOT).as_posix(),
        original_report_sha256=digest(original_report),
        report_path=destination_report.relative_to(PROJECT_ROOT).as_posix(),
        report_sha256=digest(destination_report), local_models=models, immutable_copies=copies,
        only_local_model_sha256_keys_changed=True)


def prepare(output_dir=None):
    manifest = verify_manifest()
    output = Path(output_dir) if output_dir is not None else PROJECT_ROOT / 'reproduction' / 'prepared_v2'
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    inputs = (PROJECT_ROOT / 'reproduction' / 'inputs').resolve()
    if (not output.is_relative_to(PROJECT_ROOT) or output == PROJECT_ROOT or
            output == inputs or output.is_relative_to(inputs) or inputs.is_relative_to(output)):
        raise ValueError('Prepared output must be a separate directory inside this checkout')
    existing_report = output / 'report.json'
    if existing_report.exists() and json.loads(existing_report.read_text(encoding='utf-8')).get('format') != 'must5_reproduction_prepared_v2':
        raise ValueError('Preserve the older prepared directory; choose a new v2 preparation directory')
    cache = relocate_cache(manifest, output)
    by_hash = {}
    for entry in manifest['files']:
        by_hash.setdefault((entry['role'], entry['sha256']), []).append(entry)

    def reference(sha, role, destination):
        entries = by_hash.get((role, sha), [])
        if len(entries) != 1:
            raise ValueError('Constraint reference must identify exactly one fixed ' + role)
        actual = project_path(entries[0]['path'])
        relative = os.path.relpath(actual, destination.parent).replace('\\', '/')
        if (destination.parent / relative).resolve() != actual:
            raise ValueError('Relocated reference does not resolve to the fixed input')
        return relative

    generated, action_count = [], 0
    for name in manifest['training']['constraints']:
        original = project_path(name)
        destination = output / 'constraints' / original.name
        rows, row_count = [], 0
        for line in original.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            source = json.loads(line)
            row = deepcopy(source)
            row['source_file'] = reference(row['source_sha256'], 'source_game', destination)
            for item in row['action_evidence']:
                item['proof_file'] = reference(item['proof_file_sha256'], 'proof', destination)
                action_count += 1
            # Nothing about boards, actors, labels, game ids or proof hashes changes.
            restored = deepcopy(row)
            restored['source_file'] = source['source_file']
            for item, before in zip(restored['action_evidence'], source['action_evidence']):
                item['proof_file'] = before['proof_file']
            if restored != source:
                raise ValueError('Relocation changed a constraint beyond its file references')
            rows.append(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
            row_count += 1
        data = ''.join(rows).encode('utf-8')
        write_same_or_new(destination, data)
        generated.append(dict(path=destination.relative_to(PROJECT_ROOT).as_posix(),
            sha256=digest(destination), original_path=name, original_sha256=digest(original), records=row_count))
    report = dict(format='must5_reproduction_prepared_v2', manifest_sha256=digest(MANIFEST_PATH),
        data_source=cache['data_source'], cache_relocation=cache,
        input_files=len(manifest['files']), input_bytes=sum(entry['bytes'] for entry in manifest['files']),
        training_sources=len(manifest['training_source_sha256']), constraints=generated,
        constraint_records=sum(row['records'] for row in generated), action_evidence=action_count,
        constraint_relocation_verified=True, original_input_bytes_unchanged=True,
        independent_proof_revalidation='Not run by prepare; the unchanged train_global loader verifies every defense before training.',
        training_started=False, expected=manifest['training']['expected'])
    write_same_or_new(output / 'report.json', (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    return manifest, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', help='Separate prepared directory within this checkout (default: reproduction/prepared_v2)')
    args = parser.parse_args(argv)
    try:
        _, report = prepare(args.output_dir)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
