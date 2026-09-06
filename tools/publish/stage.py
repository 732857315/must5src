"""Stage must5 and must5src locally, without Git initialization or any upload."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.browser.prepare_pages import validate_archive

SOURCE_SUFFIXES = {'.py', '.c', '.mjs', '.html', '.css', '.svg', '.webmanifest', '.js'}
TOOL_SUFFIXES = SOURCE_SUFFIXES | {'.sh', '.ps1', '.cmd', '.bat'}
EXCLUDED = {'.git', '.repo', 'exports', 'training_runs', 'web_acceptance', 'node_modules',
            '__pycache__', '.pytest_cache', '.venv', 'venv'}
EXCLUDED_PROBES = {'tests/browser_input_probe.py', 'tests/browser_large_receipt_probe.py'}
REQUIRED = ('requirements-repro.txt', 'REPRODUCE.md', 'reproduction/run.py',
            'reproduction/manifest.json', 'package.json', 'package-lock.json',
            'tools/publish/templates/web-README.md', 'tools/publish/templates/source-README.md',
            'tools/publish/templates/REFERENCE-LICENSE', 'web/browser/DEPLOY.md',
            'web/browser/ONNX-RUNTIME-LICENSE.txt', 'web/browser/ONNX-RUNTIME-ThirdPartyNotices.txt')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_relative(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('Unsafe source path: ' + repr(name))
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError('Unsafe source path: ' + repr(name))
    return path


def source_file(root, name):
    relative = safe_relative(name)
    path = root.joinpath(*relative.parts)
    for part in (path, *path.parents):
        if part == root.parent:
            break
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Source file crosses a link: ' + str(part))
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError('Required source file is missing: ' + name)
    return path


def git_files(root):
    command = ['git', '-c', 'safe.directory=' + root.as_posix(), 'ls-files', '-z']
    result = subprocess.run(command, cwd=root, capture_output=True, check=True)
    return [name for name in result.stdout.decode('utf-8').split('\0') if name]


def source_names(root, tracked):
    names = set()
    for name in tracked:
        path = safe_relative(name)
        if any(part in EXCLUDED for part in path.parts) or name in EXCLUDED_PROBES:
            continue
        fixture = path.suffix == '.json' and (path.parts[0] == 'examples' or
                    (path.parts[0] == 'tests' and 'fixtures' in path.parts))
        if path.suffix in SOURCE_SUFFIXES or fixture:
            if (root / name).is_file():
                names.add(name)
    # New tools/tests belong with the code they validate even before local Git staging.
    for folder, suffixes in (('tools', TOOL_SUFFIXES), ('tests', SOURCE_SUFFIXES)):
        for path in (root / folder).rglob('*'):
            relative = path.relative_to(root)
            name = relative.as_posix()
            if (path.is_file() and path.suffix in suffixes and
                    not any(part in EXCLUDED for part in relative.parts) and name not in EXCLUDED_PROBES):
                names.add(name)
    for path in (root / 'tools/publish/templates').rglob('*'):
        if path.is_file():
            names.add(path.relative_to(root).as_posix())
    for path in root.glob('package*.json'):
        if path.is_file():
            names.add(path.name)
    for path in (root / 'reproduction').rglob('*'):
        relative = path.relative_to(root)
        if not path.is_file() or path.name.endswith('.invocation.json') or any(p in EXCLUDED or p in {'runs', 'outputs', 'prepared', 'prepared_v2'} for p in relative.parts[1:]):
            continue
        if (len(relative.parts) > 2 and relative.parts[1] == 'inputs') or path.suffix in SOURCE_SUFFIXES | {'.md', '.txt'} or path.name in {'manifest.json', 'VERIFIED.json', '.gitignore'}:
            names.add(relative.as_posix())
    names.update(REQUIRED)
    return names


def absolute_json_paths(files):
    findings = []
    def walk(value, filename, at='$'):
        if isinstance(value, str) and (re.match(r'^[A-Za-z]:[\\/]', value) or value.startswith('/')):
            findings.append(dict(file=filename, json_path=at, value=value,
                                 preserved_input_provenance=filename.startswith('reproduction/inputs/')))
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, filename, at + '.' + key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, filename, at + '[' + str(i) + ']')
    for name, data in files.items():
        if name.endswith('.json'):
            walk(json.loads(data.decode('utf-8-sig')), name)
    return findings


@dataclass
class StagePlan:
    repos: dict[str, dict[str, bytes]]
    manifest: dict


def inventory(root=ROOT, *, tracked=None):
    root = Path(root).resolve()
    # Fail before producing any checkout if the reproduction kit is incomplete.
    for name in REQUIRED:
        source_file(root, name)
    reproduction = json.loads(source_file(root, 'reproduction/manifest.json').read_text(encoding='utf-8-sig'))
    if reproduction.get('format') != 'must5_reproduction_v1' or not isinstance(reproduction.get('files'), list) or not reproduction['files']:
        raise ValueError('Invalid reproduction manifest')
    declared = set()
    for item in reproduction['files']:
        if not isinstance(item, dict):
            raise ValueError('Invalid reproduction input record')
        name = item.get('path')
        safe_relative(name)
        if not name.startswith('reproduction/inputs/') or name in declared:
            raise ValueError('Invalid or duplicate reproduction input: ' + str(name))
        declared.add(name)
    names = source_names(root, git_files(root) if tracked is None else tracked)
    names = {name for name in names if not name.startswith('reproduction/inputs/')} | declared
    files = {name: source_file(root, name).read_bytes() for name in sorted(names)}
    for item in reproduction['files']:
        name = item['path']
        data = files[name]
        if type(item.get('bytes')) is not int or len(data) != item['bytes'] or sha(data) != item.get('sha256'):
            raise ValueError('Missing or changed reproduction input: ' + name)
    files['README.md'] = files['tools/publish/templates/source-README.md']
    files['REFERENCE-LICENSE'] = files['tools/publish/templates/REFERENCE-LICENSE']
    files['.gitattributes'] = b'* -text\n'
    ignore = source_file(root, '.gitignore').read_bytes() if (root / '.gitignore').is_file() else b''
    files['.gitignore'] = ignore.rstrip() + b'\n\n# Fixed inputs are versioned; new reproduction results are disposable.\n!/reproduction/\n!/reproduction/inputs/\n!/reproduction/inputs/**\n/reproduction/runs/\n/reproduction/outputs/\n/reproduction/prepared/\n/reproduction/prepared_v2/\n'
    area = root / 'exports/browser'
    verified = validate_archive(area / 'must5-browser.zip', json.loads((area / 'release.json').read_text(encoding='utf-8-sig')))
    assets = json.loads(verified.files['assets.json'].decode('utf-8-sig'))['assets']
    web = {name: verified.files[name] for name in set(assets) | {'assets.json', 'sw.js'}}
    web['.nojekyll'] = b''
    web['.gitattributes'] = b'* -text\n'
    web['README.md'] = files['tools/publish/templates/web-README.md']
    web['REFERENCE-LICENSE'] = files['tools/publish/templates/REFERENCE-LICENSE']
    findings = absolute_json_paths(files)
    unsafe = [row for row in findings if not row['preserved_input_provenance']]
    web_findings = absolute_json_paths(web)
    if unsafe or web_findings:
        raise ValueError('Local absolute JSON paths outside frozen input provenance: ' + json.dumps(unsafe + web_findings, ensure_ascii=False))
    repos = {'must5': web, 'must5src': files}
    records = {repo: dict(files={name: dict(sha256=sha(data), bytes=len(data))
                                      for name, data in sorted(contents.items())},
                         file_count=len(contents), bytes=sum(map(len, contents.values())))
               for repo, contents in repos.items()}
    manifest = dict(format='must5_two_repository_stage_v1', created_utc=datetime.now(timezone.utc).isoformat(),
                    release=verified.metadata, repos=records, absolute_json_path_audit=findings,
                    provenance_note='Original frozen input JSON paths are preserved byte-for-byte and disclosed above; operational paths use the portable reproduction kit.',
                    git_initialized=False, uploaded=False)
    return StagePlan(repos, manifest)


def _new_or_empty(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Destination crosses a link: ' + str(part))
    path = path.resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Output must be new or empty: ' + str(path))
    return path


def stage(plan, output_dir, manifest_path=None):
    output = _new_or_empty(output_dir)
    manifest_path = Path(manifest_path).absolute() if manifest_path else output.with_name(output.name + '.manifest.json')
    if manifest_path.is_symlink() or manifest_path.exists() or manifest_path.resolve().is_relative_to(output):
        raise ValueError('Manifest must be a new file outside the output directory')
    # All contents and hashes are frozen in memory before any destination writes.
    output.mkdir(parents=True, exist_ok=True)
    for repo, contents in plan.repos.items():
        for name, data in sorted(contents.items()):
            relative = safe_relative(name)
            target = output / repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(plan.manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return dict(output_dir=str(output), manifest=str(manifest_path),
                repos={name: dict(file_count=record['file_count'], bytes=record['bytes'])
                       for name, record in plan.manifest['repos'].items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, help='New external manifest path; defaults to OUTPUT.manifest.json')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print inventory without writing anything')
    args = parser.parse_args(argv)
    try:
        plan = inventory()
        result = plan.manifest if args.dry_run else stage(plan, args.output_dir, args.manifest)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, 'Staging refused: ' + str(error) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
