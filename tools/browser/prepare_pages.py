"""Validate and stage the already-built static browser package; standard library only.

Default: copy the accepted local ZIP into exports/browser/github-pages/bundle.
--check --output DIR: validate that portable bundle and extract into a new or
empty directory. No model export, training, browser, network, or Git operations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / 'exports/browser/github-pages/bundle'
FORMAT = 'must5_github_pages_bundle_v1'
MAX_ZIP_BYTES = 100 * 1024 * 1024
MAX_CONTENT_BYTES = 256 * 1024 * 1024
METADATA_FILES = {'assets.json', 'sw.js', 'DEPLOY.md'}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _json(data):
    def invalid_constant(value):
        raise ValueError('Non-finite JSON value: ' + value)
    return json.loads(data.decode('utf-8-sig'), object_pairs_hook=_object,
                      parse_constant=invalid_constant)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid ' + name)
    return value


def _name(name):
    # These are package file names, never URLs or platform-specific paths.
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_./-]+', name):
        raise ValueError('Unsafe ZIP/asset path: ' + repr(name))
    parts = name.split('/')
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {
        prefix + str(i) for prefix in ('COM', 'LPT') for i in range(1, 10)}
    if any(part in ('', '.', '..') or part.endswith('.') or
           part.split('.')[0].upper() in reserved for part in parts):
        raise ValueError('Unsafe ZIP/asset path: ' + repr(name))
    return name


def _required_assets():
    names = set('index.html style.css app.mjs core.mjs geometry.mjs star-points.mjs '
                'search-budget.mjs engine-worker.mjs app.webmanifest icon.svg search.wasm'.split())
    names.update({'models/opponent.onnx', 'models/play.onnx'})
    for rows in range(5, 9):
        for cols in range(5, 9):
            stem = ('global' if rows == 8 else f'global{rows}') if rows == cols else f'global{rows}x{cols}'
            names.add('models/' + stem + '.onnx')
    names.update('vendor/' + name for name in (
        'ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm',
        'ONNX-RUNTIME-LICENSE', 'ONNX-RUNTIME-ThirdPartyNotices.txt'))
    return names


@dataclass
class VerifiedArchive:
    metadata: dict
    files: dict[str, bytes]
    archive_bytes: bytes


def validate_archive(archive, release, *, portable=False):
    """Check identity and all local resources without consulting training files."""
    archive = Path(archive)
    if not isinstance(release, dict):
        raise ValueError('Release metadata must be an object')
    if portable and (set(release) != {'format', 'asset_version', 'sha256', 'bytes', 'files'}
                     or release.get('format') != FORMAT):
        raise ValueError('Invalid portable release format')
    if not re.fullmatch(r'[0-9a-f]{64}', str(release.get('sha256', ''))):
        raise ValueError('Invalid release SHA256')
    if not re.fullmatch(r'[0-9a-f]{20}', str(release.get('asset_version', ''))):
        raise ValueError('Invalid asset version')
    expected_bytes = _integer(release.get('bytes'), 'release bytes', 1)
    expected_files = _integer(release.get('files'), 'release files', 1)
    if expected_bytes > MAX_ZIP_BYTES or archive.stat().st_size != expected_bytes:
        raise ValueError('Release ZIP byte count mismatch or oversized archive')
    data = archive.read_bytes()
    if len(data) != expected_bytes or _sha(data) != release['sha256']:
        raise ValueError('Release ZIP SHA256 mismatch')
    contents, folded = {}, set()
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        entries = bundle.infolist()
        if len(entries) != expected_files or len(entries) > 256:
            raise ValueError('Release ZIP file count mismatch')
        total = 0
        for info in entries:
            # orig_filename retains raw archive spelling before platform normalization.
            name = _name(info.orig_filename)
            if info.orig_filename != info.filename or name.casefold() in folded:
                raise ValueError('Duplicate or truncated ZIP name: ' + name)
            folded.add(name.casefold())
            mode = stat.S_IFMT(info.external_attr >> 16)
            if info.is_dir() or mode not in (0, stat.S_IFREG) or info.external_attr & 0x410:
                raise ValueError('ZIP links, directories and special files are not allowed: ' + name)
            if info.flag_bits & 1:
                raise ValueError('Encrypted ZIP entry: ' + name)
            total += info.file_size
            if info.file_size < 0 or total > MAX_CONTENT_BYTES:
                raise ValueError('Oversized ZIP contents')
            # Reading each member validates its ZIP CRC before any extraction.
            contents[name] = bundle.read(info)
    for name in contents:
        parts = name.split('/')
        if any('/'.join(parts[:i]).casefold() in folded for i in range(1, len(parts))):
            raise ValueError('ZIP file/directory path conflict: ' + name)
    if not METADATA_FILES.issubset(contents):
        raise ValueError('Missing ZIP metadata')
    manifest = _json(contents['assets.json'])
    if not isinstance(manifest, dict) or not isinstance(manifest.get('assets'), dict):
        raise ValueError('Invalid assets.json')
    assets = manifest['assets']
    if not _required_assets().issubset(assets):
        raise ValueError('Missing required runtime/model assets')
    for name, record in assets.items():
        _name(name)
        if name in METADATA_FILES or not isinstance(record, dict) or set(record) != {'sha256', 'bytes'}:
            raise ValueError('Invalid asset record: ' + name)
        size = _integer(record['bytes'], 'asset bytes')
        digest = record['sha256']
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Invalid asset SHA256: ' + name)
        if name not in contents or len(contents[name]) != size or _sha(contents[name]) != digest:
            raise ValueError('Missing or mixed-version asset: ' + name)
    if set(contents) != set(assets) | METADATA_FILES:
        raise ValueError('Undeclared files in ZIP')
    # This is the exact content-version algorithm used by tools/browser/build.py.
    version = _sha(json.dumps(assets, sort_keys=True).encode())[:20]
    if manifest.get('version') != version or release['asset_version'] != version:
        raise ValueError('Content-derived asset version mismatch')
    worker = contents['sw.js'].decode('utf-8-sig')
    versions = re.findall(r'^\s*const\s+VERSION\s*=\s*("[^"\n]*")\s*;\s*$', worker, re.MULTILINE)
    lists = re.findall(r'^\s*const\s+FILES\s*=\s*(\[.*?\])\s*;\s*$', worker, re.MULTILINE | re.DOTALL)
    if len(versions) != 1 or len(lists) != 1 or _json(versions[0].encode()) != version:
        raise ValueError('Service worker version declaration mismatch')
    cached = _json(lists[0].encode())
    expected_cache = {'./' + name for name in assets} | {'./assets.json'}
    if (not isinstance(cached, list) or not all(isinstance(x, str) for x in cached)
            or len(cached) != len(set(cached)) or set(cached) != expected_cache):
        raise ValueError('Service worker cache manifest mismatch')
    app = _json(contents['app.webmanifest'])
    if not isinstance(app, dict) or app.get('start_url') != './' or app.get('scope') != './':
        raise ValueError('Web manifest must use relative start_url and scope')
    icons = app.get('icons')
    if not isinstance(icons, list) or not icons:
        raise ValueError('Web manifest icons are missing')
    for icon in icons:
        source = icon.get('src') if isinstance(icon, dict) else None
        if isinstance(source, str) and source.startswith('./'):
            source = source[2:]
        if _name(source) not in assets:
            raise ValueError('Web manifest icon missing from assets')
    metadata = dict(format=FORMAT, asset_version=version, sha256=_sha(data),
                    bytes=len(data), files=len(contents))
    return VerifiedArchive(metadata, contents, data)


def _empty_destination(path):
    path = Path(path).absolute()
    # Reject junctions/symlinks at every existing component before resolving it.
    for part in (path, *path.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Output path crosses a link: ' + str(part))
    path = path.resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Output must be a new or empty directory: ' + str(path))
    return path


def prepare_bundle(archive, release_path, bundle):
    verified = validate_archive(archive, _json(Path(release_path).read_bytes()))
    target = _empty_destination(bundle)
    target.mkdir(parents=True, exist_ok=True)
    (target / 'site.zip').write_bytes(verified.archive_bytes)
    (target / 'release.json').write_text(json.dumps(verified.metadata, indent=2) + '\n', encoding='utf-8')
    return verified.metadata


def check_bundle(bundle, output=None):
    bundle = Path(bundle)
    verified = validate_archive(bundle / 'site.zip', _json((bundle / 'release.json').read_bytes()), portable=True)
    if output is not None:
        target = _empty_destination(output)
        target.mkdir(parents=True, exist_ok=True)
        for name, data in verified.files.items():
            path = target.joinpath(*name.split('/'))
            if not path.resolve().is_relative_to(target):
                raise ValueError('Extraction path escaped output directory')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(data)
        (target / '.nojekyll').write_bytes(b'')
    return verified.metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--archive', type=Path, help='Source ZIP; preparation only')
    parser.add_argument('--release', type=Path, help='Source release.json; preparation only')
    parser.add_argument('--check', action='store_true', help='Validate the portable deployment bundle')
    parser.add_argument('--output', type=Path, help='New or empty extraction directory; --check only')
    args = parser.parse_args(argv)
    if args.check and (args.archive is not None or args.release is not None):
        parser.error('--archive and --release are preparation-only options')
    if args.output is not None and not args.check:
        parser.error('--output requires --check')
    try:
        result = (check_bundle(args.bundle, args.output) if args.check else prepare_bundle(
            args.archive or ROOT / 'exports/browser/must5-browser.zip',
            args.release or ROOT / 'exports/browser/release.json', args.bundle))
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
        parser.exit(1, 'Package validation failed: ' + str(error) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
