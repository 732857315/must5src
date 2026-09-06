"""Small independent package fixtures; no model runtime, browser or network."""
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile

from tools.browser import prepare_pages as pages


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fixture_files(extra_assets=None, missing_asset=None, app_changes=None):
    names = ['index.html', 'style.css', 'app.mjs', 'core.mjs', 'geometry.mjs',
             'star-points.mjs', 'search-budget.mjs', 'engine-worker.mjs',
             'icon.svg', 'search.wasm', 'models/opponent.onnx', 'models/play.onnx']
    names += ['models/' + name + '.onnx' for name in (
        'global', 'global5', 'global6', 'global7', 'global5x6', 'global5x7', 'global5x8',
        'global6x5', 'global6x7', 'global6x8', 'global7x5', 'global7x6', 'global7x8',
        'global8x5', 'global8x6', 'global8x7')]
    names += ['vendor/' + name for name in ('ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs',
              'ort-wasm-simd-threaded.wasm', 'ONNX-RUNTIME-LICENSE', 'ONNX-RUNTIME-ThirdPartyNotices.txt')]
    files = {name: ('fixture:' + name).encode() for name in names}
    app = {'start_url': './', 'scope': './', 'icons': [{'src': 'icon.svg'}]}
    app.update(app_changes or {})
    files['app.webmanifest'] = json.dumps(app).encode()
    files.update(extra_assets or {})
    if missing_asset:
        del files[missing_asset]
    assets = {name: {'sha256': digest(data), 'bytes': len(data)} for name, data in files.items()}
    version = digest(json.dumps(assets, sort_keys=True).encode())[:20]
    files['assets.json'] = json.dumps({'version': version, 'assets': assets,
                                     'models': {'global': {'checkpoint_path': 'missing/private/training.pt'}}}).encode()
    files['sw.js'] = ('const VERSION=' + json.dumps(version) + ';\nconst FILES=' +
                      json.dumps(['./' + name for name in assets] + ['./assets.json']) + ';\n').encode()
    files['DEPLOY.md'] = b'Prebuilt package fixture'
    return files, version


class PreparePagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def archive(self, files=None, version=None, extra_entries=()):
        if files is None:
            files, version = fixture_files()
        path = self.root / 'source.zip'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as bundle:
                for name, data in files.items():
                    bundle.writestr(name, data)
                for name, data in extra_entries:
                    if isinstance(name, str):
                        # ZipInfo normally normalizes Windows backslashes while
                        # constructing fixtures. Preserve raw malicious spelling.
                        info = zipfile.ZipInfo('raw-entry')
                        info.filename = info.orig_filename = name
                        name = info
                    bundle.writestr(name, data)
        release = dict(archive='Z:/unavailable/original/source.zip', sha256=digest(path.read_bytes()),
                       bytes=path.stat().st_size, files=len(files) + len(extra_entries), asset_version=version)
        return path, release

    def assert_invalid(self, files=None, version=None, extra_entries=(), message=None):
        path, release = self.archive(files, version, extra_entries)
        with self.assertRaises((ValueError, zipfile.BadZipFile)) as caught:
            pages.validate_archive(path, release)
        if message:
            self.assertIn(message, str(caught.exception))

    def test_prepare_portable_bundle_and_isolated_cli_extract(self):
        archive, release = self.archive()
        source = self.root / 'source_release.json'
        source.write_text(json.dumps(release))
        bundle = self.root / 'bundle'
        result = pages.prepare_bundle(archive, source, bundle)
        self.assertEqual(set(result), {'format', 'asset_version', 'sha256', 'bytes', 'files'})
        self.assertEqual((bundle / 'site.zip').read_bytes(), archive.read_bytes())
        archive.unlink()
        source.unlink()
        output = self.root / 'site'
        completed = subprocess.run([sys.executable, '-I', str(Path(pages.__file__).resolve()),
                                   '--check', '--bundle', str(bundle), '--output', str(output)],
                                  cwd=self.root, capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), result)
        self.assertEqual((output / '.nojekyll').read_bytes(), b'')
        self.assertEqual((output / 'models/global5x8.onnx').read_bytes(), b'fixture:models/global5x8.onnx')
        self.assertEqual(len([p for p in output.rglob('*') if p.is_file()]), result['files'] + 1)

    def test_release_outer_identity_and_counts(self):
        path, release = self.archive()
        for field, value in [('sha256', '0' * 64), ('bytes', release['bytes'] + 1),
                             ('files', release['files'] - 1), ('bytes', True), ('files', True)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                pages.validate_archive(path, dict(release, **{field: value}))

    def test_crc_integrity_even_with_matching_release_sha(self):
        path, release = self.archive()
        data = path.read_bytes()
        old = b'fixture:models/opponent.onnx'
        self.assertEqual(data.count(old), 1)
        data = data.replace(old, b'FIXTURE:models/opponent.onnx')
        path.write_bytes(data)
        release.update(sha256=digest(data), bytes=len(data))
        with self.assertRaises(zipfile.BadZipFile):
            pages.validate_archive(path, release)

    def test_unsafe_paths_rejected(self):
        for name in ('../escape', '/absolute', 'C:/drive', 'a\\b', 'a//b',
                     './dot', 'a/../b', 'NUL.txt', 'trailing.', 'percent%2fpath'):
            with self.subTest(name=name):
                self.assert_invalid(extra_entries=[(name, b'x')], message='Unsafe')

    def test_links_special_files_and_duplicate_names_rejected(self):
        for mode in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFDIR):
            info = zipfile.ZipInfo('link')
            info.create_system = 3
            info.external_attr = (mode | 0o777) << 16
            with self.subTest(mode=mode):
                self.assert_invalid(extra_entries=[(info, b'target')], message='links')
        self.assert_invalid(extra_entries=[('index.html', b'duplicate')], message='Duplicate')
        self.assert_invalid(extra_entries=[('INDEX.HTML', b'case alias')], message='Duplicate')

    def test_file_directory_conflict_rejected_before_extraction(self):
        files, version = fixture_files(extra_assets={'index.html/child.js': b'child'})
        self.assert_invalid(files, version, message='conflict')

    def test_missing_and_mixed_model_assets_rejected(self):
        files, version = fixture_files()
        del files['models/global5x8.onnx']
        self.assert_invalid(files, version, message='Missing or mixed-version')
        files, version = fixture_files()
        files['models/play.onnx'] = b'different weights'
        self.assert_invalid(files, version, message='Missing or mixed-version')

    def test_model_variant_must_not_disappear_from_consistent_manifest(self):
        files, version = fixture_files(missing_asset='models/global8x5.onnx')
        self.assert_invalid(files, version, message='required runtime/model')

    def test_extra_zip_file_rejected(self):
        self.assert_invalid(extra_entries=[('private.txt', b'not declared')], message='Undeclared')

    def test_asset_size_and_content_version_rejected(self):
        files, version = fixture_files()
        manifest = json.loads(files['assets.json'])
        manifest['assets']['models/play.onnx']['bytes'] += 1
        files['assets.json'] = json.dumps(manifest).encode()
        self.assert_invalid(files, version, message='Missing or mixed-version')
        files, version = fixture_files()
        manifest = json.loads(files['assets.json'])
        manifest['version'] = '0' * 20
        files['assets.json'] = json.dumps(manifest).encode()
        files['sw.js'] = files['sw.js'].replace(version.encode(), b'0' * 20)
        self.assert_invalid(files, '0' * 20, message='Content-derived')

    def test_service_worker_version_and_cache_list_match_assets(self):
        files, version = fixture_files()
        for worker in (files['sw.js'].replace(version.encode(), b'0' * 20),
                       files['sw.js'].replace(b'"./models/play.onnx", ', b''),
                       files['sw.js'].replace(b'"./assets.json"', b'"./assets.json", "./assets.json"')):
            with self.subTest(worker=worker[-80:]):
                self.assert_invalid(dict(files, **{'sw.js': worker}), version, message='Service worker')

    def test_manifest_relative_scope_start_and_icon(self):
        for changes in ({'scope': '/'}, {'start_url': 'https://example.com/'},
                        {'icons': [{'src': '../escape.svg'}]}, {'icons': [{'src': 'absent.svg'}]}):
            with self.subTest(changes=changes):
                files, version = fixture_files(app_changes=changes)
                self.assert_invalid(files, version)

    def test_duplicate_json_keys_and_portable_schema_rejected(self):
        files, version = fixture_files()
        files['assets.json'] = files['assets.json'].replace(b'{', b'{"version":"duplicate",', 1)
        self.assert_invalid(files, version, message='Duplicate JSON key')
        archive, release = self.archive()
        portable = pages.validate_archive(archive, release).metadata
        with self.assertRaises(ValueError):
            pages.validate_archive(archive, dict(portable, archive='C:/private'), portable=True)

    def test_output_must_be_empty_and_rejection_does_not_touch_it(self):
        archive, release = self.archive()
        source = self.root / 'release.json'
        source.write_text(json.dumps(release))
        bundle = self.root / 'bundle'
        pages.prepare_bundle(archive, source, bundle)
        output = self.root / 'site'
        output.mkdir()
        marker = output / 'keep.txt'
        marker.write_text('keep')
        with self.assertRaises(ValueError):
            pages.check_bundle(bundle, output)
        self.assertEqual(list(output.iterdir()), [marker])
        self.assertEqual(marker.read_text(), 'keep')
        with self.assertRaises(ValueError):
            pages.prepare_bundle(archive, source, bundle)

    def test_invalid_bundle_never_creates_output(self):
        archive, release = self.archive()
        source = self.root / 'release.json'
        source.write_text(json.dumps(release))
        bundle = self.root / 'bundle'
        pages.prepare_bundle(archive, source, bundle)
        (bundle / 'site.zip').write_bytes(b'bad')
        output = self.root / 'not-created'
        with self.assertRaises(ValueError):
            pages.check_bundle(bundle, output)
        self.assertFalse(output.exists())

    def test_normalized_output_cannot_bypass_nonempty_check(self):
        output = self.root / 'occupied'
        output.mkdir()
        marker = output / 'keep.txt'
        marker.write_text('keep')
        alias = self.root / 'not-created' / '..' / 'occupied'
        with self.assertRaises(ValueError):
            pages._empty_destination(alias)
        self.assertEqual(marker.read_text(), 'keep')
        self.assertFalse((self.root / 'not-created').exists())

    def test_existing_empty_output_is_supported(self):
        archive, release = self.archive()
        source = self.root / 'release.json'
        source.write_text(json.dumps(release))
        bundle = self.root / 'bundle'
        pages.prepare_bundle(archive, source, bundle)
        output = self.root / 'empty'
        output.mkdir()
        pages.check_bundle(bundle, output)
        self.assertTrue((output / 'index.html').is_file())


if __name__ == '__main__':
    unittest.main()
