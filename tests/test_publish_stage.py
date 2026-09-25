"""Local temporary-file tests for repository boundaries; no Git initialization."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.publish import stage
from tests.browser.test_prepare_pages import fixture_files


class PublishStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'source'
        self.root.mkdir()
        def put(name, data):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.put = put
        for name in stage.REQUIRED:
            put(name, b'fixture document')
        put('package.json', b'{}')
        put('package-lock.json', b'{}')
        put('reproduction/run.py', b'print("portable")\r\n')
        put('train_global.py', b'# frozen code\r\n')
        put('.gitignore', b'/exports/\n/training_runs/\n')
        put('tools/new_untracked.py', b'# new tool')
        put('tests/new_untracked.py', b'# new regression')
        put('tests/browser_input_probe.py', b'# excluded')
        put('tests/browser_large_receipt_probe.py', b'# excluded')
        put('tests/browser/fixtures/small.json', b'{"moves":[]}')
        put('examples/board.json', b'[]')
        put('model.pt', b'old root weight')
        put('GOAL_PROGRESS.md', b'old history')
        put('exports/secret.py', b'# not source')
        put('node_modules/accidental.js', b'not source')
        put('reproduction/prepared/run.py', b'# prepared output')
        put('reproduction/runs/result.json', b'{"path":"C:/private"}')
        put('reproduction/prepared.invocation.json', b'{"path":"C:/private"}')
        put('reproduction/inputs/unlisted.tmp', b'not in manifest')
        inputs = {'reproduction/inputs/checkpoint.pt': b'frozen checkpoint',
                  'reproduction/inputs/original.json': b'{"original_path":"D:/old/training"}'}
        records = []
        for name, data in inputs.items():
            put(name, data)
            records.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
        put('reproduction/manifest.json', json.dumps(dict(format='must5_reproduction_v1', files=records)).encode())
        files, version = fixture_files()
        archive = self.root / 'exports/browser/must5-browser.zip'
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, 'w') as z:
            for name, data in files.items():
                z.writestr(name, data)
        put('exports/browser/release.json', json.dumps(dict(
            asset_version=version, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            bytes=archive.stat().st_size, files=len(files))).encode())
        self.tracked = ['train_global.py', 'model.pt', 'GOAL_PROGRESS.md',
                        'tests/browser_input_probe.py', 'tests/browser_large_receipt_probe.py',
                        'tests/browser/fixtures/small.json', 'examples/board.json',
                        'exports/secret.py', 'node_modules/accidental.js']

    def plan(self):
        return stage.inventory(self.root, tracked=self.tracked)

    def test_repository_boundaries_and_portable_inputs(self):
        plan = self.plan()
        web, source = plan.repos['must5'], plan.repos['must5src']
        self.assertEqual(len(web), 40)
        self.assertEqual(web[".gitattributes"], b"* -text\n")
        self.assertNotIn('DEPLOY.md', web)
        self.assertNotIn('site.zip', web)
        self.assertNotIn('manifest.json', web)
        self.assertIn('.nojekyll', web)
        self.assertIn('tools/new_untracked.py', source)
        self.assertIn('tests/new_untracked.py', source)
        self.assertEqual(source['train_global.py'], b'# frozen code\r\n')
        self.assertEqual(source['.gitattributes'], b'* -text\n')
        self.assertIn(b'!/reproduction/inputs/**', source['.gitignore'])
        for name in ('model.pt', 'GOAL_PROGRESS.md', 'exports/secret.py', 'node_modules/accidental.js',
                     'tests/browser_input_probe.py', 'tests/browser_large_receipt_probe.py',
                     'reproduction/prepared/run.py', 'reproduction/runs/result.json',
                     'reproduction/prepared.invocation.json', 'reproduction/inputs/unlisted.tmp'):
            self.assertNotIn(name, source)
        self.assertTrue(all(x['preserved_input_provenance'] for x in plan.manifest['absolute_json_path_audit']))

    def test_stage_writes_external_manifest_and_matching_bytes_without_git(self):
        plan = self.plan()
        output = Path(self.temp.name) / 'checkout'
        result = stage.stage(plan, output)
        manifest = Path(result['manifest'])
        self.assertFalse(manifest.is_relative_to(output))
        self.assertEqual(set(p.name for p in output.iterdir()), {'must5', 'must5src'})
        for repo, record in json.loads(manifest.read_text())['repos'].items():
            for name, metadata in record['files'].items():
                data = (output / repo / name).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), metadata['sha256'])
                self.assertEqual(len(data), metadata['bytes'])
            self.assertFalse((output / repo / '.git').exists())
        self.assertEqual((self.root / 'train_global.py').read_bytes(), b'# frozen code\r\n')

    def test_missing_required_reproduction_entrypoint_rejects(self):
        (self.root / 'reproduction/run.py').unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.plan()

    def test_changed_manifest_input_rejects(self):
        self.put('reproduction/inputs/checkpoint.pt', b'changed')
        with self.assertRaisesRegex(ValueError, 'changed reproduction'):
            self.plan()

    def test_repeated_staging_uses_canonical_docs_and_does_not_duplicate_ignores(self):
        first = self.plan()
        self.put('.gitignore', first.repos['must5src']['.gitignore'])
        second = self.plan()
        self.assertEqual(first.repos['must5src']['.gitignore'], second.repos['must5src']['.gitignore'])
        self.assertEqual(second.repos['must5src']['README.md'], (self.root / 'README.md').read_bytes())
        self.assertEqual(second.repos['must5']['REFERENCE-LICENSE'], (self.root / 'REFERENCE-LICENSE').read_bytes())

    def test_operational_absolute_json_path_rejects(self):
        self.put('examples/board.json', b'{"path":"C:/not-portable"}')
        with self.assertRaisesRegex(ValueError, 'absolute JSON paths'):
            self.plan()

    def test_nonempty_output_and_internal_or_existing_manifest_reject(self):
        plan = self.plan()
        output = Path(self.temp.name) / 'checkout'
        output.mkdir()
        marker = output / 'keep.txt'
        marker.write_text('keep')
        with self.assertRaises(ValueError):
            stage.stage(plan, output)
        self.assertEqual(marker.read_text(), 'keep')
        fresh = Path(self.temp.name) / 'fresh'
        with self.assertRaises(ValueError):
            stage.stage(plan, fresh, fresh / 'manifest.json')
        self.assertFalse(fresh.exists())
        external = Path(self.temp.name) / 'existing.json'
        external.write_text('keep')
        with self.assertRaises(ValueError):
            stage.stage(plan, fresh, external)
        self.assertFalse(fresh.exists())
        self.assertEqual(external.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
