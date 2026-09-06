"""Checkpoint path/identity regressions; fake bytes and mocked loaders only."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, sentinel

from tools.browser.model_sources import (DEFAULT_CHECKPOINTS, checkpoint_metadata,
                                         resolve_checkpoint, verified_checkpoint_paths)


class ModelSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for role, filename in DEFAULT_CHECKPOINTS.items():
            path = self.root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('old-' + role).encode())
        self.custom = self.root / 'training_runs/custom run/global.pt'
        self.custom.parent.mkdir(parents=True)
        self.custom.write_bytes(b'distinct-new-global')

    def manifest(self, custom=False, legacy=False):
        models = {role: checkpoint_metadata(role, root=self.root) for role in DEFAULT_CHECKPOINTS}
        if custom:
            models['global'] = checkpoint_metadata('global', self.custom, root=self.root)
        if legacy:
            for item in models.values(): item.pop('checkpoint_path')
        return dict(version='fixture', models=models)

    def test_old_manifest_without_paths_uses_legacy_files_with_hash_check(self):
        paths = verified_checkpoint_paths(self.manifest(legacy=True), root=self.root)
        self.assertEqual(paths, {role: (self.root / path).resolve() for role, path in DEFAULT_CHECKPOINTS.items()})
        (self.root / DEFAULT_CHECKPOINTS['global']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch: global'):
            verified_checkpoint_paths(self.manifest_with_original_global_hash(), root=self.root)

    def manifest_with_original_global_hash(self):
        manifest = self.manifest(legacy=True)
        manifest['models']['global']['checkpoint_sha256'] = hashlib.sha256(b'old-global').hexdigest()
        return manifest

    def test_custom_global_metadata_is_actual_relative_path_and_sha(self):
        manifest = self.manifest(custom=True)
        metadata = manifest['models']['global']
        self.assertEqual(metadata['checkpoint_path'], 'training_runs/custom run/global.pt')
        self.assertEqual(metadata['checkpoint_sha256'], hashlib.sha256(b'distinct-new-global').hexdigest())
        self.assertEqual(verified_checkpoint_paths(manifest, root=self.root)['global'], self.custom)

    def test_relative_paths_ignore_process_cwd_and_absolute_paths_remain_supported(self):
        before = Path.cwd()
        elsewhere = self.root / 'other-cwd'; elsewhere.mkdir()
        try:
            os.chdir(elsewhere)
            self.assertEqual(resolve_checkpoint('global', 'training_runs/custom run/global.pt', root=self.root), self.custom)
            self.assertEqual(resolve_checkpoint('global', self.custom, root=self.root), self.custom)
        finally:
            os.chdir(before)
        with tempfile.TemporaryDirectory() as external:
            path = Path(external).resolve() / 'outside.pt'; path.write_bytes(b'external-selected-source')
            metadata = checkpoint_metadata('global', path, root=self.root)
            self.assertEqual(metadata['checkpoint_path'], path.as_posix())
            manifest = self.manifest(); manifest['models']['global'] = metadata
            self.assertEqual(verified_checkpoint_paths(manifest, root=self.root)['global'], path)

    def test_explicit_invalid_or_missing_path_never_falls_back_to_existing_v3(self):
        for path in (None, True, 3.0, '', ' ', [], 'missing.pt', 'training_runs'):
            manifest = self.manifest()
            manifest['models']['global']['checkpoint_path'] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                verified_checkpoint_paths(manifest, root=self.root)

    def test_matching_v3_hash_cannot_verify_a_different_selected_checkpoint(self):
        manifest = self.manifest(custom=True)
        manifest['models']['global']['checkpoint_sha256'] = hashlib.sha256(b'old-global').hexdigest()
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch: global'):
            verified_checkpoint_paths(manifest, root=self.root)

    def test_all_roles_and_sha_fields_are_required_even_with_explicit_paths(self):
        baseline = self.manifest(custom=True)
        for defect in ('missing_role', 'missing_sha', 'boolean_sha', 'malformed_sha', 'wrong_local'):
            manifest = copy.deepcopy(baseline)
            if defect == 'missing_role': del manifest['models']['play']
            elif defect == 'missing_sha': del manifest['models']['global']['checkpoint_sha256']
            elif defect == 'boolean_sha': manifest['models']['global']['checkpoint_sha256'] = True
            elif defect == 'malformed_sha': manifest['models']['global']['checkpoint_sha256'] = 'z' * 64
            else: manifest['models']['opponent']['checkpoint_sha256'] = '0' * 64
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                verified_checkpoint_paths(manifest, root=self.root)

    def test_build_cli_keeps_v3_default_and_accepts_explicit_checkpoint(self):
        from tools.browser import build
        self.assertEqual(build.parse_args([]).global_checkpoint, DEFAULT_CHECKPOINTS['global'])
        self.assertEqual(build.parse_args(['--global-checkpoint', str(self.custom)]).global_checkpoint, str(self.custom))
        with patch.object(build, 'ROOT', self.root), patch.object(build, 'OUT', self.root / 'untouched-web'), \
                patch.object(build, 'load_model') as local, patch.object(build, 'load_global_model') as global_load:
            with self.assertRaisesRegex(ValueError, 'checkpoint missing'):
                build.main(['--global-checkpoint', 'missing.pt'])
            local.assert_not_called(); global_load.assert_not_called()
            self.assertFalse((self.root / 'untouched-web').exists())

    def test_reference_loads_manifest_selected_global_and_checks_every_hash_first(self):
        from tests.browser import reference
        manifest = self.manifest(custom=True)
        with patch.object(reference, 'ROOT', self.root), \
                patch.object(reference, 'load_model', side_effect=[(sentinel.opponent, {}), (sentinel.play, {})]) as local, \
                patch.object(reference, 'load_global_model', return_value=(sentinel.global_model, {})) as global_load:
            self.assertEqual(reference.load_reference_models(manifest), (sentinel.opponent, sentinel.play, sentinel.global_model))
            self.assertEqual([call.args for call in local.call_args_list],
                             [(self.root / DEFAULT_CHECKPOINTS['opponent'], 'opponent'),
                              (self.root / DEFAULT_CHECKPOINTS['play'], 'play')])
            global_load.assert_called_once_with(self.custom)
        manifest['models']['global']['checkpoint_sha256'] = '0' * 64
        with patch.object(reference, 'ROOT', self.root), patch.object(reference, 'load_model') as local, \
                patch.object(reference, 'load_global_model') as global_load:
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch: global'):
                reference.load_reference_models(manifest)
            local.assert_not_called(); global_load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
