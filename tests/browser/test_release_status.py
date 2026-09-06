"""Small, offline release-status tests; no release build or browser is started."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.browser import check_release as release
from tests.browser.test_match_evidence import Fixture
from web_match_runner import Journal, canonical_json, file_hash


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(record), encoding='utf-8')


def make_run(root, name, *, version='current', formal=True, created='2026-09-06T01:00:00+00:00',
             completed=True, legacy=False):
    run = Path(root)/'web_acceptance'/name
    f = Fixture(run)
    f.manifest.update(format=release.PYTHON_FORMAT if legacy else release.BROWSER_FORMAT,
                      formal=formal, asset_version=version, created_utc=created,
                      run_id='run-'+name, candidate_id='candidate-'+name)
    asset = run/'frozen/web/assets.json'
    write(asset, {'version': version})
    f.manifest['web_hashes']['assets.json'] = file_hash(asset)
    f.result.update(run_id=f.manifest['run_id'], candidate_id=f.manifest['candidate_id'])
    for record in f.result['opponent_moves'] + f.result['evidence']:
        record['game_id'] = f.result['game_id']
    for index, payload in enumerate(f.payloads):
        payload.update(run_id=f.manifest['run_id'], candidate_id=f.manifest['candidate_id'])
        f.write(index)
    write(run/'manifest.json', f.manifest)
    f.result['evidence_audit'] = f.verify()
    write(run/'games/game_000001/result.json', f.result)
    journal = Journal(run/'events.jsonl')
    journal.append('run_frozen', dict(run_id=f.manifest['run_id'], candidate_id=f.manifest['candidate_id']))
    if completed:
        for reference in f.result['evidence']:
            journal.append('browser_evidence', reference)
        for record in f.result['opponent_moves']:
            journal.append('human_move_played', record)
        item = {key: f.result[key] for key in ('game_id', 'ai_side', 'outcome', 'winner',
                                               'plies', 'termination', 'canonical_trajectory')}
        item.update(result_file='games/game_000001/result.json',
                    result_sha256=file_hash(run/'games/game_000001/result.json'))
        journal.append('game_completed', item)
    return f, journal


class ReleaseStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def collect(self):
        return release.collect_acceptance_status(self.root, 'current')

    def test_no_formal_run_has_zero_evidenced_games_and_no_goal(self):
        status = self.collect()
        self.assertIsNone(status['current_browser_formal'])
        self.assertEqual(status['current_browser_formal_games'], 0)
        self.assertFalse(status['goal_met'])

    def test_real_verified_completion_is_counted_without_a_status_json(self):
        f, journal = make_run(self.root, 'formal')
        report = self.collect()
        selected = report['current_browser_formal']
        self.assertEqual(report['current_browser_formal_games'], 1)
        self.assertEqual(report['current_browser_formal_unique_games'], 1)
        self.assertEqual(selected['evidence_games_verified'], 1)
        self.assertEqual(selected['candidate_id'], f.manifest['candidate_id'])
        self.assertEqual(selected['outcomes']['2']['loss'], 1)
        self.assertEqual(selected['journal_last_sha256'], journal.events[-1]['sha256'])
        self.assertFalse(selected['running'])

    def test_newest_matching_formal_selected_without_combining_candidates(self):
        make_run(self.root, 'older', created='2026-09-06T00:00:00+00:00')
        current, _ = make_run(self.root, 'newer', created='2026-09-06T02:00:00+00:00', completed=False)
        make_run(self.root, 'different-version', version='later-version', created='2026-09-06T05:00:00+00:00')
        make_run(self.root, 'integration', formal=False, created='2026-09-06T06:00:00+00:00')
        status = self.collect()
        self.assertEqual(status['current_browser_formal']['run_id'], current.manifest['run_id'])
        self.assertEqual(status['current_browser_formal_games'], 0)
        self.assertEqual(len(status['browser_runs']), 4)
        self.assertEqual(len(status['historical_browser_formal_runs']), 2)
        self.assertEqual(len(status['browser_integration_runs']), 1)
        self.assertEqual(status['current_browser_integration']['completed_attempts'], 1)
        self.assertEqual(sum(run['counts_toward_current_browser_goal'] for run in status['browser_runs']), 1)
        self.assertFalse(status['goal_met'])

    def test_only_other_versions_or_nonformal_never_count_toward_current_goal(self):
        make_run(self.root, 'old', version='old')
        make_run(self.root, 'probe', formal=False)
        status = self.collect()
        self.assertIsNone(status['current_browser_formal'])
        self.assertEqual(status['current_browser_formal_games'], 0)
        self.assertFalse(status['goal_met'])

    def test_cached_status_and_unjournaled_terminal_file_do_not_count(self):
        f, _ = make_run(self.root, 'uncommitted', completed=False)
        write(f.run/'run_status.json', dict(status='running', goal_met=True,
              completed_attempts=2000, ai_first_unique_games=1000, ai_second_unique_games=1000))
        status = self.collect()
        self.assertEqual(status['current_browser_formal_games'], 0)
        self.assertFalse(status['goal_met'])
        self.assertEqual(status['current_browser_formal']['status'], 'paused')

    def test_python_history_is_exposed_separately_from_browser_formal(self):
        make_run(self.root, 'recursive_global_v3_formal', legacy=True)
        make_run(self.root, 'browser-probe', formal=False)
        status = self.collect()
        self.assertEqual(len(status['python_historical_runs']), 1)
        self.assertEqual(status['previous_python_candidate']['completed_attempts'], 1)
        self.assertFalse(status['python_historical_runs'][0]['browser_evidence_verified'])
        self.assertFalse(status['previous_python_candidate']['counts_toward_current_browser_goal'])
        self.assertEqual(status['current_browser_formal_games'], 0)

    def test_result_byte_tampering_is_rejected_not_hidden_by_cached_status(self):
        f, _ = make_run(self.root, 'tampered')
        path = f.run/'games/game_000001/result.json'
        path.write_text(path.read_text(encoding='utf-8')+' ', encoding='utf-8')
        write(f.run/'run_status.json', dict(completed_attempts=0))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.collect()

    def test_signed_result_still_needs_actual_browser_evidence(self):
        f, _ = make_run(self.root, 'tampered-evidence')
        png = f.run/f.result['evidence'][0]['screenshot']
        png.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.collect()

    def test_result_identity_and_journal_fields_must_agree_even_after_rehash(self):
        f, journal = make_run(self.root, 'identity')
        result_path = f.run/'games/game_000001/result.json'
        f.result['candidate_id'] = 'other-candidate'
        write(result_path, f.result)
        records = [(event['kind'], copy.deepcopy(event['data'])) for event in journal.events]
        records[-1][1]['result_sha256'] = file_hash(result_path)
        (f.run/'events.jsonl').write_text('', encoding='utf-8')
        replacement = Journal(f.run/'events.jsonl')
        for kind, data in records:
            replacement.append(kind, data)
        with self.assertRaisesRegex(ValueError, 'another run/candidate'):
            self.collect()

    def test_rehashed_result_cannot_change_the_saved_deterministic_evidence_audit(self):
        f, journal = make_run(self.root, 'audit-difference')
        f.result['evidence_audit']['human_moves'] = 999
        result_path = f.run/'games/game_000001/result.json'
        write(result_path, f.result)
        records = [(event['kind'], copy.deepcopy(event['data'])) for event in journal.events]
        records[-1][1]['result_sha256'] = file_hash(result_path)
        (f.run/'events.jsonl').write_text('', encoding='utf-8')
        replacement = Journal(f.run/'events.jsonl')
        for kind, data in records:
            replacement.append(kind, data)
        with self.assertRaisesRegex(ValueError, 'saved terminal report'):
            self.collect()

    def test_journal_signature_and_duplicate_completion_are_rejected(self):
        f, journal = make_run(self.root, 'duplicate')
        journal.append('game_completed', journal.events[-1]['data'])
        with self.assertRaisesRegex(ValueError, 'completed twice'):
            self.collect()
        path = f.run/'events.jsonl'
        lines = path.read_text(encoding='utf-8').splitlines()
        item = json.loads(lines[-1])
        item['data']['plies'] = 99
        lines[-1] = json.dumps(item)
        path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Journal integrity'):
            self.collect()

    def test_runner_liveness_checks_identity_pid_end_marker_and_process(self):
        f, _ = make_run(self.root, 'runner', completed=False)
        baseline = dict(pid=123, run_id=f.manifest['run_id'],
                        candidate_id=f.manifest['candidate_id'], ended_utc=None)
        path = f.run/'runtime/runner.json'
        for changed, alive, expected, reason in (
                ({}, True, True, 'live_matching_runner'),
                ({}, False, False, 'runner_process_exited'),
                ({'run_id': 'other'}, True, False, 'runner_identity_mismatch'),
                ({'candidate_id': 'other'}, True, False, 'runner_identity_mismatch'),
                ({'pid': True}, True, False, 'invalid_runner_pid'),
                ({'pid': 0}, True, False, 'invalid_runner_pid'),
                ({'ended_utc': '2026-09-06T01:05:00Z'}, True, False, 'runner_ended'),
                ({'end_utc': '2026-09-06T01:05:00Z'}, True, False, 'runner_ended')):
            write(path, dict(baseline, **changed))
            with self.subTest(changed=changed, alive=alive), patch.object(release, 'process_alive', return_value=alive) as check:
                observed = release.runner_state(f.run, f.manifest)
                self.assertEqual(observed['running'], expected)
                self.assertEqual(observed['reason'], reason)
                self.assertEqual(check.called, reason in ('live_matching_runner', 'runner_process_exited'))
        write(path, dict(pid=123, run_id=f.manifest['run_id'], end_utc=None))
        with patch.object(release, 'process_alive', return_value=True):
            self.assertTrue(release.runner_state(f.run, f.manifest)['running'])
        write(path, dict(pid=123, run_id=f.manifest['run_id']))
        with patch.object(release, 'process_alive', return_value=True) as check:
            self.assertFalse(release.runner_state(f.run, f.manifest)['running'])
            check.assert_not_called()

    def test_live_runner_updates_actual_summary_but_dead_process_is_paused(self):
        f, _ = make_run(self.root, 'live', completed=False)
        write(f.run/'runtime/runner.json', dict(pid=123, run_id=f.manifest['run_id'], ended_utc=None))
        write(f.run/'run_status.json', dict(status='paused'))
        with patch.object(release, 'process_alive', return_value=True):
            self.assertEqual(self.collect()['current_browser_formal']['status'], 'running')
        write(f.run/'run_status.json', dict(status='running'))
        with patch.object(release, 'process_alive', return_value=False):
            self.assertEqual(self.collect()['current_browser_formal']['status'], 'paused')

    def test_goal_met_is_taken_only_from_the_selected_matching_formal_run(self):
        make_run(self.root, 'old-goal', created='2026-09-06T00:00:00Z', completed=False)
        make_run(self.root, 'current', created='2026-09-06T01:00:00Z', completed=False)
        make_run(self.root, 'probe-goal', formal=False, created='2026-09-06T02:00:00Z', completed=False)
        original = release._browser_run
        def already_audited(run, manifest, version):
            result = original(run, manifest, version)
            result['goal_met'] = run.name != 'current'
            return result
        with patch.object(release, '_browser_run', side_effect=already_audited):
            self.assertFalse(self.collect()['goal_met'])
        def current_goal(run, manifest, version):
            result = original(run, manifest, version)
            result['goal_met'] = run.name == 'current'
            return result
        with patch.object(release, '_browser_run', side_effect=current_goal):
            self.assertTrue(self.collect()['goal_met'])

    def test_manifest_version_is_bound_to_actual_frozen_assets(self):
        f, _ = make_run(self.root, 'version', completed=False)
        f.manifest['asset_version'] = 'forged-current'
        write(f.run/'manifest.json', f.manifest)
        with self.assertRaisesRegex(ValueError, 'frozen package'):
            self.collect()


if __name__ == '__main__':
    unittest.main()
