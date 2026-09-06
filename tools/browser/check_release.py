"""Verify the actual browser release, checkpoint identity and matching QA report."""
import hashlib
import json
import sys
from pathlib import Path
import zipfile
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from web_match_runner import Journal, canonical_json, file_hash, summarize
from tools.browser.match_evidence import verify_game_evidence
from tools.browser.search_budget import frozen_search_budget
from tools.browser.match_runtime import process_alive
from tools.browser.model_sources import verified_checkpoint_paths

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


BROWSER_FORMAT = 'gomoku_browser_local_acceptance_v1'
PYTHON_FORMAT = 'gomoku_real_browser_acceptance_v1'


def _required_text(record, key):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Missing or invalid ' + key)
    return value


def _created(record):
    value = _required_text(record, 'created_utc')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('created_utc must include a timezone')
    return parsed


def _inside(base, name):
    if not isinstance(name, str) or not name:
        raise ValueError('Invalid recorded file path')
    path = (base/name).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError('Missing recorded file or path outside its run: ' + name)
    return path


def runner_state(run, manifest):
    """Observe the recorded runner process; never trust cached run_status flags."""
    path = run/'runtime/runner.json'
    result = dict(running=False, pid=None, process_alive=None, reason='no_runner_record')
    if not path.exists():
        return result
    record = read(path)
    if not isinstance(record, dict):
        raise ValueError('Runner record must be an object')
    result.update(pid=record.get('pid'), started_utc=record.get('started_utc'),
                  ended_utc=record.get('ended_utc', record.get('end_utc')))
    if (record.get('run_id') != manifest['run_id']
            or ('candidate_id' in record and record['candidate_id'] != manifest['candidate_id'])):
        result['reason'] = 'runner_identity_mismatch'
        return result
    if not any(name in record for name in ('ended_utc', 'end_utc')):
        result['reason'] = 'missing_end_marker'
        return result
    if any(record.get(name) is not None for name in ('ended_utc', 'end_utc') if name in record):
        result['reason'] = 'runner_ended'
        return result
    pid = record.get('pid')
    if isinstance(pid, bool) or not isinstance(pid, int) or not 0 < pid <= 2**32-1:
        result['reason'] = 'invalid_runner_pid'
        return result
    try:
        alive = bool(process_alive(pid))
    except OSError as exc:
        result.update(reason='process_check_failed', error=str(exc))
        return result
    result.update(running=alive, process_alive=alive,
                  reason='live_matching_runner' if alive else 'runner_process_exited')
    return result


def _journal(run):
    path = run/'events.jsonl'
    if not path.is_file():
        raise ValueError('Acceptance manifest has no journal: ' + str(run))
    return Journal(path).events


def _browser_run(run, manifest, asset_version):
    if type(manifest.get('formal')) is not bool:
        raise ValueError('Browser manifest formal must be boolean')
    for key in ('run_id', 'candidate_id', 'asset_version'):
        _required_text(manifest, key)
    _created(manifest)
    hashes = manifest.get('web_hashes')
    if not isinstance(hashes, dict) or 'assets.json' not in hashes:
        raise ValueError('Browser manifest is missing frozen web hashes')
    web = run/'frozen/web'
    for name, expected in hashes.items():
        if file_hash(_inside(web, name)) != expected:
            raise ValueError('Frozen browser asset changed: ' + name)
    frozen_search_budget(web, hashes, manifest.get('search_budget'))
    if read(web/'assets.json').get('version') != manifest['asset_version']:
        raise ValueError('Manifest asset version differs from the frozen package')
    events = _journal(run)
    frozen = [event for event in events if event['kind'] == 'run_frozen']
    if (len(frozen) != 1 or frozen[0] != events[0]
            or any(frozen[0]['data'].get(key) != manifest[key] for key in ('run_id', 'candidate_id'))):
        raise ValueError('Browser journal is not bound to this frozen run')
    evidence, opponents, completed, game_ids = {}, {}, [], set()
    for event in events:
        kind, data = event['kind'], event['data']
        if kind in ('browser_evidence', 'human_move_played'):
            target = evidence if kind == 'browser_evidence' else opponents
            target.setdefault(_required_text(data, 'game_id'), []).append(data)
        if kind != 'game_completed':
            continue
        game_id = _required_text(data, 'game_id')
        if game_id in game_ids:
            raise ValueError('The same game was completed twice in the journal')
        game_ids.add(game_id)
        path = _inside(run, _required_text(data, 'result_file'))
        expected = _required_text(data, 'result_sha256')
        if file_hash(path) != expected:
            raise ValueError('Completed result SHA256 mismatch: ' + game_id)
        result = read(path)
        for key in ('run_id', 'candidate_id'):
            if result.get(key) != manifest[key]:
                raise ValueError('Completed result belongs to another run/candidate')
        for key in ('game_id', 'ai_side', 'outcome', 'winner', 'plies',
                    'termination', 'canonical_trajectory'):
            if key not in result or key not in data or canonical_json(result[key]) != canonical_json(data[key]):
                raise ValueError('Completed result differs from its journal: ' + key)
        if canonical_json(result.get('evidence')) != canonical_json(evidence.get(game_id, [])):
            raise ValueError('Completed evidence references differ from the journal')
        if canonical_json(result.get('opponent_moves')) != canonical_json(opponents.get(game_id, [])):
            raise ValueError('Completed challenger records differ from the journal')
        checked = verify_game_evidence(run, manifest, result)
        if checked.get('verified') is not True:
            raise ValueError('Completed browser evidence was not verified')
        if canonical_json(checked) != canonical_json(result.get('evidence_audit')):
            raise ValueError('Recomputed evidence audit differs from the saved terminal report')
        completed.append(event)
    runner = runner_state(run, manifest)
    errors = [event for event in events if event['kind'] == 'run_error']
    error = None
    if errors and not runner['running']:
        latest = errors[-1]
        started = runner.get('started_utc')
        if not started or datetime.fromisoformat(latest['utc']) >= datetime.fromisoformat(started):
            error = latest['data'].get('error')
    summary = summarize(manifest, completed, running=runner['running'], error=error)
    summary.update(format=BROWSER_FORMAT, directory=str(run), asset_version=manifest['asset_version'],
                   created_utc=manifest['created_utc'], matches_current_assets=manifest['asset_version'] == asset_version,
                   runner=runner, running=runner['running'], evidence_games_verified=len(completed),
                   journal_events=len(events), journal_last_sha256=events[-1]['sha256'],
                   counts_toward_current_browser_goal=False)
    return summary


def _python_history(run, manifest):
    """Expose historical counters separately; do not call them browser QA."""
    events = _journal(run)
    summary = summarize(manifest, events)
    summary.update(directory=str(run), created_utc=manifest.get('created_utc'),
                   counts_toward_current_browser_goal=False, browser_evidence_verified=False,
                   evidence_scope='Historical Python-engine journal counters; excluded from current browser acceptance.')
    return summary


def collect_acceptance_status(root, asset_version):
    """Audit each run separately and choose the newest matching formal candidate.

    Only direct children of web_acceptance are run directories. Browser profiles,
    frozen source trees and game artifacts are never mistaken for separate runs.
    A corrupt completed result fails the check instead of falling back to an
    older passing candidate or trusting a precomputed run_status.json.
    """
    root = Path(root).resolve()
    browser, python = [], []
    for path in sorted((root/'web_acceptance').glob('*/manifest.json')):
        run = path.parent.resolve()
        if not run.is_relative_to(root/'web_acceptance'):
            raise ValueError('Acceptance run escaped its root')
        manifest = read(path)
        if manifest.get('format') == BROWSER_FORMAT:
            browser.append(_browser_run(run, manifest, asset_version))
        elif manifest.get('format') == PYTHON_FORMAT:
            python.append(_python_history(run, manifest))
    browser.sort(key=lambda record: (_created(record), record['run_id'], record['directory']))
    matching = [record for record in browser if record['formal'] and record['matches_current_assets']]
    current = matching[-1] if matching else None
    if current is not None:
        current['counts_toward_current_browser_goal'] = True
    integration = [record for record in browser if not record['formal']]
    matching_integration = [record for record in integration if record['matches_current_assets']]
    previous = next((record for record in python
                     if Path(record['directory']).name == 'recursive_global_v3_formal'), None)
    return dict(
        browser_runs=browser, current_browser_formal=current,
        current_browser_formal_games=current['completed_attempts'] if current else 0,
        current_browser_formal_unique_games=(sum(current['all_unique_games'].values()) if current else 0),
        current_browser_integration=matching_integration[-1] if matching_integration else None,
        browser_integration_runs=integration,
        historical_browser_formal_runs=[record for record in browser if record['formal'] and record is not current],
        python_historical_runs=python,
        previous_python_candidate=(dict(run_id=previous['run_id'], candidate_id=previous['candidate_id'],
            status=previous['status'], completed_attempts=previous['completed_attempts'],
            unique_outcomes=previous['outcomes'], duplicates=previous['duplicate_complete_games'],
            counts_toward_current_browser_goal=False) if previous else None),
        goal_met=bool(current and current['goal_met']),
    )


def main():
    web=ROOT/'web/browser'
    area=ROOT/'exports/browser'
    assets=read(web/'assets.json')
    release=read(area/'release.json')
    pointer=read(area/'latest_qa.json')
    qa_dir=Path(pointer['directory']).resolve()
    if not qa_dir.is_relative_to(area):raise RuntimeError('QA directory escaped browser outputs')
    qa=read(qa_dir/'report.json')
    if not qa.get('passed') or not qa.get('cleanup_complete') or not pointer.get('passed'):raise RuntimeError('Latest browser QA failed')
    if qa.get('asset_version')!=assets['version'] or release['asset_version']!=assets['version']:
        raise RuntimeError('Browser release and latest QA refer to different asset versions')
    archive=Path(release['archive']).resolve()
    if not archive.is_relative_to(area) or sha(archive)!=release['sha256']:
        raise RuntimeError('Release archive identity changed')
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:raise RuntimeError('Archive integrity check failed')
        for name,meta in assets['assets'].items():
            path=(web/name).resolve()
            if not path.is_relative_to(web) or sha(path)!=meta['sha256'] or path.stat().st_size!=meta['bytes']:
                raise RuntimeError('Asset changed: '+name)
            if bundle.read(name)!=path.read_bytes():raise RuntimeError('Stale ZIP asset: '+name)
        for name in ['assets.json','sw.js','DEPLOY.md']:
            if bundle.read(name)!=(web/name).read_bytes():raise RuntimeError('Stale ZIP metadata: '+name)
    verified_checkpoint_paths(assets, root=ROOT)
    acceptance=collect_acceptance_status(ROOT,assets['version'])
    status=dict(checked_utc=datetime.now(timezone.utc).isoformat(),verified=True,
        asset_version=assets['version'],archive=release,models=assets['models'],
        latest_qa=str(qa_dir/'report.json'),qa_checks=len(qa['checks']),**acceptance)
    (area/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
