"""Frozen, resumable formal games against the AI running entirely in a browser."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from web_match_runner import (Journal,JsonProcess,atomic_json,canonical_json,digest,exclusive,
    file_hash,locate_browser,next_ai_side,summarize as original_summary,strip_paths,utc)
from native_search import native_library
from tools.browser.match_runtime import BrowserRuntime
from tools.browser.match_verify import verify_browser_state,verify_extension,verify_terminal_game
from tools.browser.match_evidence import verify_game_evidence
from tools.browser.search_budget import frozen_search_budget

FORMAT='gomoku_browser_local_acceptance_v1'
OPTIONS=dict(time_limit=1.5,max_nodes=150000,depth=9,candidate_width=24,
             forcing_seconds=.15,forcing_nodes=20000,forcing_depth=32)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def summarize(*args,**kwargs):
    result=original_summary(*args,**kwargs)
    result['format']=FORMAT
    return result


def verify_frozen(run,manifest):
    if manifest.get('format')!=FORMAT:raise ValueError('Wrong browser acceptance format')
    for group,folder in [('source_hashes','source'),('web_hashes','web')]:
        base=run/'frozen'/folder
        for name,expected in manifest[group].items():
            path=(base/name).resolve()
            if not path.is_relative_to(base.resolve()) or file_hash(path)!=expected:
                raise ValueError('Frozen file changed: '+name)
    frozen_search_budget(run/'frozen/web',manifest['web_hashes'],manifest.get('search_budget'))
    if file_hash(manifest['browser'])!=manifest['browser_sha256']:
        raise ValueError('Browser executable changed; retain previous run')
    identity={k:manifest[k] for k in ('web_hashes','source_hashes','seconds','seed','opponent','browser_sha256','goal')}
    if digest(identity)!=manifest['candidate_id']:raise ValueError('Candidate identity mismatch')


def _registry_record(run,manifest):
    return dict(run_id=manifest['run_id'],candidate_id=manifest['candidate_id'],
                run_dir=str(Path(run).resolve()),manifest_sha256=digest(manifest),
                formal=manifest['formal'])


def _registered_manifest(entry):
    run=Path(entry['run_dir'])
    if not run.is_absolute():raise RuntimeError('Candidate registry contains a nonabsolute run path')
    try:
        manifest=read(run/'manifest.json')
    except (OSError,ValueError) as exc:
        raise RuntimeError('Prior candidate run is unavailable; retain and resume '+str(run)) from exc
    if (_registry_record(run,manifest)!=entry):
        raise RuntimeError('Registered candidate manifest changed: '+str(run))
    return run,manifest


def _completed_registered_run(entry):
    """Read and audit an old run while its nonblocking run lock is held."""
    run,manifest=_registered_manifest(entry)
    try:
        with exclusive(run/'run.lock'):
            journal=Journal(run/'events.jsonl')
            completed=[e['data'] for e in journal.events if e['kind']=='game_completed']
            status=summarize(manifest,journal.events)
            if status['black_losses'] or status['white_fixed_sample_failed']:
                raise RuntimeError('This candidate previously failed; results cannot be redrawn: '+str(run))
            allocated=[e['data']['game_id'] for e in journal.events if e['kind']=='game_allocated']
            finished=[e['game_id'] for e in completed]
            if (not allocated or len(set(allocated))!=len(allocated)
                    or len(set(finished))!=len(finished) or set(allocated)!=set(finished)):
                raise RuntimeError('Prior candidate has an unfinished or unaudited game; resume '+str(run))
            current=read(run/'current.json')
            if (current.get('phase')!='completed' or current.get('pending') is not None
                    or current.get('game_id')!=allocated[-1]):
                raise RuntimeError('Prior candidate game is not durably completed; resume '+str(run))
            for game in completed:
                result_path=(run/game['result_file']).resolve()
                if not result_path.is_relative_to(run.resolve()) or file_hash(result_path)!=game['result_sha256']:
                    raise RuntimeError('Prior terminal result is unavailable or changed: '+str(run))
                result=read(result_path)
                if any(result.get(k)!=game.get(k) for k in ('game_id','ai_side','outcome','winner','plies','termination','canonical_trajectory')):
                    raise RuntimeError('Prior terminal result disagrees with its journal: '+str(run))
                audit=verify_game_evidence(run,manifest,result)
                if audit!=result.get('evidence_audit') or audit.get('verified') is not True:
                    raise RuntimeError('Prior terminal result lacks its complete evidence audit: '+str(run))
                if game['game_id']==current['game_id']:
                    last=current.get('last')
                    if not isinstance(last,dict) or any(last.get(k)!=result.get(k) for k in ('board','history','winner','terminal','plies','ai_side')):
                        raise RuntimeError('Prior current state disagrees with its terminal result: '+str(run))
            if manifest['formal']:
                explanation='already has its fixed formal sample' if status['goal_met'] else 'has unfinished formal acceptance'
                raise RuntimeError('Candidate '+explanation+'; use its existing run '+str(run))
            return status
    except RuntimeError:
        raise
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise RuntimeError('Prior candidate is running, unfinished, or unaudited; preserve and resume '+str(run)) from exc


def _candidate_registration(run,manifest,*,create):
    """Serialize candidate ownership; different directories cannot redraw a run."""
    run=Path(run).resolve()
    registry_path=Path(manifest['registry'])
    if not registry_path.is_absolute():raise RuntimeError('Candidate registry path must be absolute')
    desired=_registry_record(run,manifest)
    with exclusive(registry_path.with_suffix('.lock')):
        registry=Journal(registry_path)
        records=[e['data'] for e in registry.events if e['kind']=='candidate_run_registered']
        same_run=[entry for entry in records if entry['run_id']==manifest['run_id']
                  or Path(entry['run_dir']).resolve()==run]
        if same_run and (len(same_run)!=1 or same_run[0]!=desired):
            raise RuntimeError('Run identity or its frozen manifest changed in the candidate registry')
        if not same_run and not create:
            raise RuntimeError('This run was not registered before its first browser action')
        for entry in records:
            if entry['candidate_id']!=manifest['candidate_id'] or entry in same_run:continue
            _completed_registered_run(entry)
            if not manifest['formal']:
                raise RuntimeError('A completed development run already exists; resume it or begin formal acceptance: '+entry['run_dir'])
        if not same_run:
            registry.append('candidate_run_registered',desired)
        return desired


def register_candidate_run(run,manifest):
    return _candidate_registration(run,manifest,create=True)


def assert_registered_candidate(run,manifest):
    return _candidate_registration(run,manifest,create=False)


def freeze(args):
    run=Path(args.run_dir).resolve()
    if run.exists():raise ValueError('Run directory already exists; resume it instead of overwriting')
    if args.seconds!=1.0:raise ValueError('This formal configuration uses the requested default 1 second')
    web=ROOT/'web/browser';assets=read(web/'assets.json')
    for name,record in assets['assets'].items():
        if file_hash(web/name)!=record['sha256']:raise ValueError('Rebuild changed browser assets before freezing')
    library=native_library()
    sources=run/'frozen/source';target_web=run/'frozen/web';sources.mkdir(parents=True);target_web.mkdir()
    files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.mjs','.c','.h')]
    files+=list((ROOT/'tools/browser').glob('*.py'))
    files.append(ROOT/'native/browser_search.c')
    native_dir=Path(library.binary_path).parent
    files += [p for p in native_dir.iterdir() if p.suffix in ('.dll','.json')]
    hashes={}
    for original in files:
        name=original.relative_to(ROOT).as_posix();target=sources/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,target);hashes[name]=file_hash(target)
    web_hashes={}
    for name in list(assets['assets'])+['assets.json','sw.js','DEPLOY.md']:
        target=target_web/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(web/name,target)
        web_hashes[name]=file_hash(target)
    args_worker=[sys.executable,'-X','utf8','-B',str(sources/'web_match_runner.py'),
        '--worker-callback','match_challenger:choose_move','--worker-options-json',canonical_json(OPTIONS),'--fingerprint-only']
    result=subprocess.run(args_worker,cwd=sources,capture_output=True,text=True,encoding='utf-8',timeout=60,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if result.returncode:raise RuntimeError(result.stderr)
    opponent=json.loads(result.stdout)['fingerprint']
    browser=locate_browser();node=shutil.which('node')
    goal=read(ROOT/'acceptance_goal.json')
    manifest=dict(format=FORMAT,run_id=uuid.uuid4().hex,created_utc=utc(),formal=args.formal,
        seconds=args.seconds,seed=args.seed,opponent=strip_paths(opponent),opponent_options=OPTIONS,
        web_hashes=web_hashes,source_hashes=hashes,asset_version=assets['version'],
        models=assets['models'],browser=browser,browser_sha256=file_hash(browser),node=node,
        python=sys.version,headed=args.headed,goal=goal,
        registry=str((ROOT/'web_acceptance/browser_candidate_registry.jsonl').resolve()),
        interpretation='AI inference and all candidate search run in the frozen browser Worker. The external challenger receives only the actual board. Trusted UI inputs only; no injected moves or outcome rewrites. First 1000 unique complete games per AI side; every first-player loss remains a failure.',
        proof_evidence_limit='Quiet certificates expose a representative PV and reply count, not every nested reply proof. Actual game results are independently replayed.')
    budget=frozen_search_budget(target_web,web_hashes,require_declaration=False)
    if budget is not None:manifest['search_budget']=budget
    manifest['candidate_id']=digest({k:manifest[k] for k in ('web_hashes','source_hashes','seconds','seed','opponent','browser_sha256','goal')})
    atomic_json(run/'manifest.json',manifest)
    journal=Journal(run/'events.jsonl');journal.append('run_frozen',{'run_id':manifest['run_id'],'candidate_id':manifest['candidate_id']})
    atomic_json(run/'run_status.json',summarize(manifest,journal.events))
    register_candidate_run(run,manifest)
    return manifest


def capture(runtime,run,manifest,journal,current,stage):
    raw=runtime.idle()
    state=verify_browser_state(raw,current['ai_side'],manifest['seconds'],require_human_turn=True)
    saved=runtime.saved()
    if not saved or saved['board']!=raw['board'] or saved['history']!=raw['history']:
        raise RuntimeError('Visible game differs from the durable browser storage')
    audit=runtime.audit()
    folder=run/'games'/current['game_id'];folder.mkdir(parents=True,exist_ok=True)
    token=uuid.uuid4().hex
    png=folder/f'evidence_{token}.png'
    screenshot_sha=runtime.screenshot(png)
    payload=dict(run_id=manifest['run_id'],candidate_id=manifest['candidate_id'],game_id=current['game_id'],
                 stage=stage,raw=raw,state=state,saved=saved,audit=audit,screenshot=str(png.relative_to(run).as_posix()))
    path=folder/f'evidence_{token}.json.gz'
    data=canonical_json(payload).encode('utf-8')
    temporary=path.with_suffix(path.suffix+'.pending')
    with temporary.open('xb') as stream:
        stream.write(gzip.compress(data,mtime=0));stream.flush();os.fsync(stream.fileno())
    # Only complete files become recoverable evidence. Keep interrupted .pending
    # files as artifacts; the original page events are not acknowledged yet.
    os.replace(temporary,path)
    reference=dict(game_id=current['game_id'],file=path.relative_to(run).as_posix(),sha256=file_hash(path),
                   screenshot=png.relative_to(run).as_posix(),screenshot_sha256=screenshot_sha)
    journal.append('browser_evidence',reference)
    if audit['events']:runtime.ack(max(event['seq'] for event in audit['events']))
    return state


def recorded_evidence(journal,game_id):
    return [e['data'] for e in journal.events if e['kind']=='browser_evidence' and e['data']['game_id']==game_id]


def restore_evidence(run,manifest,journal,current):
    # A complete fsynced file can precede its journal append if the runner stops
    # between the two. Recover it without replacing any already recorded item.
    seen={e['file'] for e in recorded_evidence(journal,current['game_id'])}
    for path in sorted((run/'games'/current['game_id']).glob('evidence_*.json.gz')):
        name=path.relative_to(run).as_posix()
        if name in seen:continue
        with gzip.open(path,'rt',encoding='utf-8') as stream:payload=json.load(stream)
        if any(payload[k]!=v for k,v in [('run_id',manifest['run_id']),('candidate_id',manifest['candidate_id']),('game_id',current['game_id'])]):
            raise ValueError('Unjournaled evidence belongs to another game')
        png=(run/payload['screenshot']).resolve()
        if not png.is_relative_to(run):raise ValueError('Evidence screenshot escaped run directory')
        verify_browser_state(payload['raw'],current['ai_side'],manifest['seconds'])
        journal.append('browser_evidence',dict(game_id=current['game_id'],file=name,sha256=file_hash(path),
            screenshot=payload['screenshot'],screenshot_sha256=file_hash(png),recovered_after_interruption=True))


def committed_idle(runtime,journal,current):
    raw=runtime.idle()
    if raw.get('storageBlocked') or raw.get('pendingCommit'):
        journal.append('storage_retry_intent',{'game_id':current['game_id'],'pending_commit':raw.get('pendingCommit')})
        runtime.click('#retry-save')
        raw=runtime.idle()
    return raw


def prepare(runtime,run,manifest,journal,current):
    raw=committed_idle(runtime,journal,current)
    if current['phase']=='opening':
        verify_browser_state(raw,current['ai_side'],manifest['seconds'])
        if not raw['started']:
            if raw['history']:raise RuntimeError('Unstarted opening has played moves')
            runtime.click('#start')
        state=capture(runtime,run,manifest,journal,current,'resumed_opening')
        verify_extension(current['opening'],state)
        if state['plies']!=(1 if current['ai_side']==1 else 0):raise RuntimeError('Resumed opening has unexplained actions')
        current.update(phase='playing',last=state,pending=None)
        atomic_json(run/'current.json',current)
        return state
    # Never reset a live game. A previous game may only be cleared after its
    # independently verified terminal result was durably journaled.
    if raw['history']:
        previous=verify_browser_state(raw,3-raw['human'],manifest['seconds'])
        completed=[e['data'] for e in journal.events if e['kind']=='game_completed']
        if not previous['terminal'] or not completed:
            raise RuntimeError('A live or unrecorded old game cannot be reset for a new attempt')
        result=read(run/completed[-1]['result_file'])
        if result['history']!=previous['history']:raise RuntimeError('Browser is not on the last completed game')
        runtime.click('#new');raw=runtime.idle()
    if raw['started']:
        raise RuntimeError('A started game has no durable opening intent; preserve it for inspection')
    else:
        if raw['n']!=16 or raw['seconds']!=manifest['seconds'] or any(raw['board']):
            raise RuntimeError('Formal opening must be empty 16x16 with the frozen 1-second setting')
        if raw['human']!=3-current['ai_side']:runtime.configure_human(3-current['ai_side'])
        initial=verify_browser_state(runtime.snapshot(),current['ai_side'],manifest['seconds'])
        current.update(phase='opening',opening=initial,last=None,pending=None)
        atomic_json(run/'current.json',current)
        journal.append('game_started',{'game_id':current['game_id'],'ai_side':current['ai_side'],'opening':initial})
        runtime.click('#start')
    state=capture(runtime,run,manifest,journal,current,'opening')
    verify_extension(current['opening'],state)
    if state['plies']!=(1 if current['ai_side']==1 else 0):raise RuntimeError('Opening has unexplained actions')
    current.update(phase='playing',last=state,pending=None)
    atomic_json(run/'current.json',current)
    return state


def record_played(journal,current,pending,state):
    item=json.loads(canonical_json({'game_id':current['game_id'],**pending,'after_plies':state['plies']}))
    previous=[e['data'] for e in journal.events if e['kind']=='human_move_played'
              and e['data']['game_id']==current['game_id']
              and e['data']['before']['plies']==pending['before']['plies']]
    if previous:
        if len(previous)!=1 or previous[0]!=item:raise RuntimeError('Conflicting played-move record')
    else:journal.append('human_move_played',item)


def main_run(run,manifest,max_games):
    journal=Journal(run/'events.jsonl');runtime=None;challenger=None
    error=None;completed_now=0
    try:
        status=summarize(manifest,journal.events)
        if status['status']=='candidate_failed' or status['goal_met']:
            print(canonical_json(status));return 0
        assert_registered_candidate(run,manifest)
        atomic_json(run/'runtime/runner.json',dict(pid=os.getpid(),run_id=manifest['run_id'],candidate_id=manifest['candidate_id'],started_utc=utc(),ended_utc=None))
        runtime=BrowserRuntime(run,manifest,journal)
        challenger=JsonProcess([sys.executable,'-X','utf8','-B',str(run/'frozen/source/web_match_runner.py'),
            '--worker-callback','match_challenger:choose_move','--worker-options-json',canonical_json(manifest['opponent_options'])],
            run/'frozen/source',run/'runtime/challenger.log')
        if strip_paths(challenger.ready['fingerprint'])!=manifest['opponent']:
            raise RuntimeError('Challenger differs from the frozen identity')
        while completed_now<max_games:
            status=summarize(manifest,journal.events)
            if status['status']=='candidate_failed' or status['goal_met']:break
            state_file=run/'current.json'
            current=read(state_file) if state_file.exists() else None
            finished={e['data']['game_id'] for e in journal.events if e['kind']=='game_completed'}
            if current and current['game_id'] in finished:
                current=None
            if current is None:
                opened=[e for e in journal.events if e['kind']=='game_allocated']
                index=len(opened)+1
                side=next_ai_side(manifest,journal.events,index-1)
                if side is None:break
                current={'game_id':f'game_{index:06d}','game_index':index,'ai_side':side,'phase':'prepare','last':None,'pending':None}
                atomic_json(state_file,current)
                journal.append('game_allocated',dict(current))
            allocated={e['data']['game_id'] for e in journal.events if e['kind']=='game_allocated'}
            if current['game_id'] not in allocated:
                journal.append('game_allocated',dict(current,recovered_after_interruption=True))
            restore_evidence(run,manifest,journal,current)
            if current['phase'] in ('prepare','opening'):
                state=prepare(runtime,run,manifest,journal,current)
            else:
                raw=committed_idle(runtime,journal,current)
                state=verify_browser_state(raw,current['ai_side'],manifest['seconds'],require_human_turn=True)
                if current['pending']:
                    before=current['pending']['before']
                    if state['history']!=before['history']:verify_extension(before,state,current['pending']['choice']['move'])
                else:
                    verify_extension(current['last'],state)
                    if state['history']!=current['last']['history']:
                        raise RuntimeError('An unplanned human move appeared without a durable search intent')
            while not state['terminal']:
                if current['pending'] is None:
                    choice=challenger.request({'board':state['board'],'side':3-current['ai_side'],
                        'context':{'seed':manifest['seed'],'game_index':current['game_index'],
                        'ai_side':current['ai_side'],'ply':state['plies'],'opponent_options':manifest['opponent_options']}},timeout=60)
                    move=choice['move']
                    if (not isinstance(move,list) or len(move)!=2 or any(type(v) is not int or not 0<=v<16 for v in move)
                        or state['board'][move[0]][move[1]]):raise RuntimeError('Challenger selected an illegal move')
                    current['pending']={'before':state,'choice':choice}
                    atomic_json(state_file,current)
                    journal.append('human_move_intent',{'game_id':current['game_id'],**current['pending']})
                pending=current['pending'];before=pending['before'];move=pending['choice']['move']
                if state['history']==before['history']:
                    runtime.click(f'[data-point="{move[0]*16+move[1]}"]')
                    runtime.wait(f'__gomokuSnapshot().history.length>{before["plies"]}||__gomokuSnapshot().storageBlocked',timeout=60)
                state=capture(runtime,run,manifest,journal,current,'human_round')
                verify_extension(before,state,move)
                record_played(journal,current,pending,state)
                current.update(last=state,pending=None)
                atomic_json(state_file,current)
                atomic_json(run/'run_status.json',summarize(manifest,journal.events,running=True))
                print(canonical_json({'game':current['game_id'],'ai_side':current['ai_side'],'confirmed_plies':state['plies']}),flush=True)
            # An interrupted terminal response still needs its pending click and
            # evidence reconciled before it can be scored.
            if current['pending']:
                pending=current['pending']
                state=capture(runtime,run,manifest,journal,current,'resumed_terminal_round')
                verify_extension(pending['before'],state,pending['choice']['move'])
                record_played(journal,current,pending,state)
                current.update(last=state,pending=None);atomic_json(state_file,current)
            result=verify_terminal_game(state)
            result.update(game_id=current['game_id'],run_id=manifest['run_id'],candidate_id=manifest['candidate_id'],
                          evidence=recorded_evidence(journal,current['game_id']),
                          opponent_moves=[e['data'] for e in journal.events if e['kind']=='human_move_played' and e['data']['game_id']==current['game_id']])
            audit=verify_game_evidence(run,manifest,result)
            result['evidence_audit']=audit
            path=run/'games'/current['game_id']/'result.json'
            if path.exists():
                if read(path)!=result:raise RuntimeError('Existing terminal result differs; never overwrite it')
            else:atomic_json(path,result)
            item={key:result[key] for key in ('game_id','ai_side','outcome','winner','plies','termination','canonical_trajectory')}
            item.update(result_file=path.relative_to(run).as_posix(),result_sha256=file_hash(path))
            journal.append('game_completed',item);completed_now+=1
            current['phase']='completed';atomic_json(state_file,current)
            atomic_json(run/'run_status.json',summarize(manifest,journal.events,running=True))
            print(canonical_json({'completed':item}),flush=True)
    except BaseException as exc:
        error=str(exc)
        journal.append('run_error',{'error':error})
        if runtime:
            try:atomic_json(run/'runtime/error_snapshot.json',runtime.snapshot())
            except Exception:pass
        raise
    finally:
        if challenger:challenger.close()
        if runtime:runtime.close_bridge()
        runner_file=run/'runtime/runner.json'
        if runner_file.exists():
            runner=read(runner_file)
            if runner['pid']==os.getpid():
                runner['ended_utc']=utc();atomic_json(runner_file,runner)
        atomic_json(run/'run_status.json',summarize(manifest,journal.events,error=error))
    print(canonical_json(summarize(manifest,journal.events)),flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--formal',action='store_true')
    parser.add_argument('--seconds',type=float,default=1.0)
    parser.add_argument('--seed',type=int,default=20260906)
    parser.add_argument('--max-games',type=int,default=2000)
    parser.add_argument('--headed',action='store_true')
    args=parser.parse_args()
    if args.max_games<1:parser.error('--max-games must be positive')
    run=Path(args.run_dir).resolve()
    if args.resume:
        manifest=read(run/'manifest.json');verify_frozen(run,manifest)
    else:manifest=freeze(args)
    frozen=run/'frozen/source/tools/browser/match.py'
    if Path(__file__).resolve()!=frozen:
        return subprocess.call([sys.executable,'-X','utf8','-B','-u',str(frozen),'--run-dir',str(run),
            '--resume','--max-games',str(args.max_games)],cwd=run/'frozen/source',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    with exclusive(run/'run.lock'):return main_run(run,manifest,args.max_games)

if __name__=='__main__':raise SystemExit(main())
