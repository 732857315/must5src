"""A fixed browser candidate cannot escape an unfinished or failed run."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.browser import match
from tests.browser.test_match_evidence import Fixture
from tests.browser.test_match_verify import BLACK_WIN
from web_match_runner import Journal,atomic_json,file_hash


class CandidateRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.registry=(self.root/'registry.jsonl').resolve()

    def run_case(self,name,*,formal=False,complete=False):
        run=self.root/name
        fixture=Fixture(run,BLACK_WIN,ai_side=1)
        manifest=dict(fixture.manifest,run_id=name,formal=formal,registry=str(self.registry))
        fixture.manifest=manifest
        for i,payload in enumerate(fixture.payloads):
            payload['run_id']=name
            fixture.write(i)
        atomic_json(run/'manifest.json',manifest)
        journal=Journal(run/'events.jsonl')
        journal.append('run_frozen',{'run_id':name,'candidate_id':manifest['candidate_id']})
        if complete:
            result=fixture.result
            result.update(run_id=name,candidate_id=manifest['candidate_id'])
            result['evidence_audit']=match.verify_game_evidence(run,manifest,result)
            path=run/'games/game_000001/result.json';atomic_json(path,result)
            journal.append('game_allocated',{'game_id':'game_000001','ai_side':1})
            item={k:result[k] for k in ('game_id','ai_side','outcome','winner','plies','termination','canonical_trajectory')}
            item.update(result_file='games/game_000001/result.json',result_sha256=file_hash(path))
            journal.append('game_completed',item)
            atomic_json(run/'current.json',dict(phase='completed',game_id='game_000001',pending=None,last=result))
        return run,manifest,journal

    def registered(self,name='old',**kwargs):
        run,manifest,journal=self.run_case(name,**kwargs)
        match.register_candidate_run(run,manifest)
        return run,manifest,journal

    def rejected_successor(self,pattern):
        run,manifest,_=self.run_case('new',formal=True)
        before=self.registry.read_bytes()
        with self.assertRaisesRegex(RuntimeError,pattern):match.register_candidate_run(run,manifest)
        self.assertEqual(self.registry.read_bytes(),before)

    def test_registration_is_durable_absolute_and_idempotent(self):
        run,manifest,_=self.registered()
        before=self.registry.read_bytes()
        match.register_candidate_run(run,manifest)
        match.assert_registered_candidate(run,manifest)
        self.assertEqual(self.registry.read_bytes(),before)
        entry=Journal(self.registry).events[0]['data']
        self.assertEqual(entry['run_dir'],str(run.resolve()))
        self.assertEqual(entry['manifest_sha256'],match.digest(manifest))

    def test_registered_unstarted_run_cannot_be_abandoned(self):
        self.registered()
        self.rejected_successor('unfinished or unaudited')

    def test_allocated_live_game_cannot_be_abandoned(self):
        run,manifest,journal=self.registered()
        journal.append('game_allocated',{'game_id':'game_000001','ai_side':1})
        atomic_json(run/'current.json',dict(phase='playing',game_id='game_000001',pending={'choice':{'move':[8,8]}}))
        self.rejected_successor('unfinished or unaudited')

    def test_black_loss_in_any_development_or_formal_run_remains_failure(self):
        run,manifest,journal=self.registered()
        journal.append('game_completed',dict(game_id='game_000001',ai_side=1,outcome='loss',canonical_trajectory='loss'))
        self.rejected_successor('previously failed')

    def test_formal_first_thousand_white_nonloss_at_most_500_is_final_failure(self):
        run,manifest,journal=self.registered(formal=True)
        for index in range(1000):
            journal.append('game_completed',dict(game_id=f'game_{index+1:06d}',ai_side=2,
                outcome='win' if index<500 else 'loss',canonical_trajectory=str(index)))
        self.rejected_successor('previously failed')

    def test_completed_formal_sample_below_target_requires_resume(self):
        self.registered(formal=True,complete=True)
        self.rejected_successor('unfinished formal acceptance')

    def test_complete_audited_nonformal_run_can_start_formal_without_copying_scores(self):
        old,old_manifest,_=self.registered(complete=True)
        run,manifest,journal=self.run_case('new',formal=True)
        match.register_candidate_run(run,manifest)
        match.assert_registered_candidate(run,manifest)
        self.assertEqual(match.summarize(manifest,journal.events)['completed_attempts'],0)
        self.assertEqual(len(Journal(self.registry).events),2)
        self.assertEqual(match.read(old/'manifest.json'),old_manifest)

    def test_development_run_cannot_be_redrawn_as_another_development_run(self):
        self.registered(complete=True)
        run,manifest,_=self.run_case('new')
        with self.assertRaisesRegex(RuntimeError,'resume it or begin formal'):match.register_candidate_run(run,manifest)

    def test_missing_or_changed_historical_result_blocks_successor(self):
        run,_,_=self.registered(complete=True)
        (run/'games/game_000001/result.json').write_text('{}',encoding='utf-8')
        self.rejected_successor('terminal result is unavailable or changed')

    def test_unverified_evidence_is_not_a_completed_development_run(self):
        run,manifest,journal=self.registered(complete=True)
        item=next(e['data'] for e in journal.events if e['kind']=='game_completed')
        result=match.read(run/item['result_file']);result['evidence_audit']={'verified':True}
        atomic_json(run/item['result_file'],result)
        item['result_sha256']=file_hash(run/item['result_file'])
        # Rebuild an internally valid journal; the independent artifact audit must still reject it.
        events=[(e['kind'],e['data']) for e in journal.events]
        path=run/'events.jsonl';path.unlink();journal=Journal(path)
        for kind,data in events:journal.append(kind,data)
        self.rejected_successor('complete evidence audit')

    def test_changed_formal_flag_or_unregistered_resume_is_rejected(self):
        run,manifest,_=self.registered()
        changed=dict(manifest,formal=True)
        with self.assertRaisesRegex(RuntimeError,'manifest changed'):match.assert_registered_candidate(run,changed)
        another,other,_=self.run_case('new',formal=True)
        with self.assertRaisesRegex(RuntimeError,'not registered'):match.assert_registered_candidate(another,other)

    def test_missing_previous_run_cannot_erase_candidate_ownership(self):
        run,_,_=self.registered()
        (run/'manifest.json').unlink()
        self.rejected_successor('Prior candidate run is unavailable')

    def lock_in_other_process(self,path):
        code="from pathlib import Path; import sys; from web_match_runner import exclusive; scope=exclusive(Path(sys.argv[1])); scope.__enter__(); print('locked',flush=True); sys.stdin.readline(); scope.__exit__(None,None,None)"
        child=subprocess.Popen([sys.executable,'-X','utf8','-c',code,str(path)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        def cleanup():
            if child.poll() is None:
                child.stdin.write('release\n');child.stdin.flush()
            child.communicate(timeout=10)
        self.addCleanup(cleanup)
        self.assertEqual(child.stdout.readline().strip(),'locked')
        return child

    def test_cross_process_registry_lock_prevents_competing_registration(self):
        run,manifest,_=self.run_case('new')
        self.lock_in_other_process(self.registry.with_suffix('.lock'))
        with self.assertRaises(OSError):match.register_candidate_run(run,manifest)
        self.assertFalse(self.registry.exists())

    def test_running_development_process_cannot_be_replaced_even_after_a_terminal_game(self):
        run,_,_=self.registered(complete=True)
        self.lock_in_other_process(run/'run.lock')
        self.rejected_successor('running, unfinished, or unaudited')


if __name__=='__main__':unittest.main()
