"""Recovery tests: unfinished browser games cannot be replaced or double counted."""
import copy
import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.browser import match
from tools.browser.match_verify import verify_browser_state
from tools.browser.match_runtime import OBSERVE
from web_match_runner import Journal


def raw_state(history=(), human=2, started=False):
    board=[0]*256
    for i,p in enumerate(history):board[p]=1+i%2
    return dict(n=16,board=board,history=list(history),human=human,seconds=1.,
                started=started,ready=True,busy=False,revision=len(history),analysis=None,
                storageBlocked=False,storageError=None,pendingCommit=None)


class Runtime:
    def __init__(self,raw):self.raw=copy.deepcopy(raw);self.clicked=[];self.configured=[]
    def idle(self):return copy.deepcopy(self.raw)
    snapshot=idle
    def click(self,selector):
        self.clicked.append(selector)
        if selector=='#retry-save':
            self.raw.update(storageBlocked=False,storageError=None,pendingCommit=None)
        elif selector=='#start':
            self.raw['started']=True
            if self.raw['human']==2 and not self.raw['history']:
                self.raw['history']=[136];self.raw['board'][136]=1;self.raw['revision']+=1
        elif selector=='#new':self.raw=raw_state(human=self.raw['human'])
        else:raise AssertionError(selector)
    def configure_human(self,side):self.configured.append(side);self.raw['human']=side


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name).resolve();self.journal=Journal(self.run/'events.jsonl')
        self.manifest={'seconds':1.}
        def captured(runtime,run,manifest,journal,current,stage):
            return verify_browser_state(runtime.idle(),current['ai_side'],manifest['seconds'])
        self.capture_patch=patch.object(match,'capture',side_effect=captured)
        self.capture_patch.start();self.addCleanup(self.capture_patch.stop)

    def current(self,side=1,phase='prepare'):
        return dict(game_id='game_000001',game_index=1,ai_side=side,phase=phase,last=None,pending=None)

    def prepare(self,runtime,current):
        return match.prepare(runtime,self.run,self.manifest,self.journal,current)

    def test_completed_black_opening_resumes_without_start_or_reset(self):
        current=self.current(phase='opening')
        current['opening']=verify_browser_state(raw_state(),1)
        runtime=Runtime(raw_state([136],started=True))
        state=self.prepare(runtime,current)
        self.assertEqual(state['plies'],1);self.assertEqual(runtime.clicked,[])
        self.assertEqual(current['phase'],'playing')

    def test_opening_intent_saved_before_start_resumes_same_opening(self):
        current=self.current(phase='opening')
        current['opening']=verify_browser_state(raw_state(),1)
        runtime=Runtime(raw_state())
        state=self.prepare(runtime,current)
        self.assertEqual(runtime.clicked,['#start']);self.assertEqual(state['plies'],1)

    def test_white_empty_started_opening_is_valid_with_durable_intent(self):
        current=self.current(2,'opening')
        current['opening']=verify_browser_state(raw_state(human=1),2)
        runtime=Runtime(raw_state(human=1,started=True))
        self.assertEqual(self.prepare(runtime,current)['plies'],0)
        self.assertFalse(runtime.clicked)

    def test_opening_storage_failure_retries_without_restarting(self):
        current=self.current(phase='opening')
        current['opening']=verify_browser_state(raw_state(),1)
        raw=raw_state([136],started=True)
        raw.update(storageBlocked=True,storageError='quota',pendingCommit={'record':{'history':[136]}})
        runtime=Runtime(raw)
        self.assertEqual(self.prepare(runtime,current)['plies'],1)
        self.assertEqual(runtime.clicked,['#retry-save'])
        self.assertEqual(self.journal.events[0]['kind'],'storage_retry_intent')

    def test_existing_rectangular_profile_is_rejected_without_reset_or_start(self):
        for rows, cols, length in ((8, 12, 96), (16, 12, 192), (16, 12, 256)):
            snapshot = raw_state()
            snapshot.update(n=rows, cols=cols, board=[0]*length)
            runtime = Runtime(snapshot)
            with self.subTest(rows=rows, cols=cols, length=length), self.assertRaises((ValueError, RuntimeError)):
                self.prepare(runtime, self.current())
            self.assertEqual(runtime.raw, snapshot)
            self.assertEqual(runtime.clicked, [])
            self.assertEqual(runtime.configured, [])
            self.assertEqual(self.journal.events, [])

    def test_unrecorded_live_game_is_never_reset(self):
        runtime=Runtime(raw_state([136,135],started=True))
        with self.assertRaisesRegex(RuntimeError,'cannot be reset'):self.prepare(runtime,self.current())
        self.assertFalse(runtime.clicked)

    def test_started_game_without_opening_intent_is_preserved(self):
        runtime=Runtime(raw_state(human=1,started=True))
        with self.assertRaisesRegex(RuntimeError,'no durable opening'):self.prepare(runtime,self.current(2))
        self.assertFalse(runtime.clicked)

    def test_unexplained_second_round_during_opening_is_rejected(self):
        current=self.current(phase='opening')
        current['opening']=verify_browser_state(raw_state(),1)
        runtime=Runtime(raw_state([136,135,137],started=True))
        with self.assertRaises(ValueError):self.prepare(runtime,current)
        self.assertFalse(runtime.clicked)

    def test_opening_is_durable_before_actual_start_input(self):
        current=self.current();runtime=Runtime(raw_state(human=1))
        original=runtime.click
        def click(selector):
            saved=match.read(self.run/'current.json')
            self.assertEqual(saved['phase'],'opening');self.assertEqual(saved['opening']['plies'],0)
            self.assertEqual(self.journal.events[-1]['kind'],'game_started')
            original(selector)
        runtime.click=click
        self.prepare(runtime,current)
        self.assertEqual(runtime.configured,[2])

    def test_duplicate_played_journal_recovery_is_idempotent(self):
        current=self.current(2)
        before=verify_browser_state(raw_state(human=1,started=True),2)
        after=verify_browser_state(raw_state([136,135],human=1,started=True),2)
        pending={'before':before,'choice':{'move':[8,8],'nodes':20}}
        match.record_played(self.journal,current,pending,after)
        match.record_played(self.journal,current,copy.deepcopy(pending),after)
        self.assertEqual(len(self.journal.events),1)
        pending['choice']['move']=[8,7]
        with self.assertRaisesRegex(RuntimeError,'Conflicting'):match.record_played(self.journal,current,pending,after)
        self.assertEqual(len(self.journal.events),1)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name).resolve();self.journal=Journal(self.run/'events.jsonl')
        self.manifest={'seconds':1.,'run_id':'test-run','candidate_id':'test-candidate'}
        self.current={'game_id':'game_000001','ai_side':2}
        self.runtime=Runtime(raw_state(human=1,started=True));self.acks=[]
        self.runtime.saved=lambda: dict(self.runtime.idle(),version=1)
        self.runtime.audit=lambda: {'page_id':'test-page','events':[{'seq':1,'kind':'test'}]}
        self.runtime.ack=self.acks.append
        def screenshot(path):
            Path(path).write_bytes(b'\x89PNG\r\n\x1a\nfixture')
            return match.file_hash(path)
        self.runtime.screenshot=screenshot

    def test_ai_error_cannot_commit_an_incomplete_human_round(self):
        self.runtime.raw=raw_state([136],human=1,started=True)
        with self.assertRaisesRegex(ValueError,'human turn'):
            match.capture(self.runtime,self.run,self.manifest,self.journal,self.current,'human_round')
        self.assertEqual(self.acks,[]);self.assertEqual(self.journal.events,[])

    def test_interrupted_evidence_write_remains_unacknowledged_and_is_not_recovered(self):
        with patch.object(match.os,'fsync',side_effect=OSError('interrupted')):
            with self.assertRaisesRegex(OSError,'interrupted'):
                match.capture(self.runtime,self.run,self.manifest,self.journal,self.current,'opening')
        folder=self.run/'games/game_000001'
        self.assertEqual(len(list(folder.glob('*.pending'))),1)
        self.assertEqual(list(folder.glob('*.json.gz')),[])
        self.assertEqual(self.acks,[]);self.assertEqual(self.journal.events,[])
        match.restore_evidence(self.run,self.manifest,self.journal,self.current)
        self.assertEqual(self.journal.events,[])

    def test_complete_unjournaled_file_is_recovered_without_replacing_any_artifact(self):
        match.capture(self.runtime,self.run,self.manifest,self.journal,self.current,'opening')
        reference=self.journal.events[0]['data'];original=match.file_hash(self.run/reference['file'])
        recovered=Journal(self.run/'recovered.jsonl')
        match.restore_evidence(self.run,self.manifest,recovered,self.current)
        match.restore_evidence(self.run,self.manifest,recovered,self.current)
        self.assertEqual(len(recovered.events),1)
        self.assertTrue(recovered.events[0]['data']['recovered_after_interruption'])
        self.assertEqual(recovered.events[0]['data']['sha256'],original)
        self.assertEqual(match.file_hash(self.run/reference['file']),original)
        self.assertEqual(self.acks,[1])

class ObserverTests(unittest.TestCase):
    def test_width_only_changes_emit_positions_and_legacy_width_defaults_to_rows(self):
        # Execute the actual observer in a tiny DOM environment; no browser,
        # worker inference, service or persisted user storage is involved.
        script = r"""
import vm from 'node:vm';
let notify, now=0;
const state={n:16,human:1,seconds:1,started:false,history:[],board:Array(256).fill(0)};
const window={Worker:class {},addEventListener(){},__gomokuSnapshot:()=>state};
const document={addEventListener(){},querySelectorAll(){
  return state.board.map((_,point)=>({dataset:{point:String(point)},
    classList:{contains:()=>false},querySelector:()=>null}));
}};
const context={window,document,JSON,ArrayBuffer,crypto:{randomUUID:()=>"page-fixture"},
 performance:{now:()=>++now},localStorage:{getItem:()=>JSON.stringify(state)},
 MutationObserver:class {constructor(callback){notify=callback;}observe(){}}};
vm.runInNewContext(OBSERVATION_SOURCE,context);
notify();
state.cols=16;notify(); // Explicit square metadata adds no new position.
for(const cols of [12,10]){state.cols=cols;state.board=Array(16*cols).fill(0);notify();}
delete state.cols;state.board=Array(256).fill(0);notify();notify();
console.log(JSON.stringify(window.__getGomokuAudit().events.map(e=>({kind:e.kind,
 cols:e.data.state.cols??e.data.state.n,cells:e.data.visible_cells.length,
 savedCells:JSON.parse(e.data.saved).board.length,history:e.data.state.history}))));
""".replace('OBSERVATION_SOURCE', json.dumps(OBSERVE))
        completed = subprocess.run(['node', '--input-type=module', '-'], input=script,
                                   text=True, capture_output=True, check=True)
        events = json.loads(completed.stdout)
        self.assertEqual([event['kind'] for event in events], ['position']*4)
        self.assertEqual([event['cols'] for event in events], [16, 12, 10, 16])
        self.assertEqual([event['cells'] for event in events], [256, 192, 160, 256])
        self.assertTrue(all(event['cells'] == event['savedCells'] for event in events))
        self.assertTrue(all(event['history'] == [] for event in events))


if __name__=='__main__':unittest.main()
