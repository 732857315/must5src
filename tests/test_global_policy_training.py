import copy
import gzip
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
import numpy as np
import torch
from global_data import canonical_board_key, physical_board_key
from global_model import global_loss
from global_policy_training import merge_constraints, preserved_policy_split, constraint_metrics
from train_global import augment_record, batches, policy_metrics, main, json_value, tactical_guard
from global_model import GlobalBoardNet
from tests import test_train_global as legacy


def distinct_record(name, extra=0):
    record=legacy.fixture(game_id=name)
    for i in range(extra):record['board'][0,i+2]=1
    record['physical_key']=physical_board_key(record['board'])
    record['board_key']=canonical_board_key(record['board'],record['side'])
    return record


def constraint(record, group):
    item=copy.deepcopy(record)
    item.update(game_id=group,source='verified_action_constraints',policy_source='verified_action_sets',
                target_policy=np.zeros_like(item['board'],dtype=np.float32),policy_mask=False,
                losing_mask=np.zeros_like(item['board'],dtype=bool),winning_mask=np.zeros_like(item['board'],dtype=bool),
                value=0.,value_valid=False,source_file='fixture.json',source_sha256='a'*64,
                action_evidence=[{'proof_file_sha256':'b'*64}])
    item['losing_mask'][2,5]=True
    return item


class PolicyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def test_rectangular_augmentation_transforms_masks_with_board_and_preserves_old_api(self):
        record=constraint(distinct_record('source'), 'source')
        original=copy.deepcopy(record)
        for turns in range(4):
            for reflect in (False,True):
                for swap in (False,True):
                    values=augment_record(record,turns,reflect,swap,include_policy_constraints=True)
                    self.assertEqual(len(values),9)
                    _,board,target,value,valid,active,weight,bad,good=values
                    point=legacy.mapped_cell((2,5),record['board'].shape,turns,reflect)
                    self.assertTrue(bad[point]);self.assertEqual(int(bad.sum()),1)
                    self.assertFalse(good.any());self.assertFalse(active);self.assertFalse(valid)
                    self.assertFalse(target.any());self.assertFalse(np.any(bad[board!=0]))
        np.testing.assert_array_equal(record['losing_mask'],original['losing_mask'])
        self.assertEqual(len(augment_record(legacy.fixture(),0,False,False)),5)
    def test_all_legal_loss_uniform_targets_are_not_ce_training(self):
        row=distinct_record('old');row['policy_source']='all_legal_proved_loss'
        row['target_policy']=(row['board']==0).astype(np.float32);row['target_policy']/=row['target_policy'].sum()
        packed,_=next(batches([row],1,include_policy_constraints=True))
        inputs,boards,policy,values,valid,active,weights,bad,good=packed
        self.assertFalse(active.item());self.assertEqual(policy.sum().item(),0)
        logits=torch.zeros((1,1,*boards.shape[-2:]),requires_grad=True)
        loss=global_loss(logits,torch.zeros(1),policy,values,valid,boards,policy_mask=active,
                         policy_weights=weights,losing_mask=bad,winning_mask=good,value_weight=0)
        loss.backward();self.assertEqual(float(logits.grad.abs().sum()),0)
        self.assertEqual(policy_metrics([row['target_policy']],[row])['total'],0)
    def test_partial_targets_are_not_reported_as_perfect_policy_accuracy(self):
        row=constraint(distinct_record('x'),'x')
        self.assertEqual(policy_metrics([np.ones_like(row['board'])/row['board'].size],[row])['total'],0)
    def test_source_mean_gives_independent_groups_equal_weight_and_reports_unknown(self):
        rows=[constraint(distinct_record('a'),'a'),constraint(distinct_record('a',1),'a'),constraint(distinct_record('b',2),'b')]
        probabilities=[]
        for row,mass in zip(rows,(.9,.9,.1)):
            p=np.zeros_like(row['board'],dtype=float);p[2,5]=mass;p[2,4]=1-mass;probabilities.append(p)
        metrics=constraint_metrics(probabilities,rows)
        self.assertAlmostEqual(metrics['losing_mass'],.5)
        self.assertAlmostEqual(metrics['quality'],.5)
        self.assertAlmostEqual(metrics['chose_unknown'],.5)
        self.assertEqual(metrics['source_groups'],2)
    def test_merge_rotated_color_swap_retains_original_owner_and_proof_identity(self):
        old=distinct_record('old_train');incoming=constraint(old,'new_source')
        incoming['board']=np.rot90(np.array([0,2,1,3])[incoming['board']])
        incoming['side']=3-old['side']
        incoming['losing_mask']=np.rot90(incoming['losing_mask']);incoming['winning_mask']=np.rot90(incoming['winning_mask'])
        records=[old];report=merge_constraints(records,[incoming])
        self.assertEqual(len(records),1);self.assertEqual(report['merged'],1)
        self.assertEqual(old['game_id'],'old_train');self.assertFalse(old['policy_mask'])
        self.assertTrue(old['losing_mask'][2,5]);self.assertFalse(old['target_policy'].any())
        self.assertEqual(old['constraint_provenance'][0]['action_evidence'][0]['proof_file_sha256'],'b'*64)
        self.assertTrue(old['constraint_provenance'][0]['target_alignments'])
    def test_frozen_membership_survives_appending_sources_and_seed_changes(self):
        rows=[distinct_record('old_train'),distinct_record('old_val',1),constraint(distinct_record('x',2),'new_a'),constraint(distinct_record('x',3),'new_b')]
        split={'train_groups':['old_train'],'validation_groups':['old_val']}
        for seed in (1,2,9):
            train,val=preserved_policy_split(rows,split,seed=seed)
            self.assertIn('old_train',{r['game_id'] for r in train});self.assertIn('old_val',{r['game_id'] for r in val})
            self.assertEqual(len(train),2);self.assertEqual(len(val),2)
            self.assertTrue({r['physical_key'] for r in train}.isdisjoint(r['physical_key'] for r in val))
    def test_all_same_source_prefixes_follow_existing_owner_without_leaking(self):
        train=distinct_record('old_train');val=distinct_record('old_val',1)
        train['constraint_source_groups']=['new_source']
        extra=constraint(distinct_record('x',2),'new_source')
        left,right=preserved_policy_split([train,val,extra],{'train_groups':['old_train'],'validation_groups':['old_val']})
        self.assertIn(extra,left);self.assertEqual(len(right),1)
    def test_sources_bridging_old_holdout_are_rejected_not_redrawn(self):
        rows=[distinct_record('old_train'),distinct_record('old_val',1)]
        for row in rows:row['constraint_source_groups']=['same_source']
        with self.assertRaisesRegex(ValueError,'connects frozen'):
            preserved_policy_split(rows,{'train_groups':['old_train'],'validation_groups':['old_val']})
    def test_tactical_guard_does_not_hide_block_regression_behind_more_wins(self):
        initial={key:{'per_source':{'immediate_win_set':{'correct':80,'total':100},
                                   'forced_block':{'correct':80,'total':100}}}
                 for key in ('policy','combined_policy')}
        current=copy.deepcopy(initial)
        for row in current.values():
            row['per_source']['immediate_win_set']['correct']=90
            row['per_source']['forced_block']['correct']=77
        result=tactical_guard(initial,current,.02)
        self.assertFalse(result['passed'])
        self.assertTrue(result['checks']['policy:immediate_win_set']['passed'])
        self.assertFalse(result['checks']['combined_policy:forced_block']['passed'])
        current['policy']['per_source']['forced_block']['total']=99
        with self.assertRaisesRegex(ValueError,'population changed'):tactical_guard(initial,current,.02)

    def test_cli_trains_saves_and_reloads_with_repeated_constraints_and_skipped_loss_rows(self):
        # The certificate importer has its own independent replay tests. Only
        # that boundary is replaced here; caching, split, model, optimizer,
        # evaluation and checkpoint reload all execute normally.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);local=root/'local';local.mkdir();source=root/'source';source.mkdir()
            hashes={}
            for name in ('opponent.pt','play.pt'):
                path=local/name;path.write_bytes(name.encode())
                hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
            rows=[distinct_record('old_train',i) for i in range(3)]+[distinct_record('old_val',3)]
            rows[1]['policy_source']='all_legal_proved_loss'
            rows[1]['target_policy']=(rows[1]['board']==0).astype(np.float32)
            rows[1]['target_policy']/=rows[1]['target_policy'].sum()
            for row in rows:row.update(search_completed_depth=0,search_requested_depth=0,value_source='unknown')
            with gzip.open(source/'dataset.jsonl.gz','wt',encoding='utf8') as stream:
                for row in rows:stream.write(json.dumps(row,default=json_value)+'\n')
            (source/'dataset_report.json').write_text(json.dumps({'local_model_sha256':hashes}),encoding='utf8')
            (source/'split.json').write_text(json.dumps({'train_groups':['old_train'],'validation_groups':['old_val']}),encoding='utf8')
            before={path.name:path.read_bytes() for path in source.iterdir()}
            additions=[constraint(rows[2],'proof_train'),constraint(rows[3],'proof_validation')]
            torch.manual_seed(42);initial=GlobalBoardNet(input_mode='relative_rgb')
            initial_weights={name:value.clone() for name,value in initial.state_dict().items()}
            output=root/'run'
            with patch('global_policy_constraints.load_policy_constraints',return_value=(additions,{})), redirect_stdout(io.StringIO()):
                summary=main(['--data-source',str(source),'--output-dir',str(output),'--local-models',str(local),
                    '--policy-constraints','verified-fixture.jsonl','--epochs','2','--batch-size','1','--threads','1',
                    '--seed','42','--input-mode','relative_rgb','--value-weight','0',
                    '--constraint-repeats','2','--searched-policy-weight','.1'])
            self.assertTrue(summary['trained']);self.assertEqual(summary['epochs_completed'],2)
            self.assertEqual(summary['split']['train_records'],3);self.assertEqual(summary['split']['validation_records'],1)
            history=json.loads((output/'history.json').read_text())
            for row in history:
                self.assertEqual(row['skipped_optimizer_updates'],1)
                self.assertEqual(row['policy_active_rows']['ce_count'],1)
                self.assertEqual(row['policy_active_rows']['losing_count'],2)
            events=[json.loads(line) for line in (output/'progress.jsonl').read_text().splitlines()]
            sampler=next(event for event in events if event['event']=='training_sampler')
            self.assertEqual(sampler['unique_train_records'],3);self.assertEqual(sampler['train_exposures_per_epoch'],4)
            payload=torch.load(output/'global.pt',map_location='cpu',weights_only=True)
            self.assertTrue(any(not torch.equal(value,initial_weights[name]) for name,value in payload['state_dict'].items()))
            self.assertEqual(payload['checkpoint_selection'],'heldout_constraint_quality_then_legacy_policy_with_tactical_guard')
            self.assertEqual(summary['validation']['constraints']['policy']['source_groups'],1)
            self.assertEqual(before,{path.name:path.read_bytes() for path in source.iterdir()})

    def test_single_new_source_is_train_only_not_both_sides(self):
        rows=[distinct_record('old_train'),distinct_record('old_val',1),constraint(distinct_record('x',2),'new_source')]
        train,val=preserved_policy_split(rows,{'train_groups':['old_train'],'validation_groups':['old_val']})
        self.assertEqual({r['game_id'] for r in val},{'old_val'})
        self.assertIn('new_source',{r['game_id'] for r in train})

if __name__=='__main__':unittest.main()
