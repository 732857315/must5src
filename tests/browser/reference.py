"""Generate CPU reference probabilities for independent browser parity checks."""
import json
from pathlib import Path
import numpy as np
import torch
from unet_pipeline import load_model
from unet_board import analyze_board
from global_inference import load_global_model, analyze_global
from tests.test_board_threat_search import browser_position
from tools.browser.model_sources import verified_checkpoint_paths

ROOT = Path(__file__).resolve().parents[2]


def load_reference_models(manifest):
    # Validate every role before loading any model. A v4 manifest can never be
    # compared against a silently hard-coded v3 CPU reference.
    paths = verified_checkpoint_paths(manifest, root=ROOT)
    opponent, _ = load_model(paths['opponent'], 'opponent')
    play, _ = load_model(paths['play'], 'play')
    global_model, _ = load_global_model(paths['global'])
    return opponent, play, global_model


def main():
    manifest = json.loads((ROOT/'web/browser/assets.json').read_text(encoding='utf-8-sig'))
    torch.set_num_threads(2)
    opponent, play, global_model = load_reference_models(manifest)
    small=np.zeros((5,5),dtype=np.uint8);small[0,:]=3;small[1,1]=1;small[2,2]=2
    big=np.zeros((19,19),dtype=np.uint8);big[0,:]=3;big[7:10,8]=[1,2,1];big[8,7]=2
    tall=np.zeros((19,5),dtype=np.uint8);tall[0,:]=3;tall[7,2]=1;tall[8,2]=2;tall[9,1]=1
    wide=tall.T.copy()
    odd=np.zeros((15,17),dtype=np.uint8);odd[7,8]=1;odd[6,8]=2
    mixed=np.zeros((16,15),dtype=np.uint8);mixed[8,7]=1;mixed[9,7]=2;mixed[:,0]=3
    cases=[]
    for name,board,side in [('actual16',browser_position(),1),('gray5',small,2),('gray19',big,2),('gray5x19',wide,2),('gray19x5',tall,2),('odd15x17',odd,1),('mixed16x15',mixed,1)]:
        local=analyze_board(board,side,opponent,play,tactical=False)
        global_result=analyze_global(board,side,local,global_model)
        cases.append(dict(name=name,n=board.shape[0],cols=board.shape[1],board=board.reshape(-1).tolist(),side=side,
          opponent=local['opponent_policy'].reshape(-1).tolist(),play=local['raw_play_policy'].reshape(-1).tolist(),
          global_policy=global_result['global_policy'].reshape(-1).tolist(),combined=global_result['combined_policy'].reshape(-1).tolist(),
          coverage=local['coverage'].reshape(-1).tolist(),window_count=local['window_count'],value=global_result['value']))
    p=ROOT/'exports/browser/reference.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(cases),encoding='utf-8')
    print('Reference cases:',len(cases))


if __name__=='__main__':main()
