"""Export the trained models and freestanding search into an offline static app."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import warnings
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from unet_pipeline import load_model
from global_inference import load_global_model
from tools.browser.model_sources import (DEFAULT_CHECKPOINTS, resolve_checkpoint,
                                         checkpoint_metadata, verified_checkpoint_paths)

OUT = ROOT / 'web/browser'


class ExportGlobal(nn.Module):
    def __init__(self, model, tokens=8):
        super().__init__()
        self.model = model
        self.token_rows, self.token_cols = (tokens, tokens) if isinstance(tokens, int) else tokens

    def forward(self, x):
        m = self.model
        encoded = m.spatial(m.stem(x))
        projected = m.token_projection(encoded)
        h, w = projected.shape[-2:]
        nr, nc = self.token_rows, self.token_cols
        # Slice/ReduceMean exports dynamic adaptive pooling exactly, including
        # sizes not divisible by eight. Traced adaptive_avg_pool2d cannot.
        rows = []
        for r in range(nr):
            cells = []
            for c in range(nc):
                patch = projected[:, :, (r*h)//nr:((r+1)*h+nr-1)//nr,
                                  (c*w)//nc:((c+1)*w+nc-1)//nc]
                cells.append(patch.mean((-2,-1), keepdim=True))
            rows.append(torch.cat(cells, dim=-1))
        tokens = torch.cat(rows, dim=-2).flatten(2).transpose(1,2)
        tokens = m.attention(tokens)
        context = tokens.transpose(1,2).reshape(x.shape[0], m.token_dim,nr,nc)
        context = F.interpolate(m.context_projection(context), size=x.shape[-2:], mode='bilinear',align_corners=False)
        fused = m.fusion(torch.cat((encoded,context),dim=1))
        values = m.value_head(torch.cat((fused.mean((-2,-1)),tokens.mean(1)),dim=1)).squeeze(-1)
        return m.policy_head(fused), values


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-models', default=None,
                        help='Directory containing opponent.pt and play.pt; relative to the repository root')
    parser.add_argument('--global-checkpoint', default=DEFAULT_CHECKPOINTS['global'],
                        help='Global checkpoint to export; relative paths use the repository root (default: global_v3)')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Validate sources before creating or overwriting any browser export files.
    source_paths = {role: (resolve_checkpoint(role, Path(args.local_models) / (role + '.pt'), root=ROOT)
                          if args.local_models is not None else resolve_checkpoint(role, root=ROOT))
                    for role in ('opponent', 'play')}
    source_paths['global'] = resolve_checkpoint('global', args.global_checkpoint, root=ROOT)
    sources = {role: checkpoint_metadata(role, path, root=ROOT) for role, path in source_paths.items()}
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    (OUT/'models').mkdir(exist_ok=True)
    (OUT/'vendor').mkdir(exist_ok=True)
    checks = []
    torch.manual_seed(20260906)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    for role in ('opponent','play'):
        path = source_paths[role]
        model, meta = load_model(path, role)
        sources[role].update(trained=meta.get('trained',True))
        target = OUT/f'models/{role}.onnx'
        inputs = (torch.rand(2,3,5,5), torch.tensor([1,2]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            torch.onnx.export(model, inputs, target, dynamo=False, opset_version=17,
                input_names=['rgb','side'], output_names=['logits'],
                dynamic_axes={'rgb':{0:'batch'},'side':{0:'batch'},'logits':{0:'batch'}})
        session = ort.InferenceSession(str(target), options, providers=['CPUExecutionProvider'])
        for batch in (1,3,32):
            rgb = torch.rand(batch,3,5,5)
            sides = (torch.arange(batch)%2+1).long()
            with torch.no_grad():
                expected = model(rgb,sides).numpy()
            actual = session.run(None,{'rgb':rgb.numpy(),'side':sides.numpy()})[0]
            error = float(np.max(np.abs(actual-expected)))
            np.testing.assert_allclose(actual, expected, atol=3e-5,rtol=3e-5)
            checks.append(dict(model=role,batch=batch,max_abs_error=error))
    path = source_paths['global']
    model, meta = load_global_model(path)
    sources['global'].update(input_mode=model.input_mode,
                             trained=meta.get('trained',True),value_used_in_search=False)
    for token_rows in (5,6,7,8):
        for token_cols in (5,6,7,8):
            if token_rows == token_cols:
                key = 'global' if token_rows == 8 else f'global{token_rows}'
            else:
                key = f'global{token_rows}x{token_cols}'
            target = OUT/f'models/{key}.onnx'
            wrapper = ExportGlobal(model,(token_rows,token_cols)).eval()
            shape = (16 if token_rows==8 else token_rows, 16 if token_cols==8 else token_cols)
            x = torch.rand(1,9,*shape)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                torch.onnx.export(wrapper,x,target,dynamo=False,opset_version=17,
                    input_names=['inputs'],output_names=['logits','value'],
                    dynamic_axes={'inputs':{2:'height',3:'width'},'logits':{2:'height',3:'width'}})
            session = ort.InferenceSession(str(target), options, providers=['CPUExecutionProvider'])
            shapes = ((8,8),(16,16),(19,17),(32,32)) if token_rows==token_cols==8 else (shape,)
            for shape in shapes:
                x = torch.rand(1,9,*shape)
                with torch.no_grad():
                    expected = model(x)
                actual = session.run(None,{'inputs':x.numpy()})
                errors = []
                for a,b in zip(actual,expected):
                    np.testing.assert_allclose(a,b.numpy(),atol=4e-5,rtol=4e-5)
                    errors.append(float(np.max(np.abs(a-b.numpy()))))
                checks.append(dict(model=target.name,shape=shape,tokens=[token_rows,token_cols],max_abs_error=errors))
    # Reject a source replaced while the ONNX variants were being produced.
    verified_checkpoint_paths({'models': sources}, root=ROOT)
    dist = ROOT/'node_modules/onnxruntime-web/dist'
    for name in ('ort.wasm.min.mjs','ort-wasm-simd-threaded.mjs','ort-wasm-simd-threaded.wasm'):
        shutil.copyfile(dist/name,OUT/'vendor'/name)
    shutil.copyfile(ROOT/'web/browser/ONNX-RUNTIME-LICENSE.txt',OUT/'vendor/ONNX-RUNTIME-LICENSE')
    shutil.copyfile(ROOT/'web/browser/ONNX-RUNTIME-ThirdPartyNotices.txt',OUT/'vendor/ONNX-RUNTIME-ThirdPartyNotices.txt')
    compiler = shutil.which('clang')
    if not compiler:
        raise RuntimeError('Building browser search requires clang with the wasm32 target and wasm-ld')
    subprocess.run([compiler,'--target=wasm32','-O3','-ffreestanding','-fno-builtin','-nostdlib',
        '-Wl,--no-entry','-Wl,--export-all','-Wl,--initial-memory=16777216','-Wl,--max-memory=33554432',
        '-Wl,-z,stack-size=262144',str(ROOT/'native/browser_search.c'),'-o',str(OUT/'search.wasm')], check=True)
    report = dict(format='must5_browser_build_v1',runtime='onnxruntime-web@1.29.0',
                  models=sources,numerical_checks=checks,default_time_seconds=1.0)
    (ROOT/'exports/browser').mkdir(parents=True,exist_ok=True)
    (ROOT/'exports/browser/model_export_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    names = ['index.html','style.css','app.mjs','core.mjs','geometry.mjs','star-points.mjs','search-budget.mjs','engine-worker.mjs',
             'app.webmanifest','icon.svg','search.wasm']
    names += [str(p.relative_to(OUT)).replace('\\','/') for folder in ('models','vendor')
              for p in sorted((OUT/folder).iterdir()) if p.is_file()]
    assets = {name:dict(sha256=digest(OUT/name),bytes=(OUT/name).stat().st_size) for name in names}
    version = hashlib.sha256(json.dumps(assets,sort_keys=True).encode()).hexdigest()[:20]
    (OUT/'assets.json').write_text(json.dumps(dict(version=version,assets=assets,models=sources),indent=2),encoding='utf-8')
    template = (OUT/'sw.template.js').read_text(encoding='utf-8-sig')
    (OUT/'sw.js').write_text(template.replace('__VERSION__',json.dumps(version)).replace('__FILES__',json.dumps(['./'+x for x in names+['assets.json']])),encoding='utf-8')
    report.update(version=version,total_asset_bytes=sum(x['bytes'] for x in assets.values()))
    (ROOT/'exports/browser/model_export_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    archive = ROOT/'exports/browser/must5-browser.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for name in names+['assets.json','sw.js','DEPLOY.md']:
            bundle.write(OUT/name,name)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError('Browser archive failed its integrity check')
    release = dict(archive=str(archive),sha256=digest(archive),bytes=archive.stat().st_size,
                   asset_version=version,files=len(names)+3)
    (ROOT/'exports/browser/release.json').write_text(json.dumps(release,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
