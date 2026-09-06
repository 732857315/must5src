"""Real Chromium checks: touch UI, local inference parity, offline reload and play."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlsplit
import uuid
import numpy as np
from web_match_runner import JsonProcess, locate_browser

ROOT=Path(__file__).resolve().parents[2]

def safe_base_path(value):
    """A URL directory made of ordinary relative path segments, never traversal."""
    if value == '':
        return ''
    reserved={'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}
    parts=value.split('/')
    if (len(value)>240 or any(not re.fullmatch(r'[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*', part)
            or part.split('.')[0].upper() in reserved for part in parts)):
        raise argparse.ArgumentTypeError('--base-path must contain safe relative names, such as must5 or apps/must5')
    return value


def https_target(value):
    """An explicit HTTPS directory URL, without credentials, query or fragment."""
    try:
        parsed=urlsplit(value)
        port=parsed.port
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Invalid HTTPS target URL') from exc
    if (parsed.scheme!='https' or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or '?' in value or '#' in value or not parsed.path.endswith('/')
            or any(char.isspace() or ord(char)<32 for char in value) or '\\' in value
            or (port is not None and not 1<=port<=65535)):
        raise argparse.ArgumentTypeError('--url requires an HTTPS directory URL ending in /, without credentials, query or fragment')
    return value


def verify_online_assets(url, manifest):
    """Read metadata only; numerical comparisons require this exact release."""
    request=urllib.request.Request(url+'assets.json',headers={'Cache-Control':'no-cache'})
    with urllib.request.urlopen(request,timeout=30) as response:
        final_url=response.geturl()
        if urlsplit(final_url).scheme!='https':
            raise ValueError('Online assets redirected away from HTTPS')
        data=response.read(2*1024*1024+1)
    if len(data)>2*1024*1024:
        raise ValueError('Online assets manifest is unexpectedly large')
    remote=json.loads(data)
    if remote.get('version')!=manifest['version']:
        raise ValueError('Online/local asset versions differ; numerical parity refused: '
                         +str(remote.get('version'))+' != '+str(manifest['version']))
    if remote.get('assets')!=manifest['assets']:
        raise ValueError('Online asset hashes differ despite a matching version; parity refused')
    return data,dict(url=final_url,version=remote['version'],sha256=hashlib.sha256(data).hexdigest(),
                     local_version=manifest['version'],asset_hashes_match=True)


def stage_assets(out, base_path, manifest):
    """Freeze only declared runtime assets under this QA run's unique staging root."""
    base_path=safe_base_path(base_path)
    if not base_path:
        raise ValueError('staging requires a nonempty base path')
    source=(ROOT/'web/browser').resolve()
    staging=(out/'staging').resolve()
    destination=(staging/base_path).resolve()
    if not destination.is_relative_to(staging):
        raise ValueError('staged base path escapes the QA directory')
    assets=manifest.get('assets')
    if not isinstance(assets,dict) or not assets:
        raise ValueError('assets.json has no declared runtime resources')
    names=list(dict.fromkeys([*assets,'assets.json','sw.js']))
    copies=[]
    for name in names:
        if not isinstance(name,str):
            raise ValueError('asset paths must be strings')
        relative=PurePosixPath(name)
        if (relative.is_absolute() or any(part in ('.','..') for part in name.split('/'))
                or '\\' in name or ':' in name or '' in name.split('/')):
            raise ValueError('unsafe asset path: '+name)
        original=(source/name).resolve()
        target=(destination/name).resolve()
        if not original.is_relative_to(source) or not target.is_relative_to(destination) or not original.is_file():
            raise ValueError('missing or escaping asset: '+name)
        copies.append((name,original,target))
    for name,original,target in copies:
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(original,target)
        if name in assets and hashlib.sha256(target.read_bytes()).hexdigest()!=assets[name]['sha256']:
            raise ValueError('staged asset SHA differs from manifest: '+name)
    if json.loads((destination/'assets.json').read_text(encoding='utf-8'))!=manifest:
        raise ValueError('assets.json changed during staging')
    return staging


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    target=parser.add_mutually_exclusive_group()
    target.add_argument('--base-path',type=safe_base_path,default='',
        help='Serve the full QA under a relative project path, e.g. must5 or apps/must5 (default: origin root)')
    target.add_argument('--url',type=https_target,
        help='Explicit existing HTTPS deployment, e.g. https://example.com/must5/; no static server is started or stopped')
    args=parser.parse_args(argv)
    # Remote first-load downloads include the ORT WASM and all offline assets.
    # Keep this allowance separate from every actual move/search wait below.
    initial_timeout=180 if args.url else 60
    manifest=json.loads((ROOT/'web/browser/assets.json').read_text(encoding='utf-8'))
    out=ROOT/'exports/browser/checks'/('qa_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
    staging=None
    if args.url:
        url=args.url
        parsed=urlsplit(url)
        origin=f'{parsed.scheme}://{parsed.netloc}'
        server_command=None
    else:
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        origin=f'http://127.0.0.1:{port}'
        url=origin+'/' + (args.base_path+'/' if args.base_path else '')
        if args.base_path:
            staging=stage_assets(out,args.base_path,manifest)
            server_command=[sys.executable,'-X','utf8','-B','-m','http.server',str(port),
                '--bind','127.0.0.1','--directory',str(staging)]
        else:
            server_command=[sys.executable,'-X','utf8','-B',str(ROOT/'tools/browser/serve.py'),'--port',str(port)]
    log=(out/'process.log').open('ab')
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    profile=out/'profile'
    server=browser=bridge=None
    report={'url':url,'origin':origin,'base_path':urlsplit(url).path.strip('/'),
        'external_url':args.url,'local_static_server_started':False,'local_static_server_stopped_for_offline':False,
        'offline_mode':'browser_network_only' if args.url else 'local_server_stopped_and_browser_network_offline',
        'staging':str(staging) if staging else None,'asset_version':manifest['version'],'checks':[],'passed':False}
    def check(name,details):report['checks'].append({'name':name,'details':details});print(name,json.dumps(details,ensure_ascii=False),flush=True)
    try:
        if args.url:
            remote_bytes,report['online_assets']=verify_online_assets(url,manifest)
            (out/'online_assets.json').write_bytes(remote_bytes)
        if server_command is not None:
            server=subprocess.Popen(server_command,cwd=ROOT,stdout=log,stderr=log,creationflags=flags)
            report['local_static_server_started']=True
        browser=subprocess.Popen([locate_browser(),'--headless=new','--no-first-run','--no-default-browser-check','--disable-crash-reporter','--disable-breakpad',
            '--remote-debugging-port=0','--remote-debugging-address=127.0.0.1',f'--user-data-dir={profile}',
            '--window-size=1360,1100','about:blank'],stdout=log,stderr=log,creationflags=flags)
        active=profile/'DevToolsActivePort';deadline=time.monotonic()+30
        while not active.exists():
            if time.monotonic()>deadline:raise TimeoutError('browser startup')
            time.sleep(.1)
        debug=int(active.read_text().splitlines()[0])
        pages=json.load(urllib.request.urlopen(f'http://127.0.0.1:{debug}/json/list'))
        page=next(p for p in pages if p['type']=='page')
        bridge=JsonProcess([shutil.which('node'),str(ROOT/'web_match_cdp.mjs'),page['webSocketDebuggerUrl'],origin],ROOT,out/'cdp.log')
        def cdp(method,params=None):return bridge.request({'method':method,'params':params or {}},timeout=60)
        def js(expression):
            r=cdp('Runtime.evaluate',{'expression':expression,'returnByValue':True,'awaitPromise':True})
            if 'exceptionDetails' in r:raise RuntimeError(str(r['exceptionDetails']))
            return r.get('result',{}).get('value')
        def wait(expression,timeout=45):
            end=time.monotonic()+timeout
            while time.monotonic()<end:
                if js(expression):return
                time.sleep(.15)
            raise TimeoutError(expression+' '+str(js("document.getElementById('message')?.textContent")))
        def click(selector,touch=False):
            rect=js(f"(()=>{{const e=document.querySelector({json.dumps(selector)});e.scrollIntoView({{block:'center'}});const r=e.getBoundingClientRect();return {{x:r.x+r.width/2,y:r.y+r.height/2}};}})()")
            if touch:
                cdp('Input.dispatchTouchEvent',{'type':'touchStart','touchPoints':[dict(rect,radiusX=2,radiusY=2)]})
                cdp('Input.dispatchTouchEvent',{'type':'touchEnd','touchPoints':[]})
            else:
                cdp('Input.dispatchMouseEvent',dict(type='mousePressed',button='left',clickCount=1,**rect))
                cdp('Input.dispatchMouseEvent',dict(type='mouseReleased',button='left',clickCount=1,**rect))
        def number(selector,value,touch=False):
            click(selector,touch)
            cdp('Input.dispatchKeyEvent',{'type':'keyDown','key':'a','code':'KeyA','windowsVirtualKeyCode':65,'modifiers':2})
            cdp('Input.dispatchKeyEvent',{'type':'keyUp','key':'a','code':'KeyA','windowsVirtualKeyCode':65,'modifiers':2})
            cdp('Input.insertText',{'text':str(value)})
            cdp('Input.dispatchKeyEvent',{'type':'keyDown','key':'Tab','code':'Tab','windowsVirtualKeyCode':9})
            cdp('Input.dispatchKeyEvent',{'type':'keyUp','key':'Tab','code':'Tab','windowsVirtualKeyCode':9})
        def shape(rows,cols,touch=False):
            number('#size',rows,touch);number('#cols',cols,touch)
            state=js('__gomokuSnapshot()')
            assert (state['n'],state['cols'],len(state['board']))==(rows,cols,rows*cols)
            assert not state['started'] and not state['history']
        def geometry(rows,cols):
            details=js("""(()=>{const b=document.getElementById('board'),s=document.getElementById('board-scroll'),cells=[...b.querySelectorAll('.cell')];return {rows:Number(b.getAttribute('aria-rowcount')),cols:Number(b.getAttribute('aria-colcount')),cells:cells.length,stones:b.querySelectorAll('.stone').length,stars:[...b.querySelectorAll('[data-star]')].map(e=>{const r=e.getBoundingClientRect(),m=e.querySelector('.star-point').getBoundingClientRect();return {point:Number(e.dataset.point),kind:e.dataset.star,offsetX:Math.abs(r.x+r.width/2-m.x-m.width/2),offsetY:Math.abs(r.y+r.height/2-m.y-m.height/2)}}),cellRects:cells.map(e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}}),documentOverflow:document.documentElement.scrollWidth>window.innerWidth,boardWidth:b.getBoundingClientRect().width,boardHeight:b.getBoundingClientRect().height,viewportHeight:innerHeight,scrollWidth:s.clientWidth}})()""")
            assert (details['rows'],details['cols'],details['cells'])==(rows,cols,rows*cols)
            expected_count=5 if rows%2 and cols%2 else 4
            assert len(details['stars'])==expected_count
            points={item['point'] for item in details['stars']}
            for item in details['stars']:
                r,c=divmod(item['point'],cols)
                assert (rows-1-r)*cols+c in points and r*cols+(cols-1-c) in points
                assert item['offsetX']<.6 and item['offsetY']<.6
            center=[x['point'] for x in details['stars'] if x['kind']=='tianyuan']
            assert center==([rows//2*cols+cols//2] if rows%2 and cols%2 else [])
            for rect in details['cellRects']:
                assert rect['width']>0 and abs(rect['width']-rect['height'])<.6
            assert not details['documentOverflow'] and details['boardWidth']<=details['scrollWidth']+1
            assert details['boardHeight']<=min(.72*details['viewportHeight'],900)+1
            details['cell_sample']=details.pop('cellRects')[0]
            return details
        def screenshot(name):
            data=cdp('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True})['data']
            (out/name).write_bytes(base64.b64decode(data))
        cdp('Page.enable');cdp('Runtime.enable');cdp('Network.enable')
        cdp('Page.addScriptToEvaluateOnNewDocument',{'source':"window.__errors=[];window.addEventListener('error',e=>__errors.push(e.message));window.addEventListener('unhandledrejection',e=>__errors.push(String(e.reason)));window.__trusted=[];document.addEventListener('click',e=>__trusted.push({trusted:e.isTrusted,target:e.target.id||e.target.closest('[data-point]')?.dataset.point}));"})
        cdp('Page.navigate',{'url':url});wait('window.__gomokuSnapshot?.().ready',initial_timeout)
        wait("document.getElementById('offline').textContent.includes('已就绪')",initial_timeout)
        assert js('location.href')==url
        scope=js('navigator.serviceWorker.ready.then(reg=>reg.scope)')
        assert scope==url, (scope,url)
        report['service_worker_scope']=scope
        if args.url:
            page_assets=js("fetch('./assets.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('assets HTTP '+r.status);return r.json()})")
            if page_assets.get('version')!=manifest['version'] or page_assets.get('assets')!=manifest['assets']:
                raise ValueError('Browser-loaded release differs from local reference; parity refused')
            report['page_asset_version']=page_assets['version']
        state=js('__gomokuSnapshot()');assert state['seconds']==1 and state['n']==15 and state['cols']==15 and len(state['board'])==225 and not state['history']
        check('default_15_and_1_second',True)
        # Actual independent browser Worker execution; no server inference exists.
        reference=json.loads((ROOT/'exports/browser/reference.json').read_text(encoding='utf-8'))
        for case in reference:
            request=dict(type='analyze',id=123,n=case['n'],cols=case.get('cols',case['n']),side=case['side'],board=case['board'],seconds=.1)
            code="""new Promise((resolve,reject)=>{const w=new Worker('./engine-worker.mjs',{type:'module'});w.onerror=e=>{w.terminate();reject(Error(e.message))};w.onmessage=({data})=>{if(data.type==='ready')w.postMessage(REQUEST);else if(data.type==='result'){w.terminate();resolve({...data,opponent:Array.from(data.opponent),play:Array.from(data.play),global:Array.from(data.global),combined:Array.from(data.combined),coverage:Array.from(data.coverage)});}else if(data.type==='error'){w.terminate();reject(Error(data.error));}};})""".replace('REQUEST',json.dumps(request))
            actual=js(code);errors={}
            for key,target in [('opponent','opponent'),('play','play'),('global','global_policy'),('combined','combined')]:
                errors[key]=float(np.max(np.abs(np.asarray(actual[key])-case[target])))
                np.testing.assert_allclose(actual[key],case[target],atol=2e-5,rtol=2e-4)
            assert actual['coverage']==case['coverage'] and actual['windowCount']==case['window_count']
            assert abs(actual['value']-case['value'])<3e-5
            assert case['board'][actual['search']['move']]==0
            check('browser_python_parity_'+case['name'],errors)
        for rows,cols in ((16,16),(15,15),(15,17),(16,15),(5,32),(32,5),(5,5)):
            shape(rows,cols)
            details=geometry(rows,cols);assert details['stones']==0
            check(f'desktop_stars_and_geometry_{rows}x{cols}',details)
            if (rows,cols) in ((15,17),(5,32),(32,5)):screenshot(f'desktop-{rows}x{cols}.png')
        shape(16,16)
        click('#edit');click('[data-point="51"]')
        assert js('__gomokuSnapshot().board[51]')==3
        assert js('document.querySelectorAll(".star-point").length')==3
        assert js('document.querySelectorAll(".forbidden .star-point").length')==0
        click('[data-point="51"]');click('#edit')
        assert js('document.querySelectorAll(".star-point").length')==4
        check('forbidden_star_hidden_and_restored',True)
        click('#edit');click('[data-point="0"]');assert js('__gomokuSnapshot().board[0]')==3
        click('#start');wait('!__gomokuSnapshot().busy');click('[data-point="0"]')
        assert '禁下' in js("document.getElementById('message').textContent")
        check('forbidden_cell_feedback',True)
        click('[data-point="136"]');wait('__gomokuSnapshot().history.length===2&&!__gomokuSnapshot().busy')
        first=js('__gomokuSnapshot()');assert first['board'][136]==1 and len(first['history'])==2
        check('real_desktop_move_ai_reply',{'plies':len(first['history']),'elapsed_ms':first['analysis']['elapsedMs']})
        screenshot('desktop.png')
        cdp('Emulation.setDeviceMetricsOverride',{'width':390,'height':844,'deviceScaleFactor':2,'mobile':True})
        cdp('Emulation.setTouchEmulationEnabled',{'enabled':True,'maxTouchPoints':5})
        click('[data-time="0.5"]',True);assert js('__gomokuSnapshot().seconds')==.5
        click('#zoom',True);assert js("document.getElementById('board').scrollWidth>document.getElementById('board-scroll').clientWidth")
        click('#zoom',True);assert js('document.documentElement.scrollWidth<=window.innerWidth')
        screenshot('mobile.png');check('mobile_touch_zoom_and_time',True)
        # Local mode stops only its own host. External mode changes this test
        # browser's network state; it never controls the deployed site.
        if server is not None:
            server.terminate();server.wait(timeout=10)
            report['local_static_server_stopped_for_offline']=True
        cdp('Network.emulateNetworkConditions',{'offline':True,'latency':0,'downloadThroughput':0,'uploadThroughput':0})
        cdp('Page.reload');wait('window.__gomokuSnapshot?.().ready&&!__gomokuSnapshot().busy',60)
        assert js('location.href')==url
        restored=js('__gomokuSnapshot()');assert restored['board']==first['board'] and restored['history']==first['history'] and restored['seconds']==.5
        restored_check=('offline_reload_network_only_saved_game_restored' if args.url
                        else 'offline_reload_server_stopped_saved_game_restored')
        check(restored_check,{'plies':len(restored['history']),'seconds':restored['seconds'],
            'static_server_stopped':report['local_static_server_stopped_for_offline'],'browser_network_offline':True})
        move=next(i for i in (119,120,135,137,151) if restored['board'][i]==0)
        click(f'[data-point="{move}"]',True);wait('__gomokuSnapshot().history.length===4&&!__gomokuSnapshot().busy',60)
        offline=js('__gomokuSnapshot()');assert len(offline['history'])==4
        assert not js('window.__errors.length')
        assert all(x['trusted'] for x in js('window.__trusted'))
        screenshot('offline-mobile.png');check('offline_touch_move_and_ai_reply',{'plies':4,'errors':js('window.__errors'),'trusted_clicks':js('window.__trusted')})
        (out/'offline_state.json').write_text(json.dumps(offline,ensure_ascii=False),encoding='utf-8')
        for rows,cols in ((15,17),(5,19),(19,5)):
            click('#new',True);shape(rows,cols,True)
            check(f'offline_mobile_stars_and_geometry_{rows}x{cols}',geometry(rows,cols))
            click('#start',True);wait('!__gomokuSnapshot().busy')
            move=rows//2*cols+cols//2
            click(f'[data-point="{move}"]',True)
            wait('__gomokuSnapshot().history.length===2&&!__gomokuSnapshot().busy',60)
            before=js('__gomokuSnapshot()')
            assert before['board'][move]==1 and len(before['analysis']['combined'])==rows*cols
            assert js('document.querySelectorAll("#board .stone").length')==2
            if rows%2 and cols%2:
                assert js('document.querySelectorAll("[data-star=tianyuan] .stone.black").length')==1
            cdp('Page.reload');wait('window.__gomokuSnapshot?.().ready&&!__gomokuSnapshot().busy',60)
            after=js('__gomokuSnapshot()')
            assert (after['n'],after['cols'])==(rows,cols)
            assert after['board']==before['board'] and after['history']==before['history']
            assert (int(js('document.getElementById("size").value')),int(js('document.getElementById("cols").value')))==(rows,cols)
            assert not js('window.__errors.length') and all(x['trusted'] for x in js('window.__trusted'))
            check(f'offline_mobile_rectangular_play_and_restore_{rows}x{cols}',{'plies':2,'move':move,'cells':rows*cols,'elapsed_ms':after['analysis']['elapsedMs']})
            screenshot(f'offline-mobile-{rows}x{cols}.png')
        report['passed']=True
    except BaseException as exc:
        report['error']=str(exc)
        if bridge:
            try:screenshot('failure.png')
            except Exception:pass
        raise
    finally:
        report['functional_passed']=report['passed']
        cleanup_errors=[]
        if bridge:
            try:bridge.request({'method':'Browser.close','params':{}},timeout=5)
            except Exception:pass
            bridge.close()
        if os.name=='nt':
            # Edge can leave its crash handler holding a profile open. Only
            # processes carrying this run's unique absolute profile qualify.
            assert profile.resolve().is_relative_to((ROOT/'exports/browser').resolve())
            command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'msedge.exe' -and $_.CommandLine -and $_.CommandLine.Contains($env:MUST5_TEST_PROFILE) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
            try:
                subprocess.run(['powershell','-NoProfile','-Command',command],
                    env={**os.environ,'MUST5_TEST_PROFILE':str(profile.resolve())},
                    stdout=log,stderr=log,creationflags=flags,timeout=20,check=True)
            except Exception as exc:cleanup_errors.append(str(exc))
        for process in (browser,server):
            if process is None:continue
            try:
                if process.poll() is None:process.terminate()
                process.wait(timeout=5)
            except Exception as exc:cleanup_errors.append(str(exc))
        log.close()
        report['cleanup_errors']=cleanup_errors
        report['cleanup_complete']=not cleanup_errors
        report['passed']=report['functional_passed'] and report['cleanup_complete']
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (ROOT/'exports/browser/latest_qa.json').write_text(json.dumps({'directory':str(out),'passed':report['passed']}),encoding='utf-8')
        if cleanup_errors:raise RuntimeError('Browser QA cleanup failed: '+str(cleanup_errors))

if __name__=='__main__':main()
