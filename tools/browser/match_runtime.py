"""Read-only instrumentation and trusted input for a durable dedicated browser."""
import base64
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

from web_match_runner import JsonProcess, atomic_json, file_hash, canonical_json, utc

FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0

# Observe worker traffic and DOM state; never alter a board, inference request,
# result, or the app's storage. Audit queues are the only mutable instrumentation.
OBSERVE = r'''
(()=>{
 const audit={page_id:crypto.randomUUID(),seq:0,events:[]};
 const clone=x=>JSON.parse(JSON.stringify(x,(_,v)=>ArrayBuffer.isView(v)?Array.from(v):v));
 const emit=(kind,data)=>audit.events.push({seq:++audit.seq,at:performance.now(),kind,data:clone(data)});
 const OriginalWorker=window.Worker;
 window.Worker=class extends OriginalWorker{
  constructor(...args){super(...args);const requests=new Map();const send=this.postMessage.bind(this);
   this.postMessage=(data,...rest)=>{if(data?.type==='analyze'){requests.set(data.id,clone(data));emit('analysis_requested',data);}return send(data,...rest);};
   this.addEventListener('message',e=>{if(e.data?.type==='result')emit('analysis_result',{request:requests.get(e.data.id),result:e.data});else if(e.data?.type==='error')emit('worker_error',e.data);});
  }
 };
 for(const type of ['click','keydown'])document.addEventListener(type,e=>emit('input',{
  type,trusted:e.isTrusted,key:e.key||null,point:e.target.closest?.('[data-point]')?.dataset.point??null,
  target:e.target.id||e.target.tagName,x:e.clientX??null,y:e.clientY??null,
  history:window.__gomokuSnapshot?.().history??null}),true);
 window.addEventListener('error',e=>emit('error',{message:e.message}));
 window.addEventListener('unhandledrejection',e=>emit('error',{message:String(e.reason)}));
 let last='';
 const observer=new MutationObserver(()=>{
  if(!window.__gomokuSnapshot)return;const state=window.__gomokuSnapshot();
  const key=JSON.stringify([state.n,state.cols??state.n,state.human,state.seconds,state.started,state.history]);
  if(last===key)return;last=key;
  const cells=[...document.querySelectorAll('#board .cell')].map(e=>({point:Number(e.dataset.point),
   cell:e.classList.contains('forbidden')?3:e.querySelector('.stone.black')?1:e.querySelector('.stone.white')?2:0}));
  emit('position',{state,visible_cells:cells,saved:localStorage.getItem('must5.browser.v1')});
 });
 observer.observe(document,{subtree:true,childList:true,attributes:true,attributeFilter:['class']});
 window.__getGomokuAudit=()=>({page_id:audit.page_id,events:clone(audit.events)});
 window.__ackGomokuAudit=seq=>{audit.events=audit.events.filter(e=>e.seq>seq);};
})();
'''


def listening(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(('127.0.0.1',port)) == 0


def process_alive(pid):
    if os.name == 'nt':
        import ctypes
        kernel=ctypes.windll.kernel32
        kernel.OpenProcess.restype=ctypes.c_void_p
        handle=kernel.OpenProcess(0x100000,False,int(pid))
        if not handle:return False
        try:return kernel.WaitForSingleObject(ctypes.c_void_p(handle),0)==0x102
        finally:kernel.CloseHandle(ctypes.c_void_p(handle))
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


class BrowserRuntime:
    def __init__(self,run,manifest,journal):
        self.run,self.manifest,self.journal=Path(run),manifest,journal
        self.source=self.run/'frozen/source'
        self.web=self.run/'frozen/web'
        self.runtime=self.run/'runtime';self.runtime.mkdir(exist_ok=True)
        self.logs=[];self.bridge=None
        self._server()
        self._browser()

    def _server(self):
        record=self.runtime/'server.json'
        if record.exists():
            entry=json.loads(record.read_text(encoding='utf-8'))
            port=entry['port']
        else:
            with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
            entry={'port':port,'run_id':self.manifest['run_id'],'pid':None}
            atomic_json(record,entry)
        if entry['run_id']!=self.manifest['run_id']:raise RuntimeError('Static server identity mismatch')
        if not listening(port):
            if entry['pid'] and process_alive(entry['pid']):
                raise RuntimeError('Recorded static server is alive but unresponsive; preserve it and retry observation')
            log=(self.runtime/'server.log').open('ab');self.logs.append(log)
            args=[sys.executable,'-X','utf8','-B','-m','http.server',str(port),'--bind','127.0.0.1','--directory',str(self.web)]
            process=subprocess.Popen(args,cwd=self.source,stdout=log,stderr=log,creationflags=FLAGS)
            entry.update(pid=process.pid,argv=args);atomic_json(record,entry)
            self.journal.append('static_server_started',entry)
            deadline=time.monotonic()+20
            while not listening(port):
                if process.poll() is not None:raise RuntimeError('Static server exited')
                if time.monotonic()>deadline:raise TimeoutError('Static server startup')
                time.sleep(.1)
        self.origin=f'http://127.0.0.1:{port}'
        payload=urllib.request.urlopen(self.origin+'/assets.json',timeout=10).read()
        import hashlib
        if hashlib.sha256(payload).hexdigest()!=self.manifest['web_hashes']['assets.json']:
            raise RuntimeError('The occupied port does not serve this frozen candidate')

    def _browser(self):
        record=self.runtime/'browser.json';profile=self.runtime/'profile'
        active=profile/'DevToolsActivePort';entry=json.loads(record.read_text()) if record.exists() else None
        debug=None
        if active.exists():
            candidate=int(active.read_text().splitlines()[0])
            if listening(candidate):debug=candidate
        if debug is None:
            if entry and process_alive(entry['pid']):
                raise RuntimeError('Recorded browser is alive without a reachable debugger; do not restart its game')
            profile.mkdir(exist_ok=True)
            log=(self.runtime/'browser.log').open('ab');self.logs.append(log)
            args=[self.manifest['browser'],'--no-first-run','--no-default-browser-check',
                  '--disable-crash-reporter','--disable-breakpad','--remote-debugging-port=0',
                  '--remote-debugging-address=127.0.0.1',f'--user-data-dir={profile}',
                  '--window-size=1400,1200','about:blank']
            if not self.manifest.get('headed'):args.insert(1,'--headless=new')
            if active.exists():
                # The old endpoint is confirmed gone; retain its record.
                previous=self.runtime/('DevToolsActivePort.previous.'+str(time.time_ns()))
                shutil.copyfile(active,previous)
                active.unlink()
            browser=subprocess.Popen(args,stdout=log,stderr=log,creationflags=FLAGS)
            entry={'pid':browser.pid,'profile':str(profile),'run_id':self.manifest['run_id'],'argv':args}
            atomic_json(record,entry);self.journal.append('browser_started',entry)
            deadline=time.monotonic()+30
            while not active.exists():
                if browser.poll() is not None:raise RuntimeError('Dedicated browser exited; preserve profile')
                if time.monotonic()>deadline:raise TimeoutError('Dedicated browser startup')
                time.sleep(.1)
            debug=int(active.read_text().splitlines()[0])
        if entry is None or entry['run_id']!=self.manifest['run_id']:
            raise RuntimeError('Unrecognized browser profile; will not attach')
        pages=json.load(urllib.request.urlopen(f'http://127.0.0.1:{debug}/json/list',timeout=10))
        pages=[p for p in pages if p['type']=='page']
        page=next((p for p in pages if p['url'].startswith(self.origin+'/')),None)
        if page is None:page=next((p for p in pages if p['url']=='about:blank'),None)
        if page is None:raise RuntimeError('The persisted browser game tab is missing; no replacement game was started')
        self.bridge=JsonProcess([self.manifest['node'],str(self.source/'web_match_cdp.mjs'),
                    page['webSocketDebuggerUrl'],self.origin],self.source,self.runtime/'cdp.log')
        self.cdp('Page.enable');self.cdp('Runtime.enable');self.cdp('Network.enable')
        identity=self.cdp('Browser.getVersion')
        previous=[e['data'] for e in self.journal.events if e['kind']=='browser_identity']
        if previous and identity!=previous[0]:raise RuntimeError('Browser runtime identity changed')
        if not previous:self.journal.append('browser_identity',identity)
        self.cdp('Page.addScriptToEvaluateOnNewDocument',{'source':OBSERVE})
        if not page['url'].startswith(self.origin+'/'):
            self.cdp('Page.navigate',{'url':self.origin+'/'})
        elif not self.js("typeof window.__getGomokuAudit==='function'"):
            raise RuntimeError('Existing game has no original audit observer; preserve it without changing the page')
        self.wait("window.__gomokuSnapshot?.().ready",timeout=60)
        self._verify_assets()

    def cdp(self,method,params=None,timeout=60):
        return self.bridge.request({'method':method,'params':params or {}},timeout=timeout)

    def js(self,expression):
        result=self.cdp('Runtime.evaluate',{'expression':expression,'awaitPromise':True,'returnByValue':True})
        if 'exceptionDetails' in result:raise RuntimeError(str(result['exceptionDetails']))
        return result.get('result',{}).get('value')

    def _verify_assets(self):
        names=list(self.manifest['web_hashes'])
        expression=r'''(async()=>{const result={};for(const name of NAMES){
          const response=await fetch('./'+name,{cache:'no-store'});if(!response.ok)throw Error(name+': '+response.status);
          const bytes=await response.arrayBuffer();const hash=await crypto.subtle.digest('SHA-256',bytes);
          result[name]=[...new Uint8Array(hash)].map(b=>b.toString(16).padStart(2,'0')).join('');}return result;})()'''.replace('NAMES',canonical_json(names))
        actual=self.js(expression)
        if actual!=self.manifest['web_hashes']:raise RuntimeError('Browser-fetched assets differ from frozen candidate')
        self.journal.append('browser_assets_verified',{'hashes':actual,'origin':self.origin})

    def wait(self,expression,timeout=45):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if self.js(expression):return
            time.sleep(.1)
        raise TimeoutError(expression+'; '+str(self.js("document.getElementById('message')?.textContent")))

    def snapshot(self):
        return self.js("JSON.parse(JSON.stringify(__gomokuSnapshot(),(_,v)=>ArrayBuffer.isView(v)?Array.from(v):v))")

    def idle(self):
        self.wait("window.__gomokuSnapshot?.().ready&&!__gomokuSnapshot().busy",timeout=60)
        return self.snapshot()

    def audit(self):return self.js('__getGomokuAudit()')

    def ack(self,seq):self.js('__ackGomokuAudit('+str(int(seq))+')')

    def saved(self):return self.js("JSON.parse(localStorage.getItem('must5.browser.v1'))")

    def click(self,selector):
        rectangle=self.js("(()=>{const e=document.querySelector(SELECTOR);if(!e||e.disabled)throw Error('Unavailable control');e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()".replace('SELECTOR',canonical_json(selector)))
        for kind in ('mousePressed','mouseReleased'):
            self.cdp('Input.dispatchMouseEvent',dict(type=kind,button='left',clickCount=1,**rectangle))
        return rectangle

    def key(self,key,code=None,vk=None):
        parameters={'key':key,'code':code or key}
        if vk is not None:parameters.update(windowsVirtualKeyCode=vk,nativeVirtualKeyCode=vk)
        self.cdp('Input.dispatchKeyEvent',dict(type='keyDown',**parameters))
        self.cdp('Input.dispatchKeyEvent',dict(type='keyUp',**parameters))

    def configure_human(self,side):
        self.click('#human');self.key('Home','Home',36)
        if side==2:self.key('ArrowDown','ArrowDown',40)
        self.key('Enter','Enter',13)
        if self.js("Number(document.getElementById('human').value)")!=side:
            raise RuntimeError('Trusted color selection did not select the requested side')

    def screenshot(self,path):
        data=self.cdp('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True})['data']
        Path(path).write_bytes(base64.b64decode(data))
        return file_hash(path)

    def close_bridge(self):
        # Keep the durable game tab/profile and static origin alive for resume.
        if self.bridge:self.bridge.close();self.bridge=None
        for stream in self.logs:stream.close()
