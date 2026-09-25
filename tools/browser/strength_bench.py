"""Paired, equal-time development games between two frozen browser engines.

Runs the real ONNX/WASM Workers in Chromium. Python independently validates
each move and terminal. These engine matches are not formal UI acceptance or
an estimate of performance against human players. Existing runs are preserved.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from board_rules import board_winner
from evaluate_unet_board import generate_paired_openings, _start_board
from web_match_runner import JsonProcess, locate_browser


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def freeze(source, target):
    source = Path(source).resolve()
    manifest = read(source / 'assets.json')
    for name, record in manifest['assets'].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source):
            raise ValueError('Asset escapes source: ' + name)
        data = path.read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Rebuild changed assets before benchmarking: ' + name)
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    (target / 'assets.json').write_bytes((source / 'assets.json').read_bytes())
    return manifest


HARNESS = '''
window.engines = new Map();
window.nextRequest = 0;
window.openEngine = (name) => new Promise((resolve, reject) => {
  const worker = new Worker(`./${name}/engine-worker.mjs`, {type: 'module'});
  const pending = new Map();
  worker.onerror = error => {
    reject(Error(error.message));
    for (const item of pending.values()) item.reject(Error(error.message));
    pending.clear();
  };
  worker.onmessage = ({data}) => {
    if (data.type === 'ready') { engines.set(name, {worker, pending}); resolve(true); }
    else if (data.type === 'error') {
      const item = pending.get(data.id);
      if (item) { pending.delete(data.id); item.reject(Error(data.error)); }
      else reject(Error(data.error));
    } else if (data.type === 'result') {
      const item = pending.get(data.id);
      pending.delete(data.id);
      if (!item) throw Error('Unmatched engine response');
      const top = policy => policy ? Array.from(policy, (p, move) => ({move, p}))
        .sort((a, b) => b.p - a.p).slice(0, 5) : [];
      item.resolve({search: data.search, elapsedMs: data.elapsedMs,
        inferenceMs: data.inferenceMs, terminal: data.terminal,
        globalTop: top(data.global), combinedTop: top(data.combined)});
    }
  };
});
window.analyzeEngine = (name, request) => new Promise((resolve, reject) => {
  const {worker, pending} = engines.get(name), id = ++nextRequest;
  pending.set(id, {resolve, reject});
  worker.postMessage({...request, type: 'analyze', id});
});
'''


def diagnostic_positions():
    fixtures = read(ROOT / 'tests/browser/fixtures/native_threat_regressions.json')
    for case in fixtures['positive_cases']:
        source = fixtures['sources'][case['source']]
        board = np.zeros((source['n'], source['n']), dtype=np.uint8)
        for i, move in enumerate(source['history'][:case['prefix_plies']]):
            board[tuple(move)] = 1 + i % 2
        yield case['name'], board, case['side'], []
    for path in sorted((ROOT / 'reproduction/inputs/constraints').glob('*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            case = json.loads(line)
            board = np.array(case['board'], dtype=np.uint8)
            bad = [item['move'][0] * board.shape[1] + item['move'][1]
                   for item in case['action_evidence'] if item['actor_value'] == -1]
            yield path.stem + '_' + str(case['prefix_plies']), board, case['side'], bad


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, default=ROOT / 'web/browser')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=6)
    parser.add_argument('--seconds', type=float, default=1)
    parser.add_argument('--size', type=int, default=15)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--skip-probes', action='store_true')
    args = parser.parse_args(argv)
    if args.pairs < 0 or not 0.1 <= args.seconds <= 30 or not 6 <= args.size <= 32:
        parser.error('Invalid pair count, time budget or board size')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    identities = {name: freeze(path, output / name) for name, path in
                  [('baseline', args.baseline), ('candidate', args.candidate)]}
    openings = generate_paired_openings(args.size, args.size, args.pairs, args.seed)
    (output / 'index.html').write_text('<!doctype html><title>Gomoku engine benchmark</title>', encoding='utf-8')
    report = dict(format='must5_browser_strength_development_v1',
                  created_utc=datetime.now(timezone.utc).isoformat(), seconds=args.seconds,
                  size=args.size, seed=args.seed, openings=openings, games=[], probes=[],
                  engines={k: v['version'] for k, v in identities.items()},
                  formal_acceptance=False, completed=False)
    (output / 'configuration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    server = browser = bridge = None
    log = (output / 'runtime.log').open('ab')
    journal = (output / 'moves.jsonl').open('w', encoding='utf-8')
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        origin = f'http://127.0.0.1:{port}'
        server = subprocess.Popen([sys.executable, '-m', 'http.server', str(port), '--bind',
                                   '127.0.0.1', '--directory', str(output)],
                                  stdout=log, stderr=log, creationflags=flags)
        profile = output / 'profile'
        browser = subprocess.Popen([locate_browser(), '--headless=new', '--no-first-run',
            '--no-default-browser-check', '--disable-crash-reporter', '--disable-breakpad',
            '--remote-debugging-port=0', '--remote-debugging-address=127.0.0.1',
            '--user-data-dir=' + str(profile), 'about:blank'],
            stdout=log, stderr=log, creationflags=flags)
        active = profile / 'DevToolsActivePort'
        deadline = time.monotonic() + 30
        while not active.exists():
            if time.monotonic() >= deadline or browser.poll() is not None:
                raise TimeoutError('Benchmark browser startup')
            time.sleep(.1)
        debug = int(active.read_text().splitlines()[0])
        pages = json.load(urllib.request.urlopen(f'http://127.0.0.1:{debug}/json/list', timeout=10))
        page = next(page for page in pages if page['type'] == 'page')
        bridge = JsonProcess([shutil.which('node'), str(ROOT / 'web_match_cdp.mjs'),
                              page['webSocketDebuggerUrl'], origin], ROOT, output / 'cdp.log')

        def cdp(method, params=None):
            return bridge.request({'method': method, 'params': params or {}}, timeout=90)

        def js(expression):
            response = cdp('Runtime.evaluate', {'expression': expression,
                'returnByValue': True, 'awaitPromise': True})
            if 'exceptionDetails' in response:
                raise RuntimeError(str(response['exceptionDetails']))
            return response.get('result', {}).get('value')

        cdp('Page.navigate', {'url': origin + '/'})
        deadline = time.monotonic() + 20
        while js('location.href') != origin + '/' or js('document.readyState') != 'complete':
            if time.monotonic() >= deadline:
                raise TimeoutError('Benchmark page startup')
            time.sleep(.1)
        js(HARNESS)
        for name in identities:
            js('openEngine(' + json.dumps(name) + ')')

        def analyze(name, board, side):
            request = dict(n=board.shape[0], cols=board.shape[1], board=board.ravel().tolist(),
                           side=side, seconds=args.seconds)
            result = js('analyzeEngine(' + json.dumps(name) + ',' + json.dumps(request) + ')')
            move = result['search']['move']
            if type(move) is not int or not 0 <= move < board.size or board.ravel()[move] != 0:
                raise ValueError('Engine returned an illegal move')
            return result

        if not args.skip_probes:
            for name, board, side, bad in diagnostic_positions():
                record = dict(name=name, board=board.tolist(), side=side, known_losing_moves=bad)
                for engine in identities:
                    record[engine] = analyze(engine, board, side)
                report['probes'].append(record)
                print(json.dumps({'probe': name, **{engine: {key: record[engine]['search'].get(key)
                    for key in ('move', 'depth', 'nodes', 'value', 'selectedStatus')} for engine in identities}}, ensure_ascii=False), flush=True)
        for pair, prefix in enumerate(openings):
            for candidate_side in (1, 2):
                board, side, history = _start_board(args.size, args.size, prefix)
                game = dict(pair=pair, candidate_side=candidate_side, opening=prefix,
                            history=history, moves=[])
                report['games'].append(game)
                while not board_winner(board) and np.any(board == 0):
                    engine = 'candidate' if side == candidate_side else 'baseline'
                    result = analyze(engine, board, side)
                    point = result['search']['move']
                    row, col = divmod(point, args.size)
                    board[row, col] = side
                    move = dict(side=side, move=[row, col], engine=engine, **result)
                    game['moves'].append(move)
                    history.append(dict(side=side, move=[row, col], opening=False))
                    journal.write(json.dumps(dict(pair=pair, candidate_side=candidate_side,
                                                 ply=len(history), **move), ensure_ascii=False) + '\n')
                    journal.flush()
                    side = 3 - side
                winner = int(board_winner(board))
                game.update(winner=winner, plies=len(history), board=board.tolist(),
                            outcome='draw' if winner == 0 else 'win' if winner == candidate_side else 'loss')
                print(json.dumps({k: game[k] for k in ('pair', 'candidate_side', 'winner', 'outcome', 'plies')}), flush=True)
                (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        report['completed'] = True
    except BaseException as exc:
        report['error'] = str(exc)
        raise
    finally:
        if bridge:
            try:
                bridge.request({'method': 'Browser.close', 'params': {}}, timeout=5)
            except Exception:
                pass
            bridge.close()
        for process in (browser, server):
            if process is not None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)
        log.close()
        journal.close()
        report['outcomes'] = {outcome: sum(g.get('outcome') == outcome for g in report['games'])
                              for outcome in ('win', 'draw', 'loss')}
        report['unfinished_games'] = sum('outcome' not in g for g in report['games'])
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'completed': report['completed'], 'outcomes': report['outcomes'],
                          'unfinished_games': report['unfinished_games']}), flush=True)


if __name__ == '__main__':
    main()
