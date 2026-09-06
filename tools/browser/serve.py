"""Serve only static browser assets; AI computation stays in each browser."""
import argparse
import json
import webbrowser
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8770)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--open',action='store_true',help='Open the page in your default browser')
    args=parser.parse_args()
    directory=Path(__file__).resolve().parents[2]/'web/browser'
    if not (directory/'assets.json').is_file():
        parser.error('Browser files are missing. Run npm ci and npm run build:browser first.')
    try:
        assets=json.loads((directory/'assets.json').read_text(encoding='utf-8'))['assets']
        if not isinstance(assets,dict) or not assets:
            raise ValueError('empty assets')
        missing=[]
        for name in [*assets,'sw.js']:
            path=(directory/name).resolve()
            if not path.is_relative_to(directory.resolve()) or not path.is_file():
                missing.append(name)
        if missing:
            parser.error('Missing browser files: '+', '.join(missing)+'. Run npm run build:browser to restore them.')
    except (OSError,ValueError,KeyError,TypeError) as exc:
        parser.error('Invalid browser assets manifest: '+str(exc))
    handler=partial(SimpleHTTPRequestHandler,directory=str(directory))
    with ThreadingHTTPServer((args.host,args.port),handler) as server:
        url=f'http://{args.host}:{server.server_port}/'
        print(f'Browser-only AI: {url}',flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__=='__main__':main()
