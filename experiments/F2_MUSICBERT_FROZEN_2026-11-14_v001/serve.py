"""Local-only server, reusing the existing tested WAV byte-range handler."""
import functools
from http.server import ThreadingHTTPServer
import json
import os
from adapter import HERE,PROJECT,_reuse

if __name__=='__main__':
    handler=_reuse(PROJECT/'experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/serve.py','frozen_range_server').Handler
    server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(handler,directory=str(HERE)))
    ready=HERE/'server_ready.json'
    with ready.open('x',encoding='utf-8') as f:
        json.dump({'pid':os.getpid(),'port':server.server_address[1],
                   'url':f'http://127.0.0.1:{server.server_address[1]}/delivery_v001/index.html'},f)
    server.serve_forever()
