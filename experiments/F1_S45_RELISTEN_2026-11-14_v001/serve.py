"""Read-only local serving with the established WAV seeking handler."""
import functools
from http.server import ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('s45_range_handler',here.parents[1]/'experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/serve.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(module.Handler,directory=str(here)))
    with (here/'server_ready.json').open('x',encoding='utf-8') as f:
        json.dump({'pid':os.getpid(),'port':server.server_address[1],
                   'url':f'http://127.0.0.1:{server.server_address[1]}/delivery_v001/index.html'},f)
    server.serve_forever()
