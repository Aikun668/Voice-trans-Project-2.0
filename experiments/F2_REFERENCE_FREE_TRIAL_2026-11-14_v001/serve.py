"""Local-only listening page server with byte-range support for WAV seeking."""
import argparse
import functools
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import json
import os
from pathlib import Path
import re


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        if not self.headers.get('Range'):return super().send_head()
        path=self.translate_path(self.path)
        if not os.path.isfile(path):return super().send_head()
        f=open(path,'rb');size=os.fstat(f.fileno()).st_size
        match=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers['Range'])
        if not match:f.close();self.send_error(416);return None
        start=int(match[1]);end=min(int(match[2]) if match[2] else size-1,size-1)
        if start> end or start>=size:f.close();self.send_error(416);return None
        self.send_response(206);self.send_header('Content-Type',self.guess_type(path))
        self.send_header('Content-Range',f'bytes {start}-{end}/{size}');self.send_header('Accept-Ranges','bytes')
        self.send_header('Content-Length',str(end-start+1));self.end_headers()
        f.seek(start);self.remaining=end-start+1;return f

    def copyfile(self,source,outputfile):
        if not hasattr(self,'remaining'):return super().copyfile(source,outputfile)
        remaining=self.remaining
        while remaining:
            block=source.read(min(65536,remaining))
            if not block:break
            outputfile.write(block);remaining-=len(block)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--ready-file',type=Path,required=True);ap.add_argument('--port',type=int,default=0);args=ap.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),functools.partial(Handler,directory=str(args.directory.resolve())))
    args.ready_file.write_text(json.dumps({'pid':os.getpid(),'port':server.server_address[1],
                                         'url':f'http://127.0.0.1:{server.server_address[1]}/index.html'}),encoding='utf-8')
    server.serve_forever()
