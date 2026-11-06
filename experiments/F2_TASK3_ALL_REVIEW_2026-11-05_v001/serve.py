from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
import argparse, mimetypes, re, webbrowser

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PAGE = '/experiments/F2_TASK3_ALL_REVIEW_2026-11-05_v001/index.html'
ALLOWED = {PAGE: HERE / 'index.html',
 '/test_dataset/my_singing/rerecord_2026-11-04_f2/ref_vocals/心墙_郭静_vocals.wav': ROOT / 'test_dataset/my_singing/rerecord_2026-11-04_f2/ref_vocals/心墙_郭静_vocals.wav',
 '/test_dataset/my_singing/rerecord_2026-11-04_f2/ref_vocals/离开我的依赖_王艳薇_vocals.wav': ROOT / 'test_dataset/my_singing/rerecord_2026-11-04_f2/ref_vocals/离开我的依赖_王艳薇_vocals.wav',
 '/test_dataset/my_singing/rerecord_2026-11-04_f2/ref_vocals/莫问归期_蒋雪儿_vocals.wav': ROOT / 'test_dataset/my_singing/rerecord_2026-11-04_f2/ref_vocals/莫问归期_蒋雪儿_vocals.wav',
 '/test_dataset/my_singing/rerecord_2026-11-04_f2/wav/心墙01-轻度跑调.wav': ROOT / 'test_dataset/my_singing/rerecord_2026-11-04_f2/wav/心墙01-轻度跑调.wav',
 '/test_dataset/my_singing/rerecord_2026-11-04_f2/wav/莫问归期01-轻度跑调.wav': ROOT / 'test_dataset/my_singing/rerecord_2026-11-04_f2/wav/莫问归期01-轻度跑调.wav',
}

class Handler(BaseHTTPRequestHandler):
 def do_HEAD(self): self.send_file(False)
 def do_GET(self): self.send_file(True)
 def send_file(self, body):
  path=unquote(urlsplit(self.path).path)
  if path == '/favicon.ico':
   self.send_response(204); self.end_headers(); return
  target=ALLOWED.get(path)
  if target is None or not target.is_file(): self.send_error(404); return
  size=target.stat().st_size; start=0; end=size-1; partial=False
  header=self.headers.get('Range')
  if header:
   m=re.fullmatch(r'bytes=(\d*)-(\d*)',header.strip())
   if not m or (not m.group(1) and not m.group(2)):
    self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return
   if not m.group(1): start=max(0,size-int(m.group(2)))
   else: start=int(m.group(1)); end=min(size-1,int(m.group(2))) if m.group(2) else size-1
   if start>=size or start>end:
    self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return
   partial=True
  self.send_response(206 if partial else 200)
  self.send_header('Content-Type',mimetypes.guess_type(str(target))[0] or 'application/octet-stream')
  self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
  self.send_header('Cache-Control','no-store')
  if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
  self.end_headers()
  if body:
   try:
    with target.open('rb') as f:
     f.seek(start); remain=end-start+1
     while remain:
      chunk=f.read(min(remain,65536))
      if not chunk:break
      self.wfile.write(chunk);remain-=len(chunk)
   except (BrokenPipeError,ConnectionResetError): pass

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--open',action='store_true');args=parser.parse_args()
 url='http://127.0.0.1:8769'+PAGE
 try: server=ThreadingHTTPServer(('127.0.0.1',8769),Handler)
 except OSError:
  print('8769 端口已被使用。若页面未打开，请关闭占用该端口的程序后重试。')
  if args.open: webbrowser.open(url)
  return
 print('核对页：'+url,flush=True)
 if args.open:webbrowser.open(url)
 server.serve_forever()
if __name__=='__main__':main()
