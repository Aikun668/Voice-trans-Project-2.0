"""扩展原唱候选窗口检查零时长；只保存字序号和时间，不保存网页歌词。"""
from pathlib import Path
from importlib.metadata import version
import hashlib
import json
import math
import os
import re
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'results'
if OUT.exists(): raise SystemExit(f'拒绝覆盖已有目录: {OUT}')
OUT.mkdir()
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import requests
from bs4 import BeautifulSoup
import soundfile as sf
import torch
from qwen_asr import Qwen3ForcedAligner

base=ROOT/'test_dataset/my_singing/rerecord_2026-11-04_f2'
refdir=base/'ref_vocals';wavdir=base/'wav'
model_path=Path(r'D:\My_Grand_Projects2026\Videos-Master-Project\Qwen3-ForcedAligner-0.6B')
sources={'离开我的依赖':'https://www.kugou.com/song-36/143hgz45.html',
         '莫问归期':'https://chinesemusicwiki.com/dont-ask-about-the-return-date%E8%8E%AB%E9%97%AE%E5%BD%92%E6%9C%9Fmo-wen-gui-qi-by-snow-j%E8%92%8B%E9%9B%AA%E5%84%BF/'}
def fetch(url):
 r=requests.get(url,timeout=45,headers={'User-Agent':'Mozilla/5.0'});r.raise_for_status()
 return BeautifulSoup(r.text,'html.parser')
print('读取两首歌曲来源，歌词只在内存中使用',flush=True)
soup=fetch(sources['离开我的依赖']);node=soup.select_one('.songContent .displayNone')
if node is None:raise SystemExit('酷狗歌词节点不可用')
entries=re.findall(r'\[(\d{2}):(\d{2}\.\d{2})\]([^\[]+)',node.get_text(' ',strip=True))
def kugou_window(start,end):
 segments=[re.sub(r'\s+','',t) for m,s,t in entries if start<=int(m)*60+float(s)<end]
 if len(segments)!=3:raise SystemExit('歌词来源的段落数发生变化，拒绝使用')
 return ''.join(segments)
lyrics={'离开我的依赖01':kugou_window(55,68),'离开我的依赖02':kugou_window(69,80)}
paragraphs=fetch(sources['莫问归期']).select('.entry-content p')
def wiki_lines(indices,expected):
 lines=[]
 for index,n in zip(indices,expected):
  nodes=[str(t) for t in paragraphs[index].contents if getattr(t,'name',None) is None]
  chinese=[''.join(re.findall(r'[\u4e00-\u9fff]',s)) for s in nodes]
  chinese=[s for s in chinese if s]
  if len(chinese)!=1 or len(chinese[0])!=n:raise SystemExit('歌词来源结构发生变化，拒绝使用')
  lines.append(chinese[0])
 return ''.join(lines)
lyrics['莫问归期01']=wiki_lines((5,6,7),(6,6,10))
lyrics['莫问归期02']=wiki_lines((10,11,12),(8,7,9))
if [len(lyrics[x]) for x in lyrics] != [17,24,22,24]:raise SystemExit('歌词长度异常')
groups=[
 ('离开我的依赖01','离开我的依赖_王艳薇_vocals.wav',[(51.4,70.8)]),
 ('离开我的依赖02','离开我的依赖_王艳薇_vocals.wav',[(66.5,82.9)]),
 ('莫问归期01','莫问归期_蒋雪儿_vocals.wav',[(111.1,127.4)]),
 ('莫问归期02','莫问归期_蒋雪儿_vocals.wav',[(131.4,148.4)]),
]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
inputs=[]
for group,ref,windows in groups:
 for i,window in enumerate(windows):
  inputs.append((group,'扩大窗口'+('B' if group=='莫问归期01' else ''),refdir/ref,window))
hashes={str(p.relative_to(ROOT)):sha(p) for _,_,p,_ in inputs}
torch.set_num_threads(4)
device='cuda:0' if torch.cuda.is_available() else 'cpu'
dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32
print('加载 Qwen；设备:',device,flush=True)
model=Qwen3ForcedAligner.from_pretrained(str(model_path),dtype=dtype,device_map=device,local_files_only=True)
outputs=[]
for i,(group,label,p,window) in enumerate(inputs,1):
 print(f'{i}/{len(inputs)} {group} {label}',flush=True)
 info=sf.info(str(p)); start=round(window[0]*info.samplerate) if window else 0;end=round(window[1]*info.samplerate) if window else info.frames
 audio,sr=sf.read(str(p),start=start,stop=end,dtype='float32',always_2d=True);audio=audio.mean(axis=1);offset=start/sr
 t=time.monotonic();aligned=list(model.align(audio=(audio,sr),text=lyrics[group],language='Chinese')[0]); elapsed=time.monotonic()-t
 # 为防歌词被错误解析或模型分词不一致，比较内存中的文字；不将词语输出或保存。
 text_match=''.join(str(x.text) for x in aligned)==lyrics[group]
 tokens=[{'index':j,'start_s':round(offset+float(x.start_time),4),'end_s':round(offset+float(x.end_time),4)} for j,x in enumerate(aligned,1)]
 finite=all(math.isfinite(x['start_s']) and math.isfinite(x['end_s']) for x in tokens)
 inside=finite and all(offset<=x['start_s']<=x['end_s']<=offset+len(audio)/sr+.05 for x in tokens)
 ordered=finite and all(tokens[i]['start_s']>=tokens[i-1]['start_s'] and tokens[i]['end_s']>=tokens[i-1]['end_s'] for i in range(1,len(tokens)))
 outputs.append({'group':group,'label':label,'source':str(p.relative_to(ROOT)),'source_start_s':offset,'duration_s':len(audio)/sr,
  'source_sha256':hashes[str(p.relative_to(ROOT))],'tokens':tokens,'seconds_used':round(elapsed,2),
  'checks':{'token_text_matches_source':text_match,'finite':finite,'inside_audio':inside,'monotonic':ordered,'zero_duration':sum(x['start_s']==x['end_s'] for x in tokens)},
  'status':'仅模型候选，原唱位置和字音边界待抽查，不能直接用于训练'})
result={'sources':sources,'source_segments':{'离开我的依赖01':'酷狗时间 55–68 秒三行','离开我的依赖02':'酷狗时间 69–80 秒三行','莫问归期01':'来源歌词第1–3行','莫问归期02':'来源歌词第6–8行'},
 'text_sha256':{k:hashlib.sha256(v.encode('utf-8')).hexdigest() for k,v in lyrics.items()},
 'model':'Qwen3-ForcedAligner-0.6B','qwen_asr_version':version('qwen-asr'),'device':device,
 'inputs_unchanged':all(sha(ROOT/k)==v for k,v in hashes.items()),'audio_output_generated':False,
 'notes':['每个原唱窗口在原基础上首尾各扩 2 秒以测试边缘影响。','网页歌词仅用于本次内存计算，不保存歌词；只保存序号与时间。','真实演唱及字边界未人工验证。'],
 'items':outputs}
(OUT/'alignment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('完成：',len(outputs),'个输入，源文件未变:',result['inputs_unchanged'],flush=True)
for x in outputs: print(x['group'],x['label'],x['checks'],flush=True)
