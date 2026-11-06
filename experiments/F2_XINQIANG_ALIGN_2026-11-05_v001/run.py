"""Qwen forced alignment on the user-confirmed 心墙 01 phrase; no audio files written."""
from pathlib import Path
import hashlib
import json
import math
import re
import sys
import time
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'results'
if OUT.exists(): raise SystemExit(f'输出目录已存在，拒绝覆盖: {OUT}')
OUT.mkdir()
import numpy as np
import soundfile as sf
import torch
from qwen_asr import Qwen3ForcedAligner

BASE=ROOT/'test_dataset/my_singing/rerecord_2026-11-04_f2'
FEEDBACK=ROOT/'experiments/F2_TASK1_CATALOG_2026-11-05_v001/feedback.md'
MODEL=Path(r'D:\My_Grand_Projects2026\Videos-Master-Project\Qwen3-ForcedAligner-0.6B')
match=re.search(r'是否对应“([^”]+)”',FEEDBACK.read_text(encoding='utf-8'))
if not match: raise SystemExit('未找到 Jade 已确认的句子；拒绝使用未经核对的文本')
text=match.group(1)
cases=[
 ('本人对照微微跑调', BASE/'wav/心墙01-对照微微跑调.wav',None, '本人该句已确认'),
 ('本人轻度跑调', BASE/'wav/心墙01-轻度跑调.wav',None, '按用户命名同一段；具体文字待抽查'),
 ('原唱候选', BASE/'ref_vocals/心墙_郭静_vocals.wav',(112.8,121.1), '用户确认同一句；逐字待抽查'),
]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
hashes={str(p.relative_to(ROOT)):sha(p) for _,p,_,_ in cases}
torch.set_num_threads(4)
device='cuda:0' if torch.cuda.is_available() else 'cpu'
dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32
print('加载本地 Qwen 对齐模型，设备:',device,flush=True)
model=Qwen3ForcedAligner.from_pretrained(str(MODEL),dtype=dtype,device_map=device,local_files_only=True)
rows=[]
for label,p,window,input_status in cases:
 print('处理:',label,flush=True)
 info=sf.info(str(p)); begin=round(window[0]*info.samplerate) if window else 0; stop=round(window[1]*info.samplerate) if window else info.frames
 x,sr=sf.read(str(p),start=begin,stop=stop,dtype='float32',always_2d=True)
 audio=x.mean(axis=1);offset=begin/sr
 start=time.monotonic()
 result=model.align(audio=(audio,sr),text=text,language='Chinese')[0]
 units=[]
 for u in result:
  t0=float(u.start_time);t1=float(u.end_time)
  units.append({'text':str(u.text),'start_local_s':t0,'end_local_s':t1,'start_source_s':round(offset+t0,4),'end_source_s':round(offset+t1,4)})
 finite=all(math.isfinite(u['start_local_s']) and math.isfinite(u['end_local_s']) for u in units)
 in_bounds=finite and all(0<=u['start_local_s']<=u['end_local_s']<=len(audio)/sr+.05 for u in units)
 monotonic=finite and all(units[i]['start_local_s']>=units[i-1]['start_local_s'] and units[i]['end_local_s']>=units[i-1]['end_local_s'] for i in range(1,len(units)))
 rows.append({'label':label,'source':str(p.relative_to(ROOT)),'source_sha256':hashes[str(p.relative_to(ROOT))],
   'source_rate':sr,'sample_offset':begin,'source_start_s':offset,'duration_s':len(audio)/sr,'input_text_status':input_status,
   'seconds_used':round(time.monotonic()-start,2),'units':units,
   'checks':{'finite':finite,'inside_audio':in_bounds,'monotonic':monotonic,'zero_duration':sum(u['end_local_s']==u['start_local_s'] for u in units)},
   'status':'模型候选，待 Jade 听感抽查；不得直接视为训练样本'})
report={'model':str(MODEL),'model_license':'Apache-2.0 (local README)','package':'qwen-asr '+version('qwen-asr'),
 'device':device,'source_hashes_unchanged':all(sha(ROOT/k)==v for k,v in hashes.items()),'audio_output_generated':False,'results':rows}
(OUT/'alignment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('源文件哈希未变:',report['source_hashes_unchanged'],flush=True)
for r in rows:print(r['label'],'字数',len(r['units']),'检查',r['checks'],flush=True)
