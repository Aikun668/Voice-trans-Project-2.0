"""只读识别原唱候选窗口，为任务一准备可批量核对的文字。"""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'results'
if OUT.exists(): raise SystemExit(f'拒绝覆盖已有输出: {OUT}')
OUT.mkdir()
sys.path.append(r'D:\My_Grand_Projects2026\Voice-trans-Project-2.0\.venv-f1-adaptive\Lib\site-packages')
import soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel

BASE=ROOT/'test_dataset/my_singing/rerecord_2026-11-04_f2/ref_vocals'
CASES=[
 ('心墙01','心墙_郭静_vocals.wav',112.8,123.2),
 ('离开我的依赖01','离开我的依赖_王艳薇_vocals.wav',53.4,68.8),
 ('离开我的依赖02','离开我的依赖_王艳薇_vocals.wav',68.5,80.9),
 ('莫问归期01-候选A','莫问归期_蒋雪儿_vocals.wav',30.6,42.9),
 ('莫问归期01-候选B','莫问归期_蒋雪儿_vocals.wav',113.1,125.4),
 ('莫问归期02','莫问归期_蒋雪儿_vocals.wav',133.4,146.4),
]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
files={name:BASE/name for _,name,_,_ in CASES}
hashes={name:sha(p) for name,p in files.items()}
model=WhisperModel(r'C:\Users\杨玉璋\.cache\huggingface\hub\models--Systran--faster-whisper-small\snapshots\536b0662742c02347bc0e980a01041f333bce120',device='cpu',compute_type='int8',local_files_only=True,cpu_threads=4)
items=[]
for label,name,start,end in CASES:
 print(label,flush=True)
 p=files[name]; info=sf.info(str(p)); y,sr=sf.read(str(p),start=int(start*info.samplerate),stop=int(end*info.samplerate),dtype='float32')
 if y.ndim==2: y=y.mean(axis=1)
 if sr!=16000:
  from math import gcd
  g=gcd(sr,16000);y=resample_poly(y,16000//g,sr//g)
 segments,_=model.transcribe(y,language='zh',beam_size=3,vad_filter=False,condition_on_previous_text=False)
 result=[{'start_in_window_s':round(s.start,2),'end_in_window_s':round(s.end,2),'text':s.text.strip()} for s in segments]
 items.append({'group':label,'source':str(p.relative_to(ROOT)),'window_s':[start,end],'segments':result,'status':'原唱候选机器识别；歌词和位置未确认'})
data={'items':items,'source_sha256':hashes,'source_hashes_unchanged':all(sha(files[k])==v for k,v in hashes.items()),'model':'Systran/faster-whisper-small, faster-whisper 1.2.1 CPU int8','note':'窗口内临时读取并重采样内存，不生成音频；位置与文字皆仅供辅助核对。'}
(OUT/'reference_candidates.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print('源文件未改变:',data['source_hashes_unchanged'],flush=True)
