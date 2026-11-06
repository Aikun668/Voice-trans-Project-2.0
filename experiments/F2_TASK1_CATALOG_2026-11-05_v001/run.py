"""只读盘点、初步定位原唱与生成歌词候选；结果仅供人工核对。"""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / 'results'
if OUTPUT.exists():
    raise SystemExit(f'拒绝覆盖已有输出目录: {OUTPUT}')
OUTPUT.mkdir()
sys.path.append(r'D:\My_Grand_Projects2026\Voice-trans-Project-2.0\.venv-f1-adaptive\Lib\site-packages')
sys.path.append(r'D:\My_Grand_Projects2026\Voice-trans-Project-2.0\.venv-f2\Lib\site-packages')
sys.path.insert(0, str(ROOT / 'f2'))
import numpy as np
import soundfile as sf
import parselmouth
from faster_whisper import WhisperModel
from poc_phase4_realign import cost_matrix, subseq_dtw

BASE = ROOT / 'test_dataset/my_singing/rerecord_2026-11-04_f2'
MODEL = Path(r'C:\Users\杨玉璋\.cache\huggingface\hub\models--Systran--faster-whisper-small\snapshots\536b0662742c02347bc0e980a01041f333bce120')
songs = {'心墙': '心墙_郭静_vocals.wav', '离开我的依赖': '离开我的依赖_王艳薇_vocals.wav', '莫问归期': '莫问归期_蒋雪儿_vocals.wav'}
targets = sorted(p for p in (BASE/'wav').glob('*.wav') if ('轻度跑调' in p.name or '对照微微跑调' in p.name))
assert len(targets) == 10

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def pitch(p):
    x,sr=sf.read(str(p),dtype='float64',always_2d=True)
    f=parselmouth.Sound(x.mean(axis=1),sr).to_pitch(time_step=.1,pitch_floor=75,pitch_ceiling=600).selected_array['frequency']
    out=np.full(len(f),np.nan); valid=f>0; out[valid]=1200*np.log2(f[valid]/440)
    return out

print('载入本地语音识别模型', flush=True)
model=WhisperModel(str(MODEL),device='cpu',compute_type='int8',local_files_only=True,cpu_threads=4)
refs={name:BASE/'ref_vocals'/filename for name,filename in songs.items()}
hashes={str(p.relative_to(ROOT)):sha(p) for p in targets+list(refs.values())}
tracks={k:pitch(p) for k,p in refs.items()}
items=[]
for n,p in enumerate(targets,1):
    song=next(k for k in songs if p.name.startswith(k)); print(f'{n}/{len(targets)} {p.name}',flush=True)
    info=sf.info(str(p)); q=pitch(p); ref=tracks[song]; best=None
    for key in range(-600,600,100):
        cost,path=subseq_dtw(cost_matrix(q,ref,float(key)))
        if best is None or cost<best[0]: best=(cost,path,key)
    cost,path,key=best
    reverse,_=subseq_dtw(cost_matrix(q,ref[::-1].copy(),float(key)))
    segments,_=model.transcribe(str(p),language='zh',beam_size=3,vad_filter=False,condition_on_previous_text=False)
    parts=[{'start':round(s.start,2),'end':round(s.end,2),'text':s.text.strip()} for s in segments]
    items.append({'song':song,'file':str(p.relative_to(ROOT)),'duration_s':round(info.duration,3),
       'reference':str(refs[song].relative_to(ROOT)), 'ref_start_s':round(path[0][1]*.1,2),
       'ref_end_s':round(path[-1][1]*.1,2),'alignment_cost':round(cost,2),'reverse_cost':round(reverse,2),
       'coarse_key_cents':key,'lyrics_candidates':parts,'status':'未确认，自动候选可能错字、漏字或匹配错误'})
report={'items':items,'source_sha256':hashes,
 'source_hashes_unchanged':all(sha(ROOT/k)==v for k,v in hashes.items()),
 'method':'F2 已有 pitch DTW，0.1 秒帧、100 音分搜索；faster-whisper-small 本地中文识别。结果只能作定位和歌词候选，不是歌词核对、逐字对齐或跑调标签。'}
(OUTPUT/'catalog.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('完成，源文件未改变:',report['source_hashes_unchanged'],flush=True)
