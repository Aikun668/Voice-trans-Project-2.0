import sys, json, hashlib, importlib.util
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
R=Path(__file__).resolve().parents[2]
D=Path(__file__).resolve().parent
sys.path.append(str(R/'.venv-f2/Lib/site-packages'))
def load(n,p):
 s=importlib.util.spec_from_file_location(n,R/p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
v=load('old','experiments/WOMENDE_FOCUS_2026-11-04_v005/run.py')
v2=load('pitch','experiments/WOMENDE_FOCUS_2026-11-03_v002/run.py')
master=load('master','experiments/F1_demo_chain_2026-11-02_v001/run.py')
base=R/'experiments/WOMENDE_FOCUS_2026-11-04_v006/audio/baseline.wav'
anchor=R/'experiments/WOMENDE_FOCUS_2026-11-04_v006/audio/anchor.wav'
source=v2.SRC
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
out=D/'audio'
if out.exists() or (D/'work').exists() or (D/'report.json').exists(): raise SystemExit('输出存在，拒绝覆盖')
work=D/'work'; work.mkdir()
source_hash=sha(source); base_hash=sha(base)
x,sr=sf.read(base); a,_=sf.read(anchor); n=len(x)
f,t,z=stft(x,fs=sr,nperseg=4096,noverlap=3072,boundary='zeros',padded=True)
curve=np.zeros(len(f))
rise=(f>=300)&(f<450); curve[rise]=(f[rise]-300)/150
curve[(f>=450)&(f<=700)]=1
fall=(f>700)&(f<1000); curve[fall]=(1000-f[fall])/300
assert np.all(curve[f<250]==0)
outputs={'anchor':a,'base':x}
for name,db in [('reduce',-2.5),('add',2.5)]:
 _,y=istft(z*10**(db*curve[:,None]/20),fs=sr,nperseg=4096,noverlap=3072,boundary=True)
 y=y[:n]
 outputs[name],_=v.master_round(master,np.asarray(y,dtype=np.float32),sr)
rows={}
srcstats=master.level_stats(sf.read(source)[0],sr)
for name,y in outputs.items():
 path=work/(name+'.wav')
 sf.write(path,y,sr,subtype='PCM_16')
 b,srb=sf.read(path)
 lufs,tp=master.ebur(path)
 pitch=v2.pitch_delta(source,path,0 if name=='anchor' else 310)
 pbase=v2.pitch_delta(base,path,0) if name!='anchor' else None
 stats=master.level_stats(b,sr)
 decay=v.decay_lift(sf.read(source)[0],b,sr)
 import scipy.signal as sig
 high=sig.sosfilt(sig.butter(4,[5000,9000],btype='bandpass',fs=sr,output='sos'),b)
 def env(x):
  size=int(.02*sr); count=len(x)//size
  return 10*np.log10(np.mean(x[:count*size].reshape(count,size)**2,axis=1)+1e-12)
 e=env(b); h=env(high); active=e>np.percentile(e,95)-30
 sib=float(np.percentile((h-e)[active],99))
 failures=[]
 if len(b)!=n or srb!=sr or not np.isfinite(b).all() or np.any(np.abs(b)>=.9999): failures.append('尺寸或削波')
 if abs(lufs+13)>.4 or tp>-1: failures.append('响度或峰值')
 target=0 if name=='anchor' else 310
 if pitch['median_cents'] is None or abs(pitch['median_cents']-target)>=5 or pitch['voiced_overlap']<.95: failures.append('原音音高检查')
 if pbase and (pbase['median_cents'] is None or abs(pbase['median_cents'])>=5 or pbase['voiced_overlap']<.95): failures.append('相对B音高检查')
 rows[name]={'lufs':lufs,'tp':tp,'samples':len(b),'finite':bool(np.isfinite(b).all()),'pitch':pitch,'pitch_vs_b':pbase,'decay_lift_db':round(decay,2),'floor_relative_db':round(stats['floor_p05_all']-stats['p95'],2),'sibilance_p99_db':round(sib,2),'sha256':sha(path),'failures':failures}
 print(name,lufs,tp,failures,flush=True)
ok=all(not x['failures'] for x in rows.values()) and sha(base)==base_hash and sha(source)==source_hash
report={'technical_pass':ok,'listening':'pending_jade','source_sha256':source_hash,'base_sha256':base_hash,'eq_db':{'reduce':-2.5,'add':2.5},'curve':'300–450 rise, 450–700 full, 700–1000 fade; gain 0 below 250','versions':rows}
(D/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
if not ok: raise SystemExit('工程门未通过，保留work')
work.rename(out)
