import sys,json,hashlib,importlib.util
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft,istft
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
R=Path(__file__).resolve().parents[2]; D=Path(__file__).resolve().parent
sys.path.append(str(R/'.venv-f2/Lib/site-packages'))
def load(n,p):
 s=importlib.util.spec_from_file_location(n,R/p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
v=load('v5','experiments/WOMENDE_FOCUS_2026-11-04_v005/run.py')
v2=load('v2','experiments/WOMENDE_FOCUS_2026-11-03_v002/run.py')
master=load('master','experiments/F1_demo_chain_2026-11-02_v001/run.py')
ht=load('ht','experiments/WOMENDE_FOCUS_2026-11-03_v001/harmonic_tilt.py')
v1=R/'experiments/WOMENDE_FOCUS_2026-11-03_v001'
base=R/'experiments/WOMENDE_FOCUS_2026-11-04_v006/audio/baseline.wav'
anchor=R/'experiments/WOMENDE_FOCUS_2026-11-04_v006/audio/anchor.wav'
source=v2.SRC
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=D/'audio'; stage=D/'work_output'
if out.exists() or stage.exists() or (D/'report.json').exists():raise SystemExit('已有输出，拒绝覆盖')
stage.mkdir()
hashes={str(p):sha(p) for p in (source,base,v1/'work/thin_keyup.wav')}
x,sr=sf.read(v1/'work/thin_keyup.wav',dtype='float64')
freq,times,z=stft(x,sr,nperseg=ht.NFFT,noverlap=ht.NFFT-ht.HOP,window='hann',boundary='even',padded=True)
t0,f0=ht.load_f0('thin_keyup')
fz=np.interp(times,t0,f0,left=0,right=0)
near=np.interp(times,t0,(f0>0).astype(float),left=0,right=0)
voiced=(near>.99)&(fz>60)
w=np.convolve(voiced.astype(float),np.ones(5)/5,mode='same')
slope=float(json.loads((v1/'report.json').read_text(encoding='utf-8'))['c2']['slope_db_per_oct'])
gain=ht.tilt_gain_db(freq,slope)
g=np.zeros(z.shape,dtype='float64')
for k in np.flatnonzero(w>0):
 f=fz[k] if fz[k]>60 else np.max(fz[max(0,k-3):k+4])
 if f<=60:continue
 d=freq-np.round(freq/f)*f
 m=np.exp(-.5*(d/(ht.SIGMA*f))**2)*(freq>.5*f)
 g[:,k]=gain*m*w[k]
masterinfo={}
processed={}
slow=gaussian_filter1d(g,sigma=15,axis=1,mode='nearest')
earlier=gaussian_filter1d(g,sigma=5,axis=1,mode='nearest')
# Work only in sustained voiced regions; fade gradually around boundaries.
stable=uniform_filter1d(voiced.astype(float),size=45,mode='nearest')
gate=gaussian_filter1d(np.clip((stable-.8)/.2,0,1),sigma=12,mode='nearest')
comp=np.clip(earlier-slow,0,3)*gate[None,:]
# The evidence points at 1.2–3 kHz; fade to zero beyond that region.
band=np.interp(freq,[0,800,1200,3000,4000,24000],[0,0,1,1,0,0])
comp*=band[:,None]
for label,amount in [('medium',0.5),('strong',1.0)]:
 adjusted=slow+amount*comp
 assert np.max(np.abs(adjusted[freq<250]))==0
 _,y=istft(z*10**(adjusted/20),sr,nperseg=ht.NFFT,noverlap=ht.NFFT-ht.HOP,window='hann',boundary=True)
 processed[label],masterinfo[label]=v.master_round(master,np.asarray(y[:len(x)],dtype=np.float32),sr)
original,_=sf.read(source); b,_=sf.read(base); a,_=sf.read(anchor)
previous_path=R/'experiments/WOMENDE_VIBRATION_LIMIT_2026-11-05_v001/audio/strong.wav'
hashes[str(previous_path)]=sha(previous_path)
previous,_=sf.read(previous_path)
rows={}
for name,samples in [('anchor',a),('previous',previous),('medium',processed['medium']),('strong',processed['strong'])]:
 path=stage/(name+'.wav');sf.write(path,samples,sr,subtype='PCM_16')
 back,s=sf.read(path); lufs,tp=master.ebur(path)
 pd=v2.pitch_delta(source,path,0 if name=='anchor' else 310)
 f1=v2.pitch_delta(base,path,0) if name in ('previous','medium','strong') else None
 failures=[]
 if len(back)!=len(original) or s!=sr or not np.isfinite(back).all() or np.any(np.abs(back)>=.9999):failures.append('samples or clipping')
 if abs(lufs+13)>.4 or tp>-1:failures.append('loudness or true peak')
 target=0 if name=='anchor' else 310
 if pd['median_cents'] is None or abs(pd['median_cents']-target)>=5 or pd['voiced_overlap']<.95:failures.append('pitch source')
 if f1 and (f1['median_cents'] is None or abs(f1['median_cents'])>=5 or f1['voiced_overlap']<.95):failures.append('pitch F1')
 stats=master.level_stats(back,sr)
 from scipy.signal import butter,sosfilt
 high=sosfilt(butter(4,[5000,9000],btype='bandpass',fs=sr,output='sos'),back)
 size=int(.02*sr); count=len(back)//size
 e=10*np.log10(np.mean(back[:count*size].reshape(count,size)**2,axis=1)+1e-12)
 h=10*np.log10(np.mean(high[:count*size].reshape(count,size)**2,axis=1)+1e-12)
 sib=round(float(np.percentile((h-e)[e>np.percentile(e,95)-30],99)),2)
 rows[name]={'lufs':lufs,'true_peak_dbtp':tp,'pitch_vs_source':pd,'pitch_vs_b':f1,'decay_lift_db':round(v.decay_lift(original,back,sr),2),'floor_relative_db':round(stats['floor_p05_all']-stats['p95'],2),'sibilance_p99_db':sib,'sha256':sha(path),'failures':failures}
 print(name,lufs,tp,failures,flush=True)
same=all(sha(Path(p))==h for p,h in hashes.items())
report={'technical_pass':same and all(not r['failures'] for r in rows.values()),'listening':'pending_jade','source_hashes':hashes,'source_hashes_unchanged':same,'smoothing_sigma_ms':80,'compensation_fraction':{'medium':0.5,'strong':1.0},'compensation_max_db':3,'compensation_band':'800-4000 Hz, full 1200-3000 Hz','original_slope_db_oct':slope,'below_250_gain_db':0,'master':masterinfo,'versions':rows}
(D/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
if not report['technical_pass']:raise SystemExit('工程门未通过；仅保留 work_output')
stage.rename(out)
