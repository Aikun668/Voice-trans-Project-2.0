import sys,json,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
R=Path(__file__).resolve().parents[2]; D=Path(__file__).resolve().parent
sys.path.append(str(R/'.venv-f2/Lib/site-packages'))
import parselmouth
paths={
 'thin':(R/'test_dataset/f1_dataset_v2/A_thin/self/wav/我们的歌-薄音版.wav',4.50,4.90),
 'shift':(R/'experiments/WOMENDE_FOCUS_2026-11-03_v001/work/thin_keyup.wav',4.50,4.90),
 'current':(R/'experiments/WOMENDE_VIBRATION_LIMIT_2026-11-05_v001/audio/strong.wav',4.50,4.90),
 'normal':(R/'test_dataset/f1_dataset_v2/A_thin/self/wav/我们的歌正常版.wav',4.50,4.80),
}
out=D/'confirmed_word.json'
if out.exists():raise SystemExit('拒绝覆盖')
hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,(p,_,_) in paths.items()}
rows={}
for name,(p,lo,hi) in paths.items():
 x,sr=sf.read(p)
 # Trim 50 ms at both ends to focus on the sustained center.
 lo+=.05; hi-=.05
 clip=x[int(lo*sr):int(hi*sr)]
 sound=parselmouth.Sound(clip,sr)
 pitch=sound.to_pitch(.01,75,600)
 f0=pitch.selected_array['frequency']; ts=pitch.xs()
 n=int(.04*sr); hlist=[]; alphas=[]
 for t,f in zip(ts,f0):
  c=int(t*sr); frame=clip[c-n//2:c+n//2]
  if f<=0 or len(frame)!=n:continue
  z=np.abs(np.fft.rfft(frame*np.hanning(n),8192))
  freq=np.fft.rfftfreq(8192,1/sr)
  db=20*np.log10(z+1e-12); power=z*z
  peaks=[]
  for harmonic in range(1,11):
   band=(freq>harmonic*f*.9)&(freq<harmonic*f*1.1)
   peaks.append(float(np.max(db[band])) if band.any() else float('nan'))
  hlist.append(np.array(peaks)-peaks[0])
  alpha=10*np.log10((power[(freq>=1000)&(freq<5000)].sum()+1e-20)/(power[(freq>=50)&(freq<1000)].sum()+1e-20))
  alphas.append(alpha)
 if len(hlist)<8:raise RuntimeError('有效浊音帧不足 '+name)
 harmonicity=sound.to_harmonicity_cc(.01,75,.1,1.)
 hnr=parselmouth.praat.call(harmonicity,'Get mean',0,0)
 h=np.array(hlist)
 rows[name]={'center_s':[round(lo,2),round(hi,2)],'frames':len(hlist),'f0_median_hz':round(float(np.median(f0[f0>0])),2),
 'alpha_median_db':round(float(np.median(alphas)),2),
 'harmonic_relative_to_h1_db':{str(i):round(float(np.nanmedian(h[:,i-1])),2) for i in range(2,11)},
 'hnr_db':round(float(hnr),2) if hnr==hnr else None}
res={'scope':'用户确认同字，中心窗时长不同，仍可能存在元音变化，不视作逐帧精确配对。','inputs_sha256':hashes,'inputs_unchanged':all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,(p,_,_) in paths.items()),'versions':rows}
out.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False,indent=2))
