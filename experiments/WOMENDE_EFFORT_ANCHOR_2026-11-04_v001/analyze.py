from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
import soundfile as sf
R=Path(__file__).resolve().parents[2]
D=Path(__file__).resolve().parent
sys.path.append(str(R/'.venv-f2/Lib/site-packages'))
sys.path.insert(0,str(R/'f2'))
from poc_phase4_realign import cents_track,mfcc_track,mfcc_cost,cost_matrix,subseq_dtw,HOP
import parselmouth
P={
 'thin':R/'test_dataset/f1_dataset_v2/A_thin/self/wav/我们的歌-薄音版.wav',
 'normal':R/'test_dataset/f1_dataset_v2/A_thin/self/wav/我们的歌正常版.wav',
 'current':R/'experiments/WOMENDE_VIBRATION_LIMIT_2026-11-05_v001/audio/strong.wav',
}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
output=D/'analysis.json'
if output.exists():raise SystemExit('拒绝覆盖结果')
hashes={k:digest(p) for k,p in P.items()}
q=cents_track(P['thin']); r=cents_track(P['normal'])
mq=mfcc_track(P['thin']); mr=mfcc_track(P['normal'])
qn=min(len(q),len(mq)); rn=min(len(r),len(mr))
q=q[:qn];r=r[:rn];mq=mq[:qn];mr=mr[:rn]
pitchcost,pp=subseq_dtw(cost_matrix(q,r,-310.))
reversecost,_=subseq_dtw(cost_matrix(q,r[::-1].copy(),-310.))
mfcost,mp=subseq_dtw(mfcc_cost(mq,mr))
mfrev,_=subseq_dtw(mfcc_cost(mq,mr[::-1].copy()))
def mapping(path,n):
 slots=[[] for _ in range(n)]
 for i,j in path:
  if i<n:slots[i].append(j)
 return np.array([np.median(a) if a else np.nan for a in slots])
pm=mapping(pp,qn); mm=mapping(mp,qn)
match=np.isfinite(pm)&np.isfinite(mm)
diff=np.abs(pm-mm)*HOP
agree=match&(diff<=.25)
report={'alignment':{'pitch_cost':round(pitchcost,2),'pitch_reverse':round(reversecost,2),'mfcc_cost':round(mfcost,3),'mfcc_reverse':round(mfrev,3),'comparison_frames':int(match.sum()),'median_disagreement_s':round(float(np.nanmedian(diff[match])),3),'within_250ms_fraction':round(float(np.mean(agree[match])),3),'note':'音高与MFCC两种自动对齐，仅为候选；未核实歌词和元音，绝不作为同字确定证据。'},'source_sha256':hashes,'windows':[]}
arrays={k:sf.read(p)[0] for k,p in P.items()}
sr=48000
def measurements(x,lo,hi):
 segment=x[int(lo*sr):int(hi*sr)]
 s=parselmouth.Sound(segment,sr)
 pitch=s.to_pitch(.01,75,600)
 f0=pitch.selected_array['frequency']; ts=pitch.xs()
 n=int(.04*sr)
 vals=[]
 for t,f in zip(ts,f0):
  if f<=0:continue
  c=int(t*sr); frame=segment[c-n//2:c+n//2]
  if len(frame)!=n:continue
  amp=np.abs(np.fft.rfft(frame*np.hanning(n),8192));freq=np.fft.rfftfreq(8192,1/sr)
  power=amp**2; db=20*np.log10(amp+1e-12)
  def h(f0):
   use=(freq>f0*.9)&(freq<f0*1.1)
   return np.max(db[use]) if np.any(use) else np.nan
  alpha=10*np.log10((power[(freq>=1000)&(freq<5000)].sum()+1e-20)/(power[(freq>=50)&(freq<1000)].sum()+1e-20))
  vals.append([f,alpha,h(f)-h(2*f)])
 if len(vals)<10:return None
 vals=np.array(vals)
 return {'voiced_frames':len(vals),'f0_median_hz':round(float(np.median(vals[:,0])),1),'alpha_db':round(float(np.nanmedian(vals[:,1])),2),'h1_h2_db':round(float(np.nanmedian(vals[:,2])),2)}
candidates=[]
for i in range(8,qn-28):
 sl=slice(i,i+20)
 if not np.all(agree[sl]):continue
 js=pm[sl]; ii=np.round(js).astype(int)
 if np.any(ii<0) or np.any(ii>=len(r)):continue
 if np.isnan(q[sl]).any() or np.isnan(r[ii]).any():continue
 err=np.abs((q[sl]-r[ii]+310+600)%1200-600)
 if np.median(err)>40 or np.percentile(err,90)>80:continue
 if np.ptp(q[sl])>100 or np.ptp(r[ii])>100:continue
 candidates.append((float(np.median(err)),i))
chosen=[]
for err,i in sorted(candidates):
 if all(abs(i-j)>40 for j in chosen):chosen.append(i)
 if len(chosen)>=3:break
for i in sorted(chosen):
 lo=i*HOP;hi=(i+20)*HOP
 rlo=float(pm[i]*HOP); rhi=float(pm[i+19]*HOP)
 if rhi-rlo<.2:continue
 vals={'thin':measurements(arrays['thin'],lo,hi),'current':measurements(arrays['current'],lo,hi),'normal':measurements(arrays['normal'],rlo,rhi)}
 if any(v is None for v in vals.values()):continue
 report['windows'].append({'thin_time_s':[round(lo,2),round(hi,2)],'normal_time_s':[round(rlo,2),round(rhi,2)],'identity_status':'自动候选，歌词与元音未确认','metrics':vals})
report['source_hashes_unchanged']=all(digest(P[k])==v for k,v in hashes.items())
output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
