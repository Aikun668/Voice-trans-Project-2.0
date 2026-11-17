"""Reference-free experimental pitch predictor. No reference input at inference."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
HOP=.02


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def new_output(path):
    p=Path(path).resolve()
    if p.exists():raise FileExistsError(p)
    if HERE not in p.parents:raise ValueError('Output must be inside this experiment')
    p.mkdir(parents=True,exist_ok=False)
    return p


def interpolate_voiced(times,f0,query):
    times,f0,query=np.asarray(times),np.asarray(f0),np.asarray(query)
    hi=np.searchsorted(times,query,side='right').clip(1,len(times)-1);lo=hi-1
    valid=(f0[lo]>0)&(f0[hi]>0)&(query>=times[0])&(query<=times[-1])
    result=np.zeros(len(query))
    result[valid]=np.exp(np.interp(query[valid],times,np.log(np.maximum(f0,1e-12))))
    # Exact voiced samples at boundaries are valid without inventing intermediate voicing.
    nearest=np.searchsorted(times,query).clip(0,len(times)-1)
    exact=np.abs(times[nearest]-query)<1e-9
    result[exact]=f0[nearest[exact]]
    return result


def own_features(f0):
    f=np.asarray(f0,dtype=float);v=f>0
    midi=np.zeros(len(f));midi[v]=69+12*np.log2(f[v]/440)
    if v.any():
        filled=np.interp(np.arange(len(f)),np.flatnonzero(v),midi[v]);center=np.median(midi[v])
    else:filled=np.zeros(len(f));center=60.
    smooth=median_filter(filled,size=9,mode='nearest')
    hist=np.zeros(12)
    for pitch in np.rint(midi[v]).astype(int)%12:hist[pitch]+=1
    hist/=max(hist.sum(),1)
    features=np.column_stack(((filled-center)/12,v.astype(float),
               filled-np.rint(filled),np.sin(2*np.pi*filled),np.cos(2*np.pi*filled),
               (filled-smooth)/3,np.clip(np.gradient(filled),-3,3)/3,
               np.tile(hist,(len(f),1))))
    return features.astype(np.float32)


def context_features(f0):
    f=np.asarray(f0,dtype=float);base=own_features(f);voiced=f>0
    if voiced.any():
        pitches=69+12*np.log2(f[voiced]/440)
        m=np.interp(np.arange(len(f)),np.flatnonzero(voiced),pitches)
    else:m=np.full(len(f),60.)
    columns=[base]
    for offset in (-50,-25,-10,-5,-2,2,5,10,25,50):
        idx=np.clip(np.arange(len(f))+offset,0,len(f)-1)
        columns.append(np.column_stack((np.clip((m[idx]-m)/3,-4,4),voiced[idx])))
    for size in (5,11,25,51,101):
        columns.append(((m-median_filter(m,size,mode='nearest'))/3)[:,None])
    current=np.rint(m).astype(int)%12
    rotated=base[:,7:19][np.arange(len(f))[:,None],(current[:,None]+np.arange(12))%12]
    columns.append(rotated)
    return np.column_stack(columns).astype(np.float32)


def model_features(model,f0):
    return context_features(f0) if getattr(model,'_f2_feature_kind',None)=='own-context-56' else own_features(f0)


def make_practice(f0,seed):
    """Known feature-level perturbations of complete note runs; preserve natural within-note motion."""
    f=np.asarray(f0,dtype=float);v=f>0
    if not v.any():return f.copy(),np.zeros(len(f))
    m=np.interp(np.arange(len(f)),np.flatnonzero(v),69+12*np.log2(f[v]/440))
    notes=np.rint(median_filter(m,11,mode='nearest')).astype(int)
    runs=[];i=0
    while i<len(f):
        if not v[i]:i+=1;continue
        j=i+1
        while j<len(f) and v[j] and notes[j]==notes[i]:j+=1
        if j-i>=8:runs.append((i,j))
        i=j
    delta=np.zeros(len(f))
    if len(runs)>=3:
        rng=np.random.default_rng(seed)
        for selected in rng.choice(len(runs),size=min(2,max(1,len(runs)//3)),replace=False):
            a,b=runs[selected];shift=float(rng.choice([-300,-200,-100,100,200,300]))
            shift=float(np.clip(shift+rng.uniform(-20,20),-300,300));delta[a:b]=shift
    changed=f*2**(delta/1200)
    return changed,-delta


def render_command(f0,prediction):
    f=np.asarray(f0);p=np.asarray(prediction,dtype=float)
    if p.shape!=f.shape or not np.isfinite(p).all():raise ValueError('Invalid model prediction')
    if not np.any(p):return np.zeros_like(p)
    cmd=gaussian_filter1d(np.clip(p,-300,300),1.5,mode='nearest')
    cmd[np.abs(cmd)<20]=0
    cmd[f<=0]=0
    return cmd


class PitchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.GRU(19,48,num_layers=1,bidirectional=True,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,x,lengths=None):
        if lengths is not None:
            x=nn.utils.rnn.pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
            h,_=self.encoder(x)
            h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True)
        else:h,_=self.encoder(x)
        return 300*torch.tanh(self.head(h).squeeze(-1))


def load_model(path):
    payload=torch.load(path,map_location='cpu')
    if payload['schema']=='own-pitch-context-hgb-v1':return payload['model']
    if payload['schema']!='own-pitch-context-v1':raise ValueError('Wrong checkpoint schema')
    model=PitchModel();model.load_state_dict(payload['state'],strict=True);model.eval()
    return model


@torch.inference_mode()
def predict_array(model,x):
    if not isinstance(model,nn.Module):
        return np.clip(np.asarray(model.predict(x),dtype=float),-300,300)
    model.eval()
    return model(torch.as_tensor(x,dtype=torch.float32)[None])[0].numpy().astype(float)


def synthesize(audio,rate,command):
    """Praat PSOLA plus complementary splice guard; zero command is an exact bypass."""
    audio=np.asarray(audio,dtype=float);command=np.asarray(command,dtype=float)
    if not np.any(command):return audio.copy(),np.zeros(len(audio))
    import parselmouth
    call=parselmouth.praat.call
    sound=parselmouth.Sound(audio,rate)
    manipulation=call(sound,'To Manipulation',.01,65,850)
    old=call(manipulation,'Extract pitch tier')
    new=call('Create PitchTier','predicted',0.,len(audio)/rate)
    grid=np.arange(len(command))*HOP
    for i in range(1,call(old,'Get number of points')+1):
        t=call(old,'Get time from index',i);f=call(old,'Get value at index',i)
        call(new,'Add point',t,f*2**(float(np.interp(t,grid,command))/1200))
    call([manipulation,new],'Replace pitch tier')
    wet=np.asarray(call(manipulation,'Get resynthesis (overlap-add)').values[0],dtype=float)
    if len(wet)!=len(audio):raise ValueError('Resynthesis changed sample count')
    # Replace only predicted active regions, including a zero-shift context margin.
    active=np.abs(command)>1.;edges=np.flatnonzero(np.diff(np.r_[False,active,False]))
    mask=np.zeros(len(audio));fade=max(1,int(.035*rate))
    for begin,end in zip(edges[::2],edges[1::2]):
        a=max(0,int((begin*HOP-.05)*rate));b=min(len(audio),int((end*HOP+.05)*rate))
        mask[a:b]=1.
        lo=max(0,a-fade);hi=min(len(audio),b+fade)
        if a>lo:mask[lo:a]=np.maximum(mask[lo:a],np.sin(np.linspace(0,np.pi/2,a-lo))**2)
        if hi>b:mask[b:hi]=np.maximum(mask[b:hi],np.cos(np.linspace(0,np.pi/2,hi-b))**2)
    result=audio+mask*(wet-audio)
    assert np.array_equal(result[mask==0],audio[mask==0])
    return result,mask
