"""Necessary input/output adaptation around frozen MusicBERT MLM weights.

No learned task head, no training. Upstream PyTorch/HF implementation supplies
the actual model; existing project Praat synthesis supplies waveform shifting.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import median_filter

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
ASSETS=HERE/'assets'
HOP=.02
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('HF_HOME',str(HERE/'cache/huggingface'))
os.environ.setdefault('NUMBA_CACHE_DIR',str(HERE/'cache/numba'))
for p in (HERE/'vendor/deps',HERE/'vendor/musicbert_hf_repo'):
    sys.path.insert(0,str(p))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
    return h.hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def new_output(path):
    p=Path(path).resolve()
    if HERE not in p.parents:raise ValueError('Output must stay inside the experiment')
    p.mkdir(parents=True,exist_ok=False)
    return p


def _reuse(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthesize(audio,rate,command):
    # Read-only reuse of an existing, separately exercised local PSOLA splice.
    module=_reuse(PROJECT/'experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/trial.py','frozen_praat_backend')
    return module.synthesize(audio,rate,command)


def pitch_tracks(y,sr):
    import parselmouth
    sound=parselmouth.Sound(y,sr)
    grid=np.arange(max(2,int(np.ceil(len(y)/sr/HOP))))*HOP
    tracks=[]
    for method in (sound.to_pitch_ac,sound.to_pitch_cc):
        p=method(time_step=HOP,pitch_floor=65,pitch_ceiling=850,voicing_threshold=.5)
        idx=np.rint((grid-p.x1)/p.dx).astype(int)
        valid=(idx>=0)&(idx<len(p.selected_array['frequency']))
        f=np.zeros(len(grid));f[valid]=p.selected_array['frequency'][idx[valid]]
        tracks.append(f)
    a,b=tracks
    agree=(a>0)&(b>0)&(np.abs(1200*np.log2(np.maximum(a,1)/np.maximum(b,1)))<=50)
    return grid,a,b,agree


def segment_notes(time,f0,agreement):
    time=np.asarray(time);f0=np.asarray(f0);agreement=np.asarray(agreement,dtype=bool)
    if len(time)!=len(f0) or len(f0)!=len(agreement):raise ValueError('Unequal analysis arrays')
    voiced=f0>0
    if not voiced.any():return []
    midi=np.zeros(len(f0));midi[voiced]=69+12*np.log2(f0[voiced]/440)
    fill=np.interp(np.arange(len(f0)),np.flatnonzero(voiced),midi[voiced])
    quant=np.rint(median_filter(fill,size=7,mode='nearest')).astype(int)
    result=[];i=0
    while i<len(f0):
        if not voiced[i]:i+=1;continue
        j=i+1
        while j<len(f0) and voiced[j] and quant[j]==quant[i]:j+=1
        values=midi[i:j];center=float(np.median(values))
        stable=(np.abs(values-center)<=.35)&agreement[i:j]
        stable_seconds=float(stable.sum()*HOP)
        reliability=float(agreement[i:j].mean())
        result.append({'index':len(result),'start':float(time[i]),'end':float(time[j-1]+HOP),
                       'begin_frame':i,'end_frame':j,'pitch':int(np.rint(center)),
                       'center':center,'stable_seconds':stable_seconds,'agreement':reliability,
                       'eligible':bool(stable_seconds>=.2-1e-8 and reliability>=.9)})
        i=j
    return result


class Codec:
    """Reuse upstream quantizers and dictionary generator without its pool side effects."""
    def __init__(self):
        source=ASSETS/'preprocess.py';tree=ast.parse(source.read_text(encoding='utf-8'))
        nodes=[]
        names={'ts_dict','ts_list','dur_enc','dur_dec'}
        functions={'t2e','d2e','v2e','b2e','gen_dictionary'}
        for node in tree.body:
            if isinstance(node,ast.Assign) and (isinstance(node.value,ast.Constant) or
                 any(isinstance(t,ast.Name) and t.id in names for t in node.targets)):
                nodes.append(node)
            elif isinstance(node,ast.For):nodes.append(node)
            elif isinstance(node,ast.FunctionDef) and node.name in functions:nodes.append(node)
        env={'math':math};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
        self.quantizers=env
        path=ASSETS/'dict.txt'
        if not path.exists():env['gen_dictionary'](str(path))
        self.symbols=['<s>','<pad>','</s>','<unk>']+[line.split()[0] for line in path.read_text().splitlines()]+['<mask>']
        self.ids={v:k for k,v in enumerate(self.symbols)}
        if len(self.ids)!=1237:raise ValueError('Official dictionary size mismatch')
        self.pitch_ids=np.array([self.ids[f'<3-{i}>'] for i in range(128)])
        self.mask=self.ids['<mask>']

    def encode(self,notes):
        # Fixed 120 BPM / 4-4 / GM piano / velocity 80 are input conventions,
        # not a claim to have recovered the true meter of an a cappella recording.
        result=[0]*8
        q=self.quantizers
        for note in notes:
            p=note['pitch']
            if not isinstance(p,(int,np.integer)) or p<0 or p>127:raise ValueError('Bad MIDI pitch')
            pos=round(note['start']*32)
            duration=max(1,round((note['end']-note['start'])*32))
            fields=(pos//64,pos%64,0,int(p),q['d2e'](duration),q['v2e'](80),q['t2e']((4,4)),q['b2e'](120))
            try:result.extend(self.ids[f'<{i}-{v}>'] for i,v in enumerate(fields))
            except KeyError as e:raise ValueError('Octuple value outside official vocabulary') from e
        result.extend([2]*8)
        return np.asarray(result,dtype=np.int64)


def tensor_hash(tensor):
    a=tensor.detach().cpu().contiguous().numpy()
    return (str(a.dtype),tuple(a.shape),hashlib.sha256(a.tobytes()).hexdigest())


class FrozenMusicBert:
    def __init__(self):
        import torch
        from musicbert_hf.checkpoints import load_musicbert_from_fairseq_checkpoint
        self.torch=torch
        torch.set_num_threads(4)
        self.checkpoint=ASSETS/'checkpoint_last_musicbert_small.pt'
        provenance=json.loads((ASSETS/'upstream.json').read_text(encoding='utf-8'))
        expected=provenance['checkpoint']['sha256']
        if sha(self.checkpoint)!=expected:raise ValueError('Checkpoint hash changed')
        original=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        state=original['model']
        self.model=load_musicbert_from_fairseq_checkpoint(str(self.checkpoint))
        # This embedding is created by HF but is never read by CompoundEmbeddings.
        # Removing it avoids retaining any randomly initialized unused parameter.
        del self.model.bert.embeddings.token_type_embeddings
        # Fairseq uses eps=1e-5; HF's default is 1e-12. Match the owning source.
        for module in self.model.modules():
            if isinstance(module,torch.nn.LayerNorm):module.eps=1e-5
        self.model.eval();self.model.requires_grad_(False)
        trained={tensor_hash(v) for v in state.values() if torch.is_tensor(v)}
        unmatched=[name for name,p in self.model.named_parameters() if tensor_hash(p) not in trained]
        if unmatched:raise ValueError('Inference parameters missing original weights: '+str(unmatched))
        self.codec=Codec()
        if self.model.config.vocab_size!=len(self.codec.ids):raise ValueError('Vocabulary mismatch')
        self.before=self.parameter_digest()
        self.meta={'checkpoint_sha256':expected,'upstream_model_commit':provenance['commit'],
                   'adapter_repo_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=HERE/'vendor/musicbert_hf_repo',text=True).strip(),
                   'num_layers':self.model.config.num_hidden_layers,'hidden_size':self.model.config.hidden_size,
                   'num_parameters':sum(p.numel() for p in self.model.parameters()),
                   'all_inference_parameters_match_pretrained_tensors':not unmatched,
                   'trainable_parameters':sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                   'original_masked_language_head_used':True,'optimizer_created':False,'training_steps':0,
                   'parameter_digest_before':self.before,'layernorm_epsilon':1e-5,
                   'unused_hf_token_type_embeddings_removed':True,'codec':{'tempo_bpm':120,'meter':'4/4','instrument':0,'velocity':80}}
        self.original_state=state

    def parameter_digest(self):
        h=hashlib.sha256()
        for name,p in self.model.named_parameters():
            h.update(name.encode());h.update(str(tensor_hash(p)).encode())
        return h.hexdigest()

    def logits(self,ids):
        torch=self.torch
        with torch.inference_mode():
            return self.model(input_ids=torch.as_tensor(ids,dtype=torch.long)).logits

    def probabilities(self,notes,indices=None,batch_size=16):
        if indices is None:indices=list(range(len(notes)))
        if not notes:return np.zeros((0,128))
        tokens=self.codec.encode(notes)
        result=[]
        for begin in range(0,len(indices),batch_size):
            group=indices[begin:begin+batch_size]
            data=np.tile(tokens,(len(group),1))
            positions=np.array([(i+1)*8+3 for i in group])
            data[np.arange(len(group)),positions]=self.codec.mask
            logits=self.logits(data)[np.arange(len(group)),positions]
            pitch_logits=logits[:,self.codec.pitch_ids]
            result.extend(self.torch.softmax(pitch_logits,dim=-1).cpu().numpy())
        return np.asarray(result,dtype=float)


def choose_target(note,probabilities):
    p=np.asarray(probabilities,dtype=float)
    if p.shape!=(128,) or not np.isfinite(p).all() or np.any(p<0):raise ValueError('Invalid pitch probabilities')
    center=float(note['center']);current=int(note['pitch'])
    candidates=np.arange(max(0,current-2),min(128,current+3))
    values=p[candidates];local=values/max(float(values.sum()),1e-30)
    selected=int(np.argmax(values));target=int(candidates[selected])
    odds=float(p[target]/max(p[current],1e-30));shift=100*(target-center)
    reasons=[]
    if target==current:reasons.append('model_keeps_current_note')
    if not note['eligible']:reasons.append('short_or_unreliable_note')
    if odds<3:reasons.append('odds_below_3')
    if local[selected]<.5:reasons.append('local_probability_below_half')
    if abs(shift)>200:reasons.append('shift_exceeds_200_cents')
    return {**note,'target':target,'raw_best_pitch':int(np.argmax(p)),'current_probability':float(p[current]),
            'target_probability':float(p[target]),'local_target_probability':float(local[selected]),
            'odds':odds,'shift_cents':shift,'accepted':not reasons,'reasons':reasons}


def make_command(grid,rows):
    grid=np.asarray(grid);command=np.zeros(len(grid))
    for row in rows:
        if not row['accepted']:continue
        active=(grid>=row['start'])&(grid<row['end'])
        t=grid[active];fade=min(.06,(row['end']-row['start'])/4)
        ramp=np.minimum(np.clip((t-row['start'])/fade,0,1),np.clip((row['end']-t)/fade,0,1))
        command[active]=row['shift_cents']*(.5-.5*np.cos(np.pi*ramp))
    return command


if __name__=='__main__':
    model=FrozenMusicBert();print(json.dumps(model.meta,ensure_ascii=False,indent=2))
    dump(HERE/'assets/model_load.json',model.meta)
