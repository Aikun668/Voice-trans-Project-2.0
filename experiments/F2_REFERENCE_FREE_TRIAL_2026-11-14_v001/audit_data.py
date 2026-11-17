"""Cross-check MFCC pairs with pretrained ASR content features and two pitch estimators."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings('ignore',message='pkg_resources is deprecated.*')
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.spatial.distance import cdist
import parselmouth
import torch
from trial import HERE,PROJECT,HOP,sha,dump,new_output,interpolate_voiced

PREVIOUS=PROJECT/'experiments/F2_POPBUTFY_CLEAN_2026-11-13_v001'
SRC=PROJECT/'third_party_neuralsvb/src'
ASR=SRC/'checkpoints/1009_pretrain_asr_english'


def csv_write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def pitch_tracks(y,sr):
    sound=parselmouth.Sound(y,sr)
    grid=np.arange(max(2,int(np.ceil(len(y)/sr/HOP))))*HOP
    tracks=[]
    for method in (sound.to_pitch_ac,sound.to_pitch_cc):
        p=method(time_step=HOP,pitch_floor=65,pitch_ceiling=850,voicing_threshold=.5)
        idx=np.rint((grid-p.x1)/p.dx).astype(int);valid=(idx>=0)&(idx<len(p.selected_array['frequency']))
        track=np.zeros(len(grid));track[valid]=p.selected_array['frequency'][idx[valid]];tracks.append(track)
    a,c=tracks
    agreement=(a>0)&(c>0)&(np.abs(1200*np.log2(np.maximum(a,1)/np.maximum(c,1)))<=50)
    return grid,a,c,agreement


class ContentEncoder:
    def __init__(self):
        os.environ.setdefault('NUMBA_CACHE_DIR',str(HERE/'numba_cache'))
        sys.path.insert(0,str(SRC))
        import yaml
        from utils.hparams import hparams
        from modules.voice_conversion.vc_modules import VCASR
        config=yaml.safe_load((ASR/'config.yaml').read_text(encoding='utf-8'))
        hparams.clear();hparams.update(config)
        checkpoint=torch.load(ASR/'model_ckpt_steps_136000.ckpt',map_location='cpu')
        state=checkpoint['state_dict']['model']
        self.model=VCASR(state['token_embed.weight'].shape[0],80)
        self.model.load_state_dict(state,strict=True);self.model.eval()
        self.config=config
        self.stride=int(np.prod(config['mel_strides']))
        from data_gen.tts.data_gen_utils import process_utterance
        self.process_utterance=process_utterance

    @torch.inference_mode()
    def extract(self,path,expected_hash):
        blob=Path(path).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=expected_hash:raise ValueError('Source hash changed')
        y,sr=sf.read(io.BytesIO(blob),dtype='float64',always_2d=True);y=y.mean(axis=1)
        g=math.gcd(sr,22050);y=resample_poly(y,22050//g,sr//g).astype(np.float32)
        grid,f0,cc,agree=pitch_tracks(y,22050)
        hp=self.config
        _,mel=self.process_utterance(y,fft_size=hp['fft_size'],hop_size=hp['hop_size'],
                   win_length=hp['win_size'],num_mels=80,fmin=hp['fmin'],fmax=hp['fmax'],sample_rate=22050,loud_norm=False)
        h=self.model(torch.from_numpy(mel.T[None].copy()))['h_content'][0].numpy()
        ht=np.arange(len(h))*hp['hop_size']*self.stride/22050
        valid=ht<len(y)/22050
        return {'time':grid,'f0':f0,'cc':cc,'agreement':agree,'h':h[valid],'ht':ht[valid],
                'duration':len(y)/22050}


def validate_pair(pair,encoder,out):
    row={k:pair[k] for k in ('pair_id','singer','song','split','split_group','amateur_path','professional_path',
                             'amateur_sha256','professional_sha256','lyric')}
    try:
        import librosa
        a=encoder.extract(pair['amateur_path'],pair['amateur_sha256'])
        p=encoder.extract(pair['professional_path'],pair['professional_sha256'])
        cost=cdist(a['h'],p['h'],metric='cosine');cost=np.nan_to_num(cost,nan=1.0)
        def dtw(c):
            _,path=librosa.sequence.dtw(C=c,step_sizes_sigma=np.array([[1,1],[1,2],[2,1]]),
                                     global_constraints=True,band_rad=.35)
            path=path[::-1];return float(c[path[:,0],path[:,1]].mean()),path
        score,path=dtw(cost);reverse,_=dtw(cost[:,::-1].copy())
        q=a['ht'][path[:,0]];r=p['ht'][path[:,1]]
        mapped=np.interp(a['time'],q,r)
        with np.load(pair['map_path'],allow_pickle=False) as previous:
            old=np.interp(a['time'],previous['amateur_time'],previous['professional_time'])
        diff=np.abs(mapped-old)
        target=interpolate_voiced(p['time'],p['f0'],mapped)
        target_cc=interpolate_voiced(p['time'],p['cc'],mapped)
        target_agree=(target>0)&(target_cc>0)&(np.abs(1200*np.log2(np.maximum(target,1)/np.maximum(target_cc,1)))<=50)
        valid=a['agreement']&target_agree&(a['time']>=q[0])&(a['time']<=q[-1])
        voiced=a['f0']>0
        coverage=float(valid.sum()/max(voiced.sum(),1))
        contrast=(reverse-score)/max(reverse,1e-9)
        row.update(content_cost=score,content_reverse_cost=reverse,content_reverse_improvement=contrast,
                   mapping_difference_median_seconds=float(np.median(diff)),mapping_difference_p95_seconds=float(np.percentile(diff,95)),
                   pitch_valid_fraction=coverage,valid_frames=int(valid.sum()),total_frames=len(valid))
        reasons=[]
        if contrast<.05:reasons.append('weak_content_contrast')
        if np.median(diff)>.08 or np.percentile(diff,95)>.20:reasons.append('independent_alignment_disagrees')
        if coverage<.70:reasons.append('pitch_or_voicing_uncertain')
        if valid.sum()<15:reasons.append('too_few_valid_frames')
        row['status']='verified_candidate' if not reasons else 'quarantine'
        row['reasons']=';'.join(reasons)
        if not reasons:
            pack=out/'sequences'/(pair['pair_id']+'.npz')
            np.savez_compressed(pack,time=a['time'],source_f0=a['f0'],source_cc=a['cc'],target_f0=target,
                                valid=valid,professional_time=p['time'],professional_f0=p['f0'],
                                professional_agreement=p['agreement'],mapped_time=mapped)
            row['pack']=str(pack.resolve())
    except Exception as e:row.update(status='error',reasons=str(e))
    return row


def summarize(rows):
    good=[r for r in rows if r['status']=='verified_candidate'];stats=defaultdict(Counter)
    for r in good:
        with np.load(r['pack'],allow_pickle=False) as z:
            v=z['valid'];s=z['source_f0'];t=z['target_f0']
            d=1200*np.log2(t[v]/s[v]);stats[r['split']].update(frames=len(d),raw_large_delta_frames=int((np.abs(d)>=60).sum()))
    return {'processed_pairs':len(rows),'statuses':dict(Counter(r['status'] for r in rows)),
            'reasons':dict(Counter(s for r in rows for s in r.get('reasons','').split(';') if s)),
            'verified_songs':len({r['split_group'] for r in good}),'verified_by_split':dict(Counter(r['split'] for r in good)),
            'frame_counts':{k:dict(v) for k,v in stats.items()},'human_pitch_truth_verified':False}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=128);ap.add_argument('--skip',type=int,default=0)
    args=ap.parse_args();out=new_output(args.out);(out/'sequences').mkdir()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    pairs=[json.loads(s) for s in (PREVIOUS/'alignment_v001/alignment.jsonl').read_text(encoding='utf-8').splitlines()]
    pairs=[p for p in pairs if p['alignment_status']=='auto_screen_candidate']
    pairs.sort(key=lambda p:hashlib.sha256(p['pair_id'].encode()).hexdigest())
    pairs=pairs[args.skip:args.skip+args.limit] if args.limit else pairs[args.skip:]
    dump(out/'run.json',{'limit':args.limit,'skip':args.skip,'pairs':len(pairs),
          'source_previous':str(PREVIOUS),'asr_checkpoint':str(ASR/'model_ckpt_steps_136000.ckpt'),
          'asr_sha256':sha(ASR/'model_ckpt_steps_136000.ckpt'),'torch':torch.__version__,
          'source_hash_rechecked_when_loaded':True,'thresholds':{'mapping_median':.08,'mapping_p95':.20,'pitch_fraction':.70,'min_frames':15}})
    start=time.monotonic();encoder=ContentEncoder();rows=[]
    with (out/'pairs.jsonl').open('w',encoding='utf-8') as f:
        for i,pair in enumerate(pairs,1):
            row=validate_pair(pair,encoder,out);rows.append(row);f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            if i%20==0 or i==len(pairs):
                f.flush();print(json.dumps({'done':i,'total':len(pairs),'statuses':dict(Counter(r['status'] for r in rows)),
                                           'seconds':round(time.monotonic()-start,1)}),flush=True)
    csv_write(out/'pairs.csv',rows);summary=summarize(rows);dump(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
