"""Construct supervised labels separately from strictly own-audio model features."""
import argparse
from collections import defaultdict,Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
import torch
from trial import HERE,PROJECT,own_features,new_output,dump,sha


def content_groups():
    inventory=PROJECT/'experiments/F2_POPBUTFY_CLEAN_2026-11-13_v001/audit_v001/inventory.csv'
    rows=list(csv.DictReader(inventory.open(encoding='utf-8-sig',newline='')))
    texts=defaultdict(set)
    for r in rows:
        if len(r['lyric_norm'].split())>=4:texts[r['split_group']].add(r['lyric_norm'])
    parent={g:g for g in texts}
    def find(g):
        while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
        return g
    keys=sorted(texts);aliases=[]
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            common=len(texts[a]&texts[b]);ratio=common/max(1,min(len(texts[a]),len(texts[b])))
            if common>=6 and ratio>=.7:
                x,y=find(a),find(b);parent[max(x,y)]=min(x,y)
                aliases.append({'a':a,'b':b,'common_long_lines':common,'containment':ratio})
    result={}
    for g in keys:
        root=find(g);bucket=int(hashlib.sha256(root.encode()).hexdigest()[:8],16)%10
        result[g]={'split_group':root,'split':'test' if bucket==0 else 'validation' if bucket==1 else 'train'}
    return result,aliases


def midi(f):
    v=f>0
    if not v.any():return np.zeros(len(f))
    m=69+12*np.log2(np.maximum(f,1)/440)
    return np.interp(np.arange(len(f)),np.flatnonzero(v),m[v])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audits',type=Path,nargs='+',required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=new_output(args.out);allrows=[]
    for folder in args.audits:
        if not (folder/'summary.json').exists():raise ValueError(f'Incomplete audit: {folder}')
        allrows.extend(json.loads(s) for s in (folder/'pairs.jsonl').read_text(encoding='utf-8').splitlines())
    assert len({r['pair_id'] for r in allrows})==len(allrows),'overlapping audits'
    groups,aliases=content_groups();dump(out/'song_aliases.json',aliases)
    eligible=[r for r in allrows if r['status']=='verified_candidate']
    raw=[];song_deltas=defaultdict(list)
    for r in eligible:
        with np.load(io.BytesIO(Path(r['pack']).read_bytes()),allow_pickle=False) as z:
            s=z['source_f0'].copy();t=z['target_f0'].copy();valid=z['valid'].copy()
            source_m=midi(s);target_m=midi(t)
            delta=100*(median_filter(target_m,9,mode='nearest')-median_filter(source_m,9,mode='nearest'))
            key=(r['singer'],r['song']);song_deltas[key].extend(delta[valid].tolist())
            raw.append((r,s,t,valid,delta))
    # Only remove whole-song integer transposition in supervision, never change source key at inference.
    offsets={k:float(100*np.rint(np.median(v)/100)) for k,v in song_deltas.items()}
    samples=[];counts=defaultdict(Counter);songsets=defaultdict(set);label_rows=[]
    for r,s,t,valid,delta in raw:
        group=groups[r['split_group']];offset=offsets[(r['singer'],r['song'])]
        y=delta-offset
        valid=valid&(np.abs(y)<=300)&(np.abs(np.gradient(midi(s)))<1.0)
        split=group['split'];n=int(valid.sum())
        row={'pair_id':r['pair_id'],**group,'singer':r['singer'],'song':r['song'],
             'song_key_offset_cents':offset,'valid_frames':n,'nonzero_frames':int((valid&(np.abs(y)>=60)).sum()),
             'normal_frames':int((valid&(np.abs(y)<=25)).sum()),'pack':r['pack']}
        label_rows.append(row)
        if n<15:continue
        samples.append({'id':r['pair_id'],'kind':'real_pair','x':own_features(s),
                        'y':y.astype(np.float32),'mask':valid,'weight':1.,'split':split,**group})
        # A professional/normal recording also teaches the identity decision.
        samples.append({'id':r['pair_id']+'-identity','kind':'normal_identity','x':own_features(t),
                        'y':np.zeros(len(t),dtype=np.float32),'mask':t>0,'weight':.35,'split':split,**group})
        counts[split].update(pairs=1,valid_frames=n,nonzero_frames=row['nonzero_frames'],normal_frames=row['normal_frames'])
        songsets[split].add(group['split_group'])
    gate=(counts['train']['pairs']>=300 and counts['train']['valid_frames']>=20000 and counts['train']['nonzero_frames']>=2000
          and len(songsets['train'])>=30 and counts['validation']['valid_frames']>=1000
          and counts['validation']['nonzero_frames']>=100 and len(songsets['validation'])>=5
          and counts['test']['valid_frames']>=1000 and counts['test']['nonzero_frames']>=100 and len(songsets['test'])>=5)
    assert not(songsets['train']&songsets['validation'] or songsets['train']&songsets['test'] or songsets['validation']&songsets['test'])
    report={'audit_pairs':len(allrows),'audit_statuses':dict(Counter(r['status'] for r in allrows)),
            'trial_data_gate_pass':gate,'counts':{k:dict(v) for k,v in counts.items()},
            'song_counts':{k:len(v) for k,v in songsets.items()},'alias_links_merged':len(aliases),
            'supervision':'machine cross-validated PPG/MFCC alignments, dual pitch estimator agreement; not human pitch truth',
            'label_smoothing_seconds':.18,'source_context_features':19,'reference_features_at_inference':0,
            'whole_song_integer_key_offset_removed_in_labels':True,'synthetic_training_examples':0,
            'readiness':'experimental_training_only' if gate else 'gate_failed'}
    dump(out/'gate.json',report);dump(out/'label_manifest.json',label_rows)
    if gate:torch.save(samples,out/'dataset.pt')
    print(json.dumps(report,ensure_ascii=False))
    if not gate:raise SystemExit(2)


if __name__=='__main__':main()
