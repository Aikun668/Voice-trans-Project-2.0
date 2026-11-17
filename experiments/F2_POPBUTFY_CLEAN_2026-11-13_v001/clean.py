"""Read-only PopBuTFy adapter. Automated candidates are never training truth."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from difflib import SequenceMatcher
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata
import zipfile
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_SOURCE = Path(r'C:\Users\杨玉璋\Downloads\PopBuTFy')
DEFAULT_LABELS = PROJECT/'third_party_neuralsvb/weights_dl/text_labels.zip'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_write(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f: return [json.loads(s) for s in f if s.strip()]


def new_output(path):
    path = Path(path).resolve()
    source = DEFAULT_SOURCE.resolve()
    if path == source or source in path.parents:
        raise ValueError('Outputs must not be inside the original dataset')
    path.mkdir(parents=True, exist_ok=False)
    return path


def normalize_lyric(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    # Apostrophes do not create a token: "I'll" and "ill" match.
    text = text.replace("'", '').replace('’', '')
    return ' '.join(re.sub(r'[^\w\s]', ' ', text).split())


def song_key(text):
    return re.sub(r'[^\w]', '', unicodedata.normalize('NFKC', text).casefold().replace('_', ''))


def match_clips(amateur, professional):
    # Missing labels are excluded BEFORE sequence matching: empty strings are not evidence.
    ai = [i for i,x in enumerate(amateur) if x['lyric_norm']]
    pi = [i for i,x in enumerate(professional) if x['lyric_norm']]
    sm = SequenceMatcher(None, [amateur[i]['lyric_norm'] for i in ai],
                         [professional[j]['lyric_norm'] for j in pi], autojunk=False)
    return [(ai[b.a+k], pi[b.b+k]) for b in sm.get_matching_blocks() for k in range(b.size)]


def song_splits(rows):
    parent = {song_key(r['song']): song_key(r['song']) for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    hashes = {}
    for r in rows:
        k = song_key(r['song']); h = r.get('sha256')
        if h and h in hashes:
            a,b = find(k),find(hashes[h]); parent[max(a,b)] = min(a,b)
        elif h: hashes[h] = k
    result = {}
    for k in parent:
        group = find(k)
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8],16) % 10
        result[k] = {'split_group': group, 'split': 'test' if bucket==0 else 'validation' if bucket==1 else 'train'}
    return result


def inspect_audio(path):
    path = Path(path); before = path.stat(); blob = path.read_bytes()
    result = {'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob),
              'mtime_ns': before.st_mtime_ns, 'file_status': 'ok'}
    try:
        x, sr = sf.read(io.BytesIO(blob), dtype='float32', always_2d=True)
        finite = bool(np.isfinite(x).all())
        peak = float(np.max(np.abs(x))) if x.size and finite else 0.0
        rms = float(np.sqrt(np.mean(x.astype('float64')**2))) if x.size and finite else 0.0
        result.update(sample_rate=sr, channels=x.shape[1], frames=len(x), duration=len(x)/sr,
                      finite=finite, peak=peak, rms_db=20*math.log10(max(rms,1e-12)),
                      near_fullscale_fraction=float(np.mean(np.abs(x)>=0.999)) if x.size else 0.0)
        if not finite: result['file_status'] = 'nonfinite'
        elif len(x)/sr < 0.25: result['file_status'] = 'too_short'
        elif result['rms_db'] < -65: result['file_status'] = 'silent_or_very_quiet'
    except Exception as e:
        result.update(file_status='decode_error', error=str(e))
    after = path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        result['file_status'] = 'source_changed_during_read'
    return result


def audit(source, labels, output, workers):
    started = time.monotonic(); source,labels = Path(source).resolve(),Path(labels).resolve()
    if output.resolve() == source or source in output.resolve().parents: raise ValueError('source is read-only')
    files = sorted(p for p in (source/'data').rglob('*') if p.suffix.lower() in ('.mp3','.wav'))
    with zipfile.ZipFile(labels) as z:
        texts = {Path(n).with_suffix('').as_posix(): z.read(n).decode('utf-8-sig').strip()
                 for n in z.namelist() if n.lower().endswith('.txt')}
    out = new_output(output)
    dump(out/'run.json', {'source':str(source),'labels':str(labels),'labels_sha256':sha(labels),
                          'date_user_timezone':'2026-11-13','training_ready':False,
                          'workers':workers,'source_file_count':len(files)})
    def one(path):
        rel = path.relative_to(source/'data')
        match = re.fullmatch(r'(.+)#singing#(.+)_(Amateur|Professional)_(\d+)', path.stem)
        row = {'path':str(path), 'relative_path':rel.as_posix()}
        if match:
            singer,song,version,index = match.groups()
            row.update(singer=singer,song=song,version=version,index=int(index))
        else: row.update(singer='',song='__unknown__',version='',index=-1)
        text = texts.get(rel.with_suffix('').as_posix(), '')
        row.update(lyric=text,lyric_norm=normalize_lyric(text),label_found=bool(text))
        row.update(inspect_audio(path))
        if not match: row['file_status']='unrecognized_filename'
        return row
    rows=[]
    with ThreadPoolExecutor(max_workers=workers) as pool, (out/'inventory.jsonl').open('w',encoding='utf-8') as f:
        for count,row in enumerate(pool.map(one,files),1):
            rows.append(row); f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            if count%1000==0:
                f.flush(); print(json.dumps({'stage':'decode','done':count,'total':len(files),'seconds':round(time.monotonic()-started,1)}),flush=True)
    splits=song_splits(rows)
    for row in rows: row.update(splits[song_key(row['song'])])
    csv_write(out/'inventory.csv', rows)
    grouped=defaultdict(lambda:defaultdict(list))
    for row in rows: grouped[(row['singer'],row['song'])][row['version']].append(row)
    pairs=[]; used=set(); repeated=Counter()
    for (singer,song),sides in sorted(grouped.items()):
        a=sorted(sides['Amateur'],key=lambda r:r['index']); p=sorted(sides['Professional'],key=lambda r:r['index'])
        ca,cp=Counter(r['lyric_norm'] for r in a),Counter(r['lyric_norm'] for r in p)
        for i,j in match_clips(a,p):
            x,y=a[i],p[j]; reasons=[]
            if x['file_status']!='ok' or y['file_status']!='ok': reasons.append('audio_check_failed')
            if ca[x['lyric_norm']]>1 or cp[y['lyric_norm']]>1: reasons.append('repeated_lyric_occurrence')
            if x['sha256']==y['sha256']: reasons.append('identical_both_versions')
            if max(x.get('near_fullscale_fraction',0),y.get('near_fullscale_fraction',0))>0.01: reasons.append('many_near_fullscale_samples')
            pid=hashlib.sha256((x['relative_path']+'|'+y['relative_path']).encode()).hexdigest()[:20]
            row={'pair_id':pid,'singer':singer,'song':song,'lyric':x['lyric'],'lyric_norm':x['lyric_norm'],
                 'amateur_path':x['path'],'professional_path':y['path'],
                 'amateur_index':x['index'],'professional_index':y['index'],
                 'amateur_sha256':x['sha256'],'professional_sha256':y['sha256'],
                 'amateur_duration':x.get('duration',0),'professional_duration':y.get('duration',0),
                 'same_index':x['index']==y['index'],
                 'audio_ok':x['file_status']=='ok' and y['file_status']=='ok',
                 'review_reasons':';'.join(reasons),**splits[song_key(song)],'training_ready':False}
            pairs.append(row);used.update((x['path'],y['path']))
    unmatched=[dict(r,unmatched_reason='missing_lyric' if not r['lyric_norm'] else 'no_ordered_exact_lyric_match') for r in rows if r['path'] not in used]
    duplicates=defaultdict(list)
    for r in rows: duplicates[r['sha256']].append(r['path'])
    csv_write(out/'pairs.csv',pairs); dump(out/'pairs.json',pairs)
    csv_write(out/'unmatched.csv',unmatched)
    csv_write(out/'audio_issues.csv',[r for r in rows if r['file_status']!='ok' or r.get('near_fullscale_fraction',0)>0.01])
    dump(out/'duplicates.json',{k:v for k,v in duplicates.items() if len(v)>1})
    summary={'source_audio_files':len(rows),'singer_ids':len({r['singer'] for r in rows}),
             'singer_song_groups':len(grouped),'canonical_song_groups':len(splits),
             'file_status':dict(Counter(r['file_status'] for r in rows)),
             'missing_lyrics':sum(not r['label_found'] for r in rows),'lyric_matched_pairs':len(pairs),
             'same_index_pairs':sum(r['same_index'] for r in pairs),
             'recovered_different_index_pairs':sum(not r['same_index'] for r in pairs),
             'unmatched_audio_files':len(unmatched),'pair_review_reasons':dict(Counter(x for r in pairs for x in r['review_reasons'].split(';') if x)),
             'pair_splits':dict(Counter(r['split'] for r in pairs)),'duplicate_hash_groups':sum(len(v)>1 for v in duplicates.values()),
             'seconds':round(time.monotonic()-started,2),'training_ready':False}
    dump(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)


def align_pairs(audit_dir, output, workers, limit):
    # Keep numerical/JIT caches inside this experiment, never inside an old environment.
    os.environ.setdefault('NUMBA_CACHE_DIR',str(HERE/'numba_cache'))
    import librosa
    import scipy
    from scipy.signal import resample_poly
    from scipy.spatial.distance import cdist
    from threadpoolctl import threadpool_limits
    out=new_output(output);(out/'maps').mkdir()
    pairs=[r for r in json.loads((audit_dir/'pairs.json').read_text(encoding='utf-8')) if r['audio_ok']]
    if limit: pairs=pairs[:limit]
    dump(out/'run.json',{'audit_dir':str(audit_dir.resolve()),'selected_pairs':len(pairs),
                         'librosa':librosa.__version__,'scipy':scipy.__version__,'numpy':np.__version__,
                         'hop_seconds':0.02,'features':'MFCC coefficients 1..12, excluding energy',
                         'dtw_steps':[[1,1],[1,2],[2,1]],'training_ready':False})
    def features(path,expected_hash):
        blob=Path(path).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=expected_hash: raise ValueError('source hash changed')
        y,sr=sf.read(io.BytesIO(blob),dtype='float32',always_2d=True); y=y.mean(axis=1)
        g=math.gcd(sr,16000);y=resample_poly(y,16000//g,sr//g)
        f=librosa.feature.mfcc(y=y,sr=16000,n_mfcc=13,n_fft=512,hop_length=320,n_mels=40,fmin=80,fmax=7600)[1:].T
        return f,len(y)/16000
    def path_for(c):
        d,p=librosa.sequence.dtw(C=c,step_sizes_sigma=np.array([[1,1],[1,2],[2,1]]),
                                 global_constraints=True,band_rad=0.35,backtrack=True)
        p=p[::-1]
        return float(c[p[:,0],p[:,1]].mean()),p
    def one(pair):
        row=dict(pair)
        try:
            a,ad=features(pair['amateur_path'],pair['amateur_sha256'])
            p,pd=features(pair['professional_path'],pair['professional_sha256'])
            ratio=pd/ad; row['duration_ratio']=ratio
            if not 0.5<=ratio<=2:
                row.update(alignment_status='duration_ratio_review',training_ready=False);return row
            sd=np.vstack((a,p)).std(axis=0)+1e-5
            c=cdist(a/sd,p/sd)/math.sqrt(a.shape[1])
            score,path=path_for(c)
            reverse,_=path_for(c[:,::-1].copy())
            improvement=(reverse-score)/max(reverse,1e-9)
            # No forced alignment or waveform rewrite. Save native, untrimmed source times.
            at=path[:,0]*0.02;pt=path[:,1]*0.02
            map_path=out/'maps'/(pair['pair_id']+'.npz')
            np.savez_compressed(map_path,amateur_time=at,professional_time=pt,
                                amateur_frame=path[:,0],professional_frame=path[:,1])
            reasons=pair['review_reasons'].split(';') if pair['review_reasons'] else []
            if improvement<0.08: reasons.append('weak_reverse_contrast')
            row.update(alignment_status='needs_review' if reasons else 'auto_screen_candidate',
                       review_reasons=';'.join(reasons),mfcc_cost=score,reverse_cost=reverse,
                       reverse_improvement=improvement,map_path=str(map_path.resolve()),
                       mapped_points=len(path),amateur_mapped_end=float(at[-1]),professional_mapped_end=float(pt[-1]))
        except Exception as e: row.update(alignment_status='alignment_error',error=str(e))
        return row
    started=time.monotonic();rows=[]
    # Warm JIT once before threads; also records the first real example.
    with threadpool_limits(limits=1):
        if pairs: rows.append(one(pairs[0]))
        print(json.dumps({'stage':'align_warmup','seconds':round(time.monotonic()-started,1),'first_status':rows[0].get('alignment_status') if rows else None}),flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool,(out/'alignment.jsonl').open('w',encoding='utf-8') as f:
            if rows: f.write(json.dumps(rows[0],ensure_ascii=False,allow_nan=False)+'\n')
            for count,row in enumerate(pool.map(one,pairs[1:]),2):
                rows.append(row); f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                if count%250==0:
                    f.flush();print(json.dumps({'stage':'align','done':count,'total':len(pairs),'seconds':round(time.monotonic()-started,1)}),flush=True)
    csv_write(out/'alignment.csv',rows)
    summary={'processed_pairs':len(rows),'alignment_status':dict(Counter(r['alignment_status'] for r in rows)),
             'seconds':round(time.monotonic()-started,2),'training_ready':False}
    dump(out/'summary.json',summary);print(json.dumps(summary),flush=True)


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('audit');a.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    a.add_argument('--labels',type=Path,default=DEFAULT_LABELS);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--workers',type=int,default=4)
    b=sub.add_parser('align');b.add_argument('--audit',type=Path,required=True)
    b.add_argument('--out',type=Path,required=True);b.add_argument('--workers',type=int,default=4)
    b.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    if args.mode=='audit':audit(args.source,args.labels,args.out,args.workers)
    else:align_pairs(args.audit,args.out,args.workers,args.limit)


if __name__=='__main__':main()
