"""Independent artifact checks; no import of the cleaning implementation."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def verify(audit,alignment,out):
    if out.exists():raise FileExistsError(out)
    inv=list(csv.DictReader((audit/'inventory.csv').open(encoding='utf-8-sig',newline='')))
    pairs=json.loads((audit/'pairs.json').read_text(encoding='utf-8'))
    run=json.loads((audit/'run.json').read_text(encoding='utf-8'))
    rows=[json.loads(x) for x in (alignment/'alignment.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(inv)==run['source_file_count']
    actual={str(p.resolve()) for p in (Path(run['source'])/'data').rglob('*') if p.suffix.lower() in ('.mp3','.wav')}
    assert actual=={r['path'] for r in inv},'inventory coverage changed'
    assert len({r['path'] for r in inv})==len(inv)
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes=list(pool.map(lambda r:digest(r['path']),inv))
    mismatches=[r['path'] for r,h in zip(inv,hashes) if r['sha256']!=h]
    assert not mismatches,'source file hash changed'
    assert digest(run['labels'])==run['labels_sha256']
    bypath={r['path']:r for r in inv};used=set();group_splits=defaultdict(set);hash_splits=defaultdict(set)
    for r in inv:
        group_splits[r['split_group']].add(r['split']);hash_splits[r['sha256']].add(r['split'])
    assert all(len(s)==1 for s in group_splits.values()),'song leakage'
    assert all(len(s)==1 for s in hash_splits.values()),'duplicate audio leakage'
    for p in pairs:
        a,b=bypath[p['amateur_path']],bypath[p['professional_path']]
        assert a['lyric_norm'] and a['lyric_norm']==b['lyric_norm']==p['lyric_norm']
        assert a['version']=='Amateur' and b['version']=='Professional'
        assert a['singer']==b['singer'] and a['song']==b['song']
        assert a['split']==b['split']==p['split']
        assert a['path'] not in used and b['path'] not in used,'clip reused across pairs'
        used.update((a['path'],b['path']))
        assert p['training_ready'] is False
    selected={p['pair_id'] for p in pairs if p['audio_ok']}
    assert {r['pair_id'] for r in rows}==selected,'not all eligible pairs processed'
    assert len(rows)==len(selected),'duplicate alignment rows'
    maps=0
    for row in rows:
        if not row.get('map_path'):continue
        with np.load(row['map_path'],allow_pickle=False) as m:
            a,b=m['amateur_time'],m['professional_time'];af,bf=m['amateur_frame'],m['professional_frame']
            assert len(a)==len(b)==row['mapped_points'] and len(a)>1
            assert np.isfinite(a).all() and np.isfinite(b).all()
            assert a[0]==b[0]==0
            assert (np.diff(a)>0).all() and (np.diff(b)>0).all(),'nonmonotonic map'
            assert np.diff(a).max()<=0.040001 and np.diff(b).max()<=0.040001
            assert np.allclose(a,af*0.02) and np.allclose(b,bf*0.02)
            assert a[-1]<=float(bypath[row['amateur_path']]['duration'])+0.001
            assert b[-1]<=float(bypath[row['professional_path']]['duration'])+0.001
        maps+=1
    assert maps==len(list((alignment/'maps').glob('*.npz'))),'orphan or missing maps'
    result={'technical_pass':True,'inventory_rows':len(inv),'source_hashes_rechecked':len(inv),
            'source_hash_mismatches':mismatches,'label_archive_hash_unchanged':True,
            'unique_lyric_pairs':len(pairs),'alignment_rows':len(rows),'time_maps_checked':maps,
            'alignment_status':dict(Counter(r['alignment_status'] for r in rows)),
            'cross_song_or_duplicate_split_leakage':False,'training_ready':False,
            'semantic_alignment_and_pitch_targets':'pending_review','listening':'pending_jade',
            'limitation':'Checks prove traceability and numerical integrity, not correct lyric/audio alignment or pitch targets.'}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--alignment',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();verify(args.audit,args.alignment,args.out)
