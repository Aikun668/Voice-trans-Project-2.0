"""Aggregate eval_out/*.json. Official clips: how close is each condition to the authors' own output? User recordings: how far from the input?"""
import json, sys
from pathlib import Path
import numpy as np
E = Path(__file__).resolve().parent / 'eval_out'
rows = [json.load(open(f, encoding='utf-8')) for f in sorted(E.glob('*.json'))]
off = [r for r in rows if r['split'] != 'user']; usr = [r for r in rows if r['split'] == 'user']
conds = ['A_legacy', 'B_aligned', 'C_target', 'D_dip', 'E_colour', 'vocoder_only']
def agg(rs, cond, key, sub=None):
    v = [(r[cond][key][sub] if sub else r[cond][key]) for r in rs if cond in r]; return round(float(np.nanmean(v)), 2) if v else None
def table(rs, label):
    print(f'\n=== {label}  (n={len(rs)})'); print(f'{"condition":13s} worst_drop  secs<-10dB  colour_dist  HNR   jitter  shimmer  pitch_dev(med/p95)  fast_f0')
    for cd in conds + (['v002_a2p'] if rs and 'v002_a2p' in rs[0] else []):
        if not all(cd in r for r in rs): continue
        print(f'{cd:13s} {agg(rs,cd,"worst_drop_db"):9}  {agg(rs,cd,"seconds_below_-10db"):9}  {agg(rs,cd,"colour","nasal_norm_db"):10}  {agg(rs,cd,"hnr_median_db"):5} {agg(rs,cd,"jitter"):7} {agg(rs,cd,"shimmer"):7}  {agg(rs,cd,"median_dev_cents")}/{agg(rs,cd,"p95_dev_cents")}  {agg(rs,cd,"fast_f0_cents")}')
    if rs and 'official' in rs[0]:
        o = lambda k, s=None: round(float(np.nanmean([(r['official'][k][s] if s else r['official'][k]) for r in rs])), 2)
        print(f'{"official":13s} {"-":9}  {"-":9}  {o("colour","nasal_norm_db"):10}  {o("hnr_median_db"):5} {o("jitter"):7} {o("shimmer"):7}  {"-":>10}  {o("fast_f0_cents")}')
for sp in ('dev', 'test'):
    rs = [r for r in off if r['split'] == sp]
    if rs: table(rs, f'official clips, split={sp}')
if usr:
    for r in usr: table([r], f'user recording {r["name"]}')
    table(usr, 'user recordings, mean')
print('\ntarget repair totals (official):', {k: int(sum(r['target_report'].get(k, 0) for r in off)) for k in ('octave_fixed_frames', 'spike_frames', 'gap_filled_frames')})
print('target repair totals (user):', {r['name']: r['target_report'] for r in usr})
