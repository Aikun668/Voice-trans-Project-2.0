"""Post-processing v2 on saved model outputs (no model re-run). Base: C_target (zuixuan: C_target_oct, i.e. with evidence-based octave repair).
 J_dip   : offset-aware dip repair only          K_full : J + colour match 1.0          L_half : J + colour match 0.5"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf
import common as c, eval_metrics as em
from colour_match import ColourConfig, colour_match
from dip_repair import repair_dips
E = Path(__file__).resolve().parent / 'eval_out'; sr = 22050
rd = lambda n, k: sf.read(str(E / 'audio' / f'{n}__{k}.wav'), dtype='float64')[0]
names = [f.stem for f in sorted(E.glob('*.json')) if f.stem not in ('post_compare',) and 'compare' not in f.stem]
res = {}
for n in names:
    sfx = '_oct' if n == 'my_zuixuan' else ''
    x, c_t, v = rd(n, 'input'), rd(n, 'C_target' + sfx), rd(n, 'vocoder_only' + sfx); k = min(map(len, (x, c_t, v))); x, c_t, v = x[:k], c_t[:k], v[:k]
    j, rep = repair_dips(c_t, v, np.zeros(k // 128 + 1), sr)
    cond = {'J_dip': j}
    for name, s in (('K_full', 1.0), ('L_half', 0.5)): cond[name], _ = colour_match(x, j, sr, ColourConfig(strength=s))
    res[n] = {'dip_report': rep, 'base': em.all_metrics(x, c_t), **{kk: em.all_metrics(x, w) for kk, w in cond.items()}}
    for kk, w in cond.items(): sf.write(E / 'audio' / f'{n}__{kk}.wav', w.astype(np.float32), sr, subtype='FLOAT')
    print(n, 'offset %.1f dB touched %.2f s' % (rep['baseline_offset_db'], rep['seconds_touched']), flush=True)
json.dump(res, open(E / 'post2_compare.json', 'w'), indent=1)
user = [k for k in res if k.startswith('my_')]; off = [k for k in res if not k.startswith('my_')]
print('official: share of clips with dip repair > 0.05 s:', round(np.mean([res[k]['dip_report']['seconds_touched'] > .05 for k in off]), 2), '| user seconds touched:', {k: round(res[k]['dip_report']['seconds_touched'], 2) for k in user})
for label, keys in (('OFFICIAL (n=%d)' % len(off), off), ('USER', user)):
    print('\n' + label); print(f'{"cond":8s} worst_drop secs<-10  colour  HNR   pitch_dev(med/p95)  fast_f0')
    for cond in ('base', 'J_dip', 'K_full', 'L_half'):
        m = lambda key, sub=None: round(float(np.nanmean([(res[k][cond][key][sub] if sub else res[k][cond][key]) for k in keys])), 2)
        print(f'{cond:8s} {m("worst_drop_db"):9} {m("seconds_below_-10db"):9} {m("colour","nasal_norm_db"):7} {m("hnr_median_db"):5} {m("median_dev_cents")}/{m("p95_dev_cents")}  {m("fast_f0_cents")}')
print('\nper user clip (K_full): ', {k: (res[k]['K_full']['worst_drop_db'], res[k]['K_full']['colour']['nasal_norm_db'], res[k]['K_full']['p95_dev_cents']) for k in user})
