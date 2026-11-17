"""Post-processing comparison on the saved model outputs (no model re-run). Conditions all start from C_target (new axis + repaired target).
 G_dip_all   : dip repair on every frame
 H_full      : G + colour match strength 1.0
 I_half      : G + colour match strength 0.5"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf
import common as c, eval_metrics as em
from colour_match import ColourConfig, colour_match
from dip_repair import repair_dips
E = Path(__file__).resolve().parent / 'eval_out'; sr = 22050
rd = lambda n, k: sf.read(str(E / 'audio' / f'{n}__{k}.wav'), dtype='float64')[0]
res = {}
for f in sorted(E.glob('*.json')):
    n = f.stem; x, c_t, v = rd(n, 'input'), rd(n, 'C_target'), rd(n, 'vocoder_only'); k = min(map(len, (x, c_t, v))); x, c_t, v = x[:k], c_t[:k], v[:k]
    g, rep = repair_dips(c_t, v, np.zeros(k // 128 + 1), sr)
    cond = {'G_dip_all': g}
    for name, s in (('H_full', 1.0), ('I_half', 0.5)):
        cond[name], _ = colour_match(x, g, sr, ColourConfig(strength=s))
    res[n] = {'dip_all_report': rep, **{kk: em.all_metrics(x, w) for kk, w in cond.items()}}
    for kk, w in cond.items(): sf.write(E / 'audio' / f'{n}__{kk}.wav', w.astype(np.float32), sr, subtype='FLOAT')
    print(n, 'touched s', round(rep['seconds_touched'], 2), flush=True)
json.dump(res, open(E / 'post_compare.json', 'w'), indent=1)
user = [k for k in res if k.startswith('my_')]; off = [k for k in res if not k.startswith('my_')]
def agg(keys, cond, key, sub=None): return round(float(np.nanmean([(res[k][cond][key][sub] if sub else res[k][cond][key]) for k in keys])), 2)
base = {json.load(open(E / f'{k}.json'))['name']: json.load(open(E / f'{k}.json')) for k in res}
print('\nfrac clips with dip repair touching > 0.05 s: official', round(np.mean([res[k]['dip_all_report']['seconds_touched'] > 0.05 for k in off]), 2), '| user', [round(res[k]['dip_all_report']['seconds_touched'], 2) for k in user])
for label, keys in (('OFFICIAL (n=%d)' % len(off), off), ('USER (n=3)', user)):
    print('\n' + label); print(f'{"cond":12s} worst_drop secs<-10  colour  HNR   pitch_dev(med/p95)  fast_f0')
    for cond in ('C_target', 'D_dip', 'G_dip_all', 'H_full', 'I_half', 'vocoder_only'):
        src = lambda k: base[k][cond] if cond in base[k] else res[k][cond]
        m = lambda key, sub=None: round(float(np.nanmean([(src(k)[key][sub] if sub else src(k)[key]) for k in keys])), 2)
        print(f'{cond:12s} {m("worst_drop_db"):9} {m("seconds_below_-10db"):9} {m("colour","nasal_norm_db"):7} {m("hnr_median_db"):5} {m("median_dev_cents")}/{m("p95_dev_cents")}  {m("fast_f0_cents")}')
