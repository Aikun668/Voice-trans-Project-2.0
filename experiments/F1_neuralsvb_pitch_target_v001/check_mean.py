"""Sampled latent vs posterior-mean latent (existing contract_v001 outputs). Praat HNR / jitter / shimmer on voiced stretches. Measurements only."""
import numpy as np, parselmouth, soundfile as sf
from parselmouth.praat import call
from pathlib import Path
base = Path(__file__).resolve().parents[1] / 'F1_neuralsvb_contract_v001/full_normal'
def m(path):
    y, s = sf.read(str(path)); s_ = parselmouth.Sound(y.astype(float), s)
    h = s_.to_harmonicity_cc(time_step=0.01, minimum_pitch=75).values[0]; h = h[h > -200]
    pp = call(s_, 'To PointProcess (periodic, cc)', 80, 750)
    return round(float(np.median(h)), 1), round(float(np.percentile(h, 10)), 1), round(call(pp, 'Get jitter (local)', 0, 0, 1e-4, .02, 1.3), 4), round(call([s_, pp], 'Get shimmer (local)', 0, 0, 1e-4, .02, 1.3, 1.6), 4), len(h)
print('clip / variant : HNR median, HNR p10, jitter, shimmer, voiced frames')
for clip in ('shige_normal', 'zuixuan_normal', 'molihua_normal'):
    for v in ('original', 'vocoder', 'a2a', 'a2a_mean', 'a2p', 'a2p_mean'):
        p = base / clip / f'{clip}__{v}.wav'
        if p.exists(): print(f'{clip:15s} {v:9s}', m(p))
