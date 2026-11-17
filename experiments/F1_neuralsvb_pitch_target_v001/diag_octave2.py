import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import f1_poc as qa
R = Path('delivery_v3/zuixuan_normal')
o, sr = qa.read_pcm16(R / 'zuixuan_normal__original.wav'); y, _ = qa.read_pcm16(R / 'zuixuan_normal__candA_full.wav'); v, _ = qa.read_pcm16(R / 'zuixuan_normal__old_v002.wav')
po, py, pv = qa.estimate_pitch(o, sr), qa.estimate_pitch(y, sr), qa.estimate_pitch(v, sr)
print('project estimator (20 ms frames): t | original  candA  old_v002  (Hz)')
for t in (2.16, 2.30, 2.42, 3.02, 3.08, 3.14, 5.12, 9.72, 9.78, 9.84, 9.96, 10.02, 12.24):
    i = int(round(t / 0.02)); print(f'{t:5.2f} | {po[i]:7.1f} {py[i]:7.1f} {pv[i]:7.1f}')
n = min(len(po), len(py)); ok = ~np.isnan(po[:n]) & ~np.isnan(py[:n]); r = py[:n][ok] / po[:n][ok]
oct_ = (np.abs(np.log2(r) - 1) < .12) | (np.abs(np.log2(r) + 1) < .12)
print('frames compared', ok.sum(), '| ratio ~2 or ~0.5 (octave):', int(oct_.sum()), '| other >35c:', int(((1200 * np.abs(np.log2(r)) > 35) & ~oct_).sum()))
print('of octave frames: output reads HIGHER than original:', int((np.log2(r)[oct_] > 0).sum()), ' LOWER:', int((np.log2(r)[oct_] < 0).sum()))
