"""Octave jumps in zuixuan candA: who is wrong - the F0 given to the model, the original, or the output? pYIN as independent judge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf, librosa
import common as c, eval_metrics as em
E = Path('eval_out/audio'); sr = 22050
rd = lambda k: sf.read(str(E / f'my_zuixuan__{k}.wav'), dtype='float64')[0]
x, y = rd('input'), rd('H_full'); n = min(len(x), len(y)); x, y = x[:n], y[:n]
fx, fy = em.praat_pitch(x), em.praat_pitch(y)
def pyin(w):
    f, v, _ = librosa.pyin(w.astype(np.float32), fmin=75, fmax=800, sr=sr, frame_length=2048, hop_length=128); return np.where(v, f, 0)
px, py = pyin(x), pyin(y)
for lo, hi in [(2.10, 2.46), (2.96, 3.18), (9.70, 10.06), (12.20, 12.30), (8.60, 8.70)]:
    print(f'--- {lo}-{hi}s   t | praat_in praat_out | pyin_in pyin_out')
    for t in np.arange(lo, hi, 0.04):
        i = int(t * sr / 128); print(f'{t:5.2f} | {fx[i]:6.0f} {fy[i]:6.0f} | {px[i]:6.0f} {py[i]:6.0f}')
