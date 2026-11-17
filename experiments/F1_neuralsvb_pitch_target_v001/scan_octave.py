"""Scan a recording for stretches where Praat (what the model receives) and pYIN disagree by an octave, and decide with a harmonic-comb test
on the ORIGINAL waveform (mean log-magnitude at k*f minus at (k-0.5)*f, k=1..8; higher = that f really is the fundamental). Read-only."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf, librosa
import common as c, eval_metrics as em
name = sys.argv[1]; sr = 22050
x = sf.read(str(Path('eval_out/audio') / f'{name}__input.wav'), dtype='float64')[0]
fp = em.praat_pitch(x)
yp, vf, _ = librosa.pyin(x.astype(np.float32), fmin=75, fmax=800, sr=sr, frame_length=2048, hop_length=128); py = np.where(vf, yp, 0)
n = min(len(fp), len(py))
def comb(f, t, win=0.05):
    i = int(t * sr); h = int(win * sr / 2); seg = x[max(i - h, 0):i + h]
    if len(seg) < 256 or f <= 0: return np.nan
    seg = seg * np.hanning(len(seg)); N = 1 << 15; S = np.abs(np.fft.rfft(seg, N)); fr = np.fft.rfftfreq(N, 1 / sr)
    g = lambda ff: np.log(S[np.abs(fr - ff) < 0.03 * ff].max() + 1e-9)
    ks = [k for k in range(1, 9) if k * f < 9000]
    return float(np.mean([g(k * f) - g((k - .5) * f) for k in ks]))
flag = np.zeros(n, bool); verdict = {}
for i in range(n):
    if fp[i] > 0 and py[i] > 0:
        r = np.log2(py[i] / fp[i])
        if abs(abs(r) - 1) < 0.12: flag[i] = True
i = 0; segs = []
while i < n:
    if flag[i]:
        j = i
        while j + 1 < n and (flag[j + 1] or flag[j + 2 if j + 2 < n else j + 1]): j += 1
        if j - i + 1 >= 3:
            ts = np.arange(i, j + 1, 2) * 128 / sr; lo = np.nanmedian([fp[k] for k in range(i, j + 1)]); hi = np.nanmedian([py[k] for k in range(i, j + 1)])
            f_lo, f_hi = (min(lo, hi), max(lo, hi))
            s_lo = np.nanmean([comb(f_lo, t) for t in ts]); s_hi = np.nanmean([comb(f_hi, t) for t in ts])
            praat_is_low = lo < hi
            segs.append({'t0': round(i * 128 / sr, 2), 't1': round(j * 128 / sr, 2), 'frames': j - i + 1, 'praat_hz': round(float(lo if praat_is_low else hi) if True else 0, 0),
                         'praat': round(float(np.nanmedian([fp[k] for k in range(i, j + 1)])), 0), 'pyin': round(float(np.nanmedian([py[k] for k in range(i, j + 1)])), 0),
                         'comb_low_f': round(float(s_lo), 2), 'comb_high_f': round(float(s_hi), 2),
                         'winner': ('Praat correct' if (praat_is_low == (s_lo >= s_hi)) else 'pYIN correct (Praat octave error)')})
        i = j + 1
    else: i += 1
print(name, 'flagged octave-disagreement stretches:', len(segs), 'frames', int(sum(s['frames'] for s in segs)))
for s in segs: print(' ', s['t0'], '-', s['t1'], 'praat', s['praat'], 'pyin', s['pyin'], '| comb @low %.2f @high %.2f' % (s['comb_low_f'], s['comb_high_f']), '=>', s['winner'])
json.dump(segs, open(f'octave_scan_{name}.json', 'w'), indent=1)
