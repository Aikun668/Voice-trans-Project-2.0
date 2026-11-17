"""Tilt (energy-weighted) vs harmonic valley filling. Voiced frames only (Praat F0 of the input, aligned), 1.5-4 kHz and 0.3-1.5 kHz.
For each frame: harmonic peaks = max in +-8% of k*f0; valleys = min within +-12% around (k-0.5)*f0.  valley_depth_dB = mean(peak dB - valley dB).
Energy tilt = band power change (dB) of output vs input over the same frames. Measurements only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf, librosa
import eval_metrics as em
E = Path('eval_out/audio'); sr = 22050
rd = lambda n, k: sf.read(str(E / f'{n}__{k}.wav'), dtype='float64')[0]
def stft(y): return np.abs(librosa.stft(y, n_fft=2048, hop_length=128, center=True)) ** 2, librosa.fft_frequencies(sr=sr, n_fft=2048)
def depth(P, fr, f0, lo, hi):
    out = []
    for i in range(P.shape[1]):
        f = f0[i] if i < len(f0) else 0
        if f < 80: continue
        vals = []
        for k in range(1, 60):
            fk = k * f
            if fk < lo or fk > hi: continue
            pk = P[(np.abs(fr - fk) < 0.08 * f), i].max(); mid = (k + .5) * f; vl = P[(np.abs(fr - mid) < 0.12 * f), i].min()
            vals.append(10 * np.log10((pk + 1e-12) / (vl + 1e-12)))
        if vals: out.append(np.mean(vals))
    return float(np.mean(out)) if out else float('nan')
for n in ('my_shige', 'my_zuixuan', 'my_molihua'):
    sfx = '_oct' if n == 'my_zuixuan' else ''
    x = rd(n, 'input'); f0 = em.praat_pitch(x); Px, fr = stft(x); res = {}
    for k in ('input', 'vocoder_only' + sfx, 'C_target' + sfx, 'N_level_full'):
        y = x if k == 'input' else rd(n, k)[:len(x)]; P, _ = stft(y); m = min(P.shape[1], Px.shape[1]); P = P[:, :m]
        voiced = np.zeros(m, bool); voiced[:min(m, len(f0))] = f0[:m] > 0; band = lambda lo, hi: 10 * np.log10(P[(fr >= lo) & (fr < hi)][:, voiced].sum() + 1e-12)
        res[k] = {'E<0.3k': band(100, 300), 'E.3-1.5k': band(300, 1500), 'E1.5-4k': band(1500, 4000), 'E4-8k': band(4000, 8000), 'valley_1.5-4k': depth(P[:, :m], fr, f0, 1500, 4000), 'valley_.3-1.5k': depth(P[:, :m], fr, f0, 300, 1500)}
    base = res['input']
    print(n); 
    for k, v in res.items():
        print('   %-14s energy change vs input (dB): <0.3k %+5.1f | .3-1.5k %+5.1f | 1.5-4k %+5.1f | 4-8k %+5.1f || valley depth dB: .3-1.5k %.1f  1.5-4k %.1f' % (k, v['E<0.3k'] - base['E<0.3k'], v['E.3-1.5k'] - base['E.3-1.5k'], v['E1.5-4k'] - base['E1.5-4k'], v['E4-8k'] - base['E4-8k'], v['valley_.3-1.5k'], v['valley_1.5-4k']))
