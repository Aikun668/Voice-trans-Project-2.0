"""Energy tilt vs time on voiced frames: per 2 s window, band energy change of C_target vs input, relative to the 1.5-4k change (dB).
Low=100-300 Hz, Mid1=300-1500, High=4-8k. A rising trend would mean one clip-wide EQ cannot fix it."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf, librosa
import eval_metrics as em
E = Path('eval_out/audio'); sr = 22050
rd = lambda n, k: sf.read(str(E / f'{n}__{k}.wav'), dtype='float64')[0]
st = lambda y: np.abs(librosa.stft(y, n_fft=2048, hop_length=128)) ** 2
fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
for n in ('my_shige', 'my_zuixuan', 'my_molihua'):
    sfx = '_oct' if n == 'my_zuixuan' else ''
    x = rd(n, 'input'); y = rd(n, 'C_target' + sfx)[:len(x)]; x = x[:len(y)]; f0 = em.praat_pitch(x); Px, Py = st(x), st(y); m = min(Px.shape[1], Py.shape[1], len(f0)); v = f0[:m] > 0
    print(n, ' window start | Low(100-300)  Mid1(300-1500)  High(4-8k)  relative to 1.5-4k (dB; negative = output has less than input)')
    for t in range(0, int(len(x) / sr) - 1, 3):
        i0, i1 = int(t * sr / 128), int((t + 3) * sr / 128); sel = v[i0:min(i1, m)]
        if sel.sum() < 50: continue
        def d(lo, hi): 
            a = Px[(fr >= lo) & (fr < hi), i0:min(i1, m)][:, sel].sum(); b = Py[(fr >= lo) & (fr < hi), i0:min(i1, m)][:, sel].sum(); return 10 * np.log10(b / a)
        ref = d(1500, 4000); print(f'   {t:4d}s  | {d(100,300)-ref:+5.1f}  {d(300,1500)-ref:+5.1f}  {d(4000,8000)-ref:+5.1f}')
