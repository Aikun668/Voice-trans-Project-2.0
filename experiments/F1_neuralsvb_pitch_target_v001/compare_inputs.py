"""Input-signal properties of the 38 official amateur takes vs the user's 3 recordings (all at 22.05 kHz, as the model sees them). Measurements only."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf, librosa, parselmouth
from parselmouth.praat import call
import eval_metrics as em
E = Path('eval_out/audio'); sr = 22050; fr = librosa.fft_frequencies(sr=sr, n_fft=1024)
rows = {}
for p in sorted(E.glob('*__input.wav')):
    n = p.name.split('__')[0]; y = sf.read(str(p), dtype='float64')[0]
    s = parselmouth.Sound(y, sr); h = s.to_harmonicity_cc(time_step=0.01, minimum_pitch=75).values[0]; h = h[h > -200]
    pp = call(s, 'To PointProcess (periodic, cc)', 80, 750); f0 = em.praat_pitch(y); v = f0 > 0
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=128)) ** 2; m = min(S.shape[1], len(v)); vm = v[:m]
    tot = S[:, :m][:, vm].sum(); hf = S[(fr >= 4000), :m][:, vm].sum() / tot
    L = 10 * np.log10(S.sum(0) + 1e-12); dr = float(np.percentile(L, 99) - np.percentile(L, 5))
    flat = float(np.mean(librosa.feature.spectral_flatness(S=np.sqrt(S[(fr >= 1000) & (fr < 4000), :m][:, vm])))) if vm.any() else np.nan
    rows[n] = dict(hnr=float(np.median(h)), hnr_p10=float(np.percentile(h, 10)), jitter=float(call(pp, 'Get jitter (local)', 0, 0, 1e-4, .02, 1.3)), shimmer=float(call([s, pp], 'Get shimmer (local)', 0, 0, 1e-4, .02, 1.3, 1.6)),
                   f0_med=float(np.median(f0[v])), hf_share_db=float(10 * np.log10(hf + 1e-12)), dyn_range_db=dr, flatness_1_4k=flat, rms_dbfs=float(20 * np.log10(np.sqrt(np.mean(y ** 2)))), seconds=len(y) / sr)
json.dump(rows, open('input_compare.json', 'w'), indent=1)
off = {k: v for k, v in rows.items() if not k.startswith('my_')}; usr = {k: v for k, v in rows.items() if k.startswith('my_')}
keys = ['hnr', 'hnr_p10', 'jitter', 'shimmer', 'f0_med', 'hf_share_db', 'dyn_range_db', 'flatness_1_4k', 'rms_dbfs']
print('metric          official: min  p10   median  p90   max   | user: molihua  shige  zuixuan')
for k in keys:
    a = np.array([v[k] for v in off.values()]); print(f'{k:14s}  {a.min():8.3f} {np.percentile(a,10):7.3f} {np.median(a):7.3f} {np.percentile(a,90):7.3f} {a.max():8.3f} | ' + '  '.join(f'{usr[u][k]:8.3f}' for u in ('my_molihua', 'my_shige', 'my_zuixuan')))
print('\nofficial clips with HNR median below 21 (roughest inputs):', sorted([(round(v['hnr'], 1), k, round(v['jitter'], 4), round(v['shimmer'], 3)) for k, v in off.items() if v['hnr'] < 21]))
print('official clips ranked by shimmer (top 6):', sorted([(round(v['shimmer'], 3), k, round(v['hnr'], 1)) for k, v in off.items()], reverse=True)[:6])
