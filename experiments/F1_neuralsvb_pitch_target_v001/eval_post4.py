"""Final candidates on saved model outputs: Q_full = level_match + voiced-only colour match 1.0 ; R_half = level_match + colour 0.5.
Reports the energy tilt on sung frames before/after (relative to the 1.5-4k change; target 0 / 0 / 0) and exports through the project checks."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf, librosa
import common as c, eval_metrics as em
from colour_match import ColourConfig, colour_match
from level_match import level_match
sys.path.insert(0, str(c.HERE.parent / 'F1_neuralsvb_contract_v001')); import feature_contract as fc
E = c.HERE / 'eval_out'; sr = 22050; fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
rd = lambda n, k: sf.read(str(E / 'audio' / f'{n}__{k}.wav'), dtype='float64')[0]
st = lambda y: np.abs(librosa.stft(y, n_fft=2048, hop_length=128)) ** 2
def band(y, f0, lo, hi):
    P = st(y); m = min(P.shape[1], len(f0)); v = f0[:m] > 0; return 10 * np.log10(P[(fr >= lo) & (fr < hi), :m][:, v].sum() + 1e-12)
def tilt(x, y, f0): ref = band(y, f0, 1500, 4000) - band(x, f0, 1500, 4000); return [round(band(y, f0, lo, hi) - band(x, f0, lo, hi) - ref, 1) for lo, hi in ((100, 300), (300, 1500), (4000, 8000))]
for n in ('my_shige', 'my_zuixuan', 'my_molihua'):
    sfx = '_oct' if n == 'my_zuixuan' else ''
    x, c_t = rd(n, 'input'), rd(n, 'C_target' + sfx); k = min(len(x), len(c_t)); x, c_t = x[:k], c_t[:k]
    f0, _ = fc.aligned_f0(x, k // 128, frame_offset_seconds=-128 / sr)
    m, lrep = level_match(x, c_t, sr); out = {}
    out['Q_full'], rq = colour_match(x, m, sr, ColourConfig(strength=1.0), voiced_f0=f0); out['R_half'], rr = colour_match(x, m, sr, ColourConfig(strength=0.5), voiced_f0=f0)
    for kk, w in out.items(): sf.write(E / 'audio' / f'{n}__{kk}.wav', w.astype(np.float32), sr, subtype='FLOAT')
    print(n, 'tilt(Low,Mid1,High) raw model', tilt(x, c_t, f0), '| level only', tilt(x, m, f0), '| Q_full', tilt(x, out['Q_full'], f0), '| R_half', tilt(x, out['R_half'], f0), '| colour gains', rq['band_gain_db'], flush=True)
print('DONE')
