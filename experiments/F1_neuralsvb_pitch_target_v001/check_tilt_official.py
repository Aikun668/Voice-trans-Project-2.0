"""Energy tilt (voiced frames, relative to the 1.5-4 kHz change) on official clips: authors' NSVB output vs input, our single-WAV C_target vs input, and the professional take vs input."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf, librosa
import common as c, eval_metrics as em
sr = 22050; fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
st = lambda y: np.abs(librosa.stft(y, n_fft=2048, hop_length=128)) ** 2
def band(y, lo, hi):
    f0 = em.praat_pitch(y); P = st(y); m = min(P.shape[1], len(f0)); v = f0[:m] > 0
    return 10 * np.log10(P[(fr >= lo) & (fr < hi), :m][:, v].sum() + 1e-12)
def tilt(x, y):
    return [band(y, lo, hi) - band(x, lo, hi) - (band(y, 1500, 4000) - band(x, 1500, 4000)) for lo, hi in ((100, 300), (300, 1500), (4000, 8000))]
rd = lambda p: sf.read(str(p), dtype='float64')[0]
rows = {'authors_NSVB': [], 'ours_single_C': [], 'professional_take': []}
for tag, cid in c.all_clips():
    d = c.LANGS[tag]; x = rd(c.clip_path(d, 'gt_a', cid)); a = tilt(x, rd(c.clip_path(d, 'NSVB', cid))); p = tilt(x, rd(c.clip_path(d, 'gt_p', cid)))
    o = tilt(x, rd(c.HERE / 'eval_out/audio' / f'{tag}{cid}__C_target.wav'))
    rows['authors_NSVB'].append(a); rows['ours_single_C'].append(o); rows['professional_take'].append(p)
print('mean over %d official clips (dB, relative to the 1.5-4k change): Low(100-300) Mid1(300-1500) High(4-8k)' % len(rows['authors_NSVB']))
for k, v in rows.items(): print(f'  {k:18s}', np.round(np.mean(v, 0), 1), ' std', np.round(np.std(v, 0), 1))
