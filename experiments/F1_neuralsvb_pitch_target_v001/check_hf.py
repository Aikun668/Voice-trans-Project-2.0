"""High-frequency excess in the *audio*: output vs input level per band, active voiced frames only. Measurements only.
Uses the already-generated vocoder-only / a2p outputs (v002 restored for the user's recordings; ours_fixed/ours for official clips)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf, librosa, common as c
EDGES = [(0, 500), (500, 1500), (1500, 4000), (4000, 6000), (6000, 8000), (8000, 11025)]
def spec(path):
    y, s = sf.read(str(path), dtype='float32'); y = y if y.ndim == 1 else y.mean(1)
    if s != 22050: y = librosa.resample(y, orig_sr=s, target_sr=22050)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2; return S, librosa.fft_frequencies(sr=22050, n_fft=1024)
def band_db(S, fr, act):
    return [10 * np.log10(S[(fr >= a) & (fr < b)][:, act].sum(0).mean() + 1e-12) for a, b in EDGES]
def compare(inp, out):
    Si, fr = spec(inp); So, _ = spec(out); n = min(Si.shape[1], So.shape[1]); Si, So = Si[:, :n], So[:, :n]
    e = 10 * np.log10(Si[(fr >= 100) & (fr < 4000)].sum(0) + 1e-12); act = e > e.max() - 30
    return [round(b - a, 1) for a, b in zip(band_db(Si, fr, act), band_db(So, fr, act))], [round(x, 1) for x in band_db(Si, fr, act)]
base = c.HERE.parent
rows = {}
v2 = base / 'F1_neuralsvb_v002_restored_2026-10-26/audio'
for nm in ('molihua', 'shige', 'zuixuan'):
    for kind in ('vocoder_only', 'a2p'):
        rows[f'{nm}/{kind}'] = compare(v2 / f'{nm}_normal__original.wav', v2 / f'{nm}_normal__{kind}.wav')
ro = base / 'F1_neuralsvb_official_demo_repro_v001'
for tag, lang, ids in [('en', 'English', [1, 2, 3]), ('zh', 'Chinese', [1, 2, 3, 4, 5])]:
    for i in ids:
        gta = ro / 'official' / f'{lang}_gt_a_{i}.wav'; ours = ro / ('ours_en' if tag == 'en' else 'ours')
        rows[f'official_{tag}{i}/vocoder_only'] = compare(gta, ours / f'{tag}{i}_P_vocoder_only_gt_a.wav')
        rows[f'official_{tag}{i}/official_NSVB'] = compare(gta, ro / 'official' / f'{lang}_NSVB_{i}.wav')
print('bands Hz:', EDGES); print('value = output minus input level (dB); last list = input band level dB')
for k, (d, lv) in rows.items(): print(f'{k:34s} excess {d}  | input level {lv}')
json.dump(rows, open('hf_check.json', 'w'), indent=1)
