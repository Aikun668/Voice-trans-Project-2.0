"""Is the user's recording inside the vocoder's comfort zone? Round-trip L1 vs input gain, plus per-band error and input levels. Measurements only.
F0: contract-aligned Praat track delayed by 1 frame (the convention the vocoder-as-judge test preferred)."""
import sys, tempfile, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'F1_neuralsvb_contract_v001'))
import numpy as np, soundfile as sf, librosa, torch, common as c
import feature_contract as fc
n = c.feature_engine(); eng = n.NeuralSVB(); tmp = Path(tempfile.mkdtemp(prefix='level_'))
from vocoders.pwg import PWG
fr = librosa.mel_frequencies(80, fmin=50, fmax=11025); bands = {'<0.5k': fr < 500, '0.5-1.5k': (fr >= 500) & (fr < 1500), '1.5-4k': (fr >= 1500) & (fr < 4000), '4k+': fr >= 4000}
def prep(path, seconds=None):
    y, s = sf.read(str(path), dtype='float32')
    if y.ndim > 1: y = y.mean(1)
    if seconds: y = y[: int(s * seconds)]
    return librosa.resample(y, orig_sr=s, target_sr=22050) if s != 22050 else y
def rt(y):
    p = tmp / 'x.wav'; sf.write(p, y, 22050, subtype='FLOAT'); wav, mel = PWG.wav2spec(str(p)); T = len(mel) - len(mel) % 4; mel = mel[:T]
    f0, _ = fc.aligned_f0(wav, len(mel), frame_offset_seconds=-128 / 22050)
    torch.manual_seed(0); w = eng.vocoder.spec2wav(mel, f0=f0)
    q = tmp / 'o.wav'; sf.write(q, w, 22050, subtype='FLOAT'); _, mel2 = PWG.wav2spec(str(q)); L = min(len(mel), len(mel2)); d = mel2[:L] - mel[:L]
    act = np.abs(mel[:L]).sum(1) > 0
    return float(np.abs(d).mean()), {k: round(float(np.abs(d[:, m]).mean()), 3) for k, m in bands.items()}, {k: round(float(d[:, m].mean()), 3) for k, m in bands.items()}
src = {'zh1': c.clip_path('Chinese', 'gt_a', 1), 'en8': c.clip_path('English', 'gt_a', 8), 'en12': c.clip_path('English', 'gt_a', 12)}
mine = c.ROOT / 'test_dataset/my_singing'
for nm in ('shige_normal', 'molihua_normal', 'zuixuan_normal'): src[nm] = mine / f'{nm}.wav'
res = {}
for name, path in src.items():
    y = prep(path, 10 if name not in ('zh1', 'en8', 'en12') else None); act = y[np.abs(y) > 1e-4]
    rms = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9); peak = 20 * np.log10(np.abs(y).max() + 1e-9)
    res[name] = {'rms_dbfs': round(float(rms), 1), 'peak_dbfs': round(float(peak), 1), 'gains': {}}
    for g in (-12, -6, 0, 6, 12):
        l1, band, bias = rt(np.clip(y * 10 ** (g / 20), -1, 1)); res[name]['gains'][g] = {'l1': round(l1, 4), 'band_l1': band, 'band_bias': bias}
    print(name, f"rms {rms:.1f} peak {peak:.1f} dBFS | L1 by gain", {g: v['l1'] for g, v in res[name]['gains'].items()}, flush=True)
    g0 = res[name]['gains'][0]; print('   at 0 dB band L1', g0['band_l1'], 'bias', g0['band_bias'], flush=True)
json.dump(res, open('level_check.json', 'w'), indent=1); print('DONE', flush=True)
