"""Vocoder-as-judge: round-trip mel -> HiFi-GAN(f0 shifted by k frames) -> mel'. Lowest L1(mel', mel) = the F0 time convention the vocoder was trained with.
k > 0 delays the F0 track by k frames. Measurements only."""
import sys, tempfile, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'F1_neuralsvb_contract_v001'))
import numpy as np, soundfile as sf, torch, common as c
import feature_contract as fc
n = c.feature_engine(); eng = n.NeuralSVB()
tmp = Path(tempfile.mkdtemp(prefix='axis3_'))
def shifted(f, k):
    out = np.zeros_like(f)
    if k >= 0: out[k:] = f[:len(f) - k] if k else f
    else: out[:k] = f[-k:]
    return out
def roundtrip_l1(path, f0v, mel, wav):
    torch.manual_seed(0); w = eng.vocoder.spec2wav(mel, f0=f0v); p = tmp / 'rt.wav'; sf.write(p, w, 22050); _, mel2 = __import__('vocoders.pwg', fromlist=['PWG']).PWG.wav2spec(str(p))
    L = min(len(mel), len(mel2)); act = np.abs(mel[:L]).sum(1) > 0
    return float(np.abs(mel2[:L] - mel[:L]).mean())
clips = [(c.clip_path(c.LANGS[t], 'gt_a', i), f'{t}{i}') for t, i in [('zh', 1), ('zh', 4), ('zh', 9), ('en', 1), ('en', 8), ('en', 12)]]
sh = tmp / 'shige8.wav'; y, s = sf.read(str(c.ROOT / 'test_dataset/my_singing/shige_normal.wav')); sf.write(sh, y[:s * 10], s); clips.append((sh, 'shige10s'))
ks = list(range(-4, 5)); res = {}
for path, name in clips:
    wav, mel, f_leg = c.features(path); f_al, _ = fc.aligned_f0(wav, len(mel)); T = len(mel) - len(mel) % 4
    mel, f_leg, f_al = mel[:T], f_leg[:T], f_al[:T]
    res[name] = {'legacy': [round(roundtrip_l1(path, shifted(f_leg, k), mel, wav), 4) for k in ks], 'aligned': [round(roundtrip_l1(path, shifted(f_al, k), mel, wav), 4) for k in ks]}
    print(name, 'legacy', res[name]['legacy'], '| aligned', res[name]['aligned'], flush=True)
json.dump({'ks': ks, 'res': res}, open('axis3.json', 'w'), indent=1)
for tr in ('legacy', 'aligned'):
    m = np.mean([res[k][tr] for k in res if k != 'shige10s'], axis=0); print('MEAN official', tr, 'best k', ks[int(np.argmin(m))], [round(float(x), 4) for x in m])
print("DONE", flush=True)
