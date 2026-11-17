"""Control: contract_v001 aligned-f0 full chain on SHIGE, baseline vs. f0 fixed ONLY at confirmed octave-error frames.
Single variable. Same seed, same model/vocoder. Fixed frames take pYIN's value (22.05 kHz, same hop)."""
import json, sys, hashlib
from pathlib import Path
import numpy as np, torch, librosa
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'experiments/F1_neuralsvb_contract_v001'), str(ROOT), str(ROOT / 'neuralsvb_f1')]
import feature_contract as contract, neuralsvb_infer as n
import run_pipeline as rp
rp.HERE = HERE  # keep execution.log inside this experiment, not contract_v001
FIX_WINDOW = (16.645, 16.690)  # audited: Praat=116 Hz (half), pYIN 230-234 Hz, harmonic comb favours ~233 Hz

source = ROOT / 'test_dataset/my_singing/shige_normal.wav'
src_hash = hashlib.sha256(source.read_bytes()).hexdigest()
torch.set_num_threads(4)
engine = n.NeuralSVB(); hp = engine.hp
wav, mel, _, spk = engine.features(str(source))
f0, info = contract.aligned_f0(wav, len(mel))
grid = np.arange(len(mel)) * 128 / 22050
yp, _, _ = librosa.pyin(wav, fmin=60, fmax=800, sr=22050, frame_length=2048, hop_length=128)
pt = librosa.times_like(yp, sr=22050, hop_length=128)
sel = (grid >= FIX_WINDOW[0]) & (grid <= FIX_WINDOW[1]) & (f0 > 0) & (f0 < 160)
f0_fix = f0.copy()
skipped = []
for k in np.where(sel)[0]:
    v = yp[np.argmin(abs(pt - grid[k]))]
    if not np.isfinite(v):  # pYIN says unvoiced: octave error not confirmed, leave untouched
        skipped.append(int(k)); continue
    assert 200 < v < 260, (k, v)
    f0_fix[k] = v
changes = [{'frame': int(k), 't': round(float(grid[k]), 4), 'from': round(float(f0[k]), 1), 'to': round(float(f0_fix[k]), 1)} for k in np.where(f0 != f0_fix)[0]]
assert 0 < len(changes) <= 12
rp.log('skipped (pYIN unvoiced, not confirmed): ' + str(skipped))
rp.log('F0 changes: ' + json.dumps(changes))

def chain(f0v, tag, raws):
    torch.manual_seed(0)
    raws[tag + '_vocoder'] = engine.vocoder.spec2wav(mel, f0=f0v)
    from utils.pitch_utils import f0_to_coarse
    count = len(mel); pad = (-count) % 4
    pm = np.pad(mel, ((0, pad), (0, 0)), mode='edge'); pf = np.pad(f0v, (0, pad), mode='edge')
    pitch = torch.tensor(f0_to_coarse(pf), dtype=torch.long)[None]
    mt = torch.tensor(pm, dtype=torch.float32)[None]; st = torch.tensor(spk, dtype=torch.float32)[None]
    align = torch.arange(len(pm), dtype=torch.long)[None]
    torch.manual_seed(0)
    with torch.no_grad():
        out = engine.model(amateur_mel=mt, prof_mel=mt, amateur_pitch=pitch, prof_pitch=pitch,
                           amateur_spk_id=st, prof_spk_id=st, a2p_alignment=align, p2a_alignment=None,
                           infer=False, concurrent_ways=['a2a', 'p2p', 'a2p'], disable_map=False)
    for way in ['a2a', 'a2p']:
        torch.manual_seed(0)
        raws[f'{tag}_{way}'] = engine.vocoder.spec2wav(out[way]['mel_out'][0].detach().numpy()[:count], f0=f0v[:count])
    rp.log('GENERATED ' + tag)

raws = {}
chain(f0, 'base', raws); chain(f0_fix, 'fix', raws)
# reproducibility vs. existing contract_v001 output
ref = ROOT / 'experiments/F1_neuralsvb_contract_v001/full_normal/shige_normal'
repro = {k: float(np.max(np.abs(raws['base_' + k] - np.load(ref / f'{k}_native.npy')))) if (ref / f'{k}_native.npy').exists() else None
         for k in ['vocoder', 'a2a', 'a2p']}
rp.log('base vs contract_v001 max abs diff: ' + json.dumps(repro))
report = rp.checked_exports(source, raws, HERE / 'shige_control', hp)
report.update({'f0_changes': changes, 'repro_vs_contract_v001_max_abs': repro, 'source_sha256_before': src_hash,
               'source_sha256_after': hashlib.sha256(source.read_bytes()).hexdigest()})
(HERE / 'shige_control' / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
np.save(HERE / 'shige_control' / 'f0_base.npy', f0); np.save(HERE / 'shige_control' / 'f0_fix.npy', f0_fix)
rp.log('FINISHED')
