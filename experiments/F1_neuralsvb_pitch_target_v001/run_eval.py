"""Evaluate the candidate settings on all official demo clips and the user's three recordings. Resumable (one json per clip).
Conditions: A_legacy (old F1 single-WAV axis, own F0) | B_aligned (new axis, own F0) | C_target (+ pitch-target repair)
            D_dip (+ dip repair) | E_colour (+ colour calibration) | official (authors' output, official clips only)
            v002_a2p (the version the user already knows, user's recordings only)."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf, librosa
import common as c, pipeline as p, eval_metrics as em
OUT = c.HERE / 'eval_out'; (OUT / 'audio').mkdir(parents=True, exist_ok=True)
def load22(path):
    y, s = sf.read(str(path), dtype='float32'); y = y if y.ndim == 1 else y.mean(1)
    return librosa.resample(y, orig_sr=s, target_sr=22050) if s != 22050 else y
jobs = [(f'{t}{i}', c.clip_path(c.LANGS[t], 'gt_a', i), c.clip_path(c.LANGS[t], 'NSVB', i), None) for t, i in c.all_clips()]
v2 = c.HERE.parent / 'F1_neuralsvb_v002_restored_2026-10-26/audio'
for nm in ('molihua', 'shige', 'zuixuan'):
    jobs.append((f'my_{nm}', c.ROOT / f'test_dataset/my_singing/{nm}_normal.wav', None, v2 / f'{nm}_normal__a2p.wav'))
b = p.Beautifier()
def save(name, cond, y): sf.write(OUT / 'audio' / f'{name}__{cond}.wav', np.asarray(y, dtype=np.float32), 22050, subtype='FLOAT')
for name, src, official, v2a2p in jobs:
    f = OUT / f'{name}.json'
    if f.exists(): continue
    t0 = time.time(); y = load22(src); res = {'name': name, 'split': c.split_of(name[:2], int(name[2:])) if not name.startswith('my_') else 'user', 'seconds': round(len(y) / 22050, 1)}
    aL = b.analyse(y, 'legacy'); baseA = b.synthesise(aL, aL.f0); save(name, 'A_legacy', baseA['a2p']); res['A_legacy'] = em.all_metrics(aL.wav, baseA['a2p'])
    a = b.analyse(y, 'aligned'); baseB = b.synthesise(a, a.f0); save(name, 'B_aligned', baseB['a2p']); res['B_aligned'] = em.all_metrics(a.wav, baseB['a2p'])
    tgt, rep = p.build_target(a, True); res['target_report'] = rep
    baseC = baseB if np.array_equal(tgt, a.f0) else b.synthesise(a, tgt)
    save(name, 'C_target', baseC['a2p']); res['C_target'] = em.all_metrics(a.wav, baseC['a2p'])
    wD, rD = p.postprocess(a, baseC, True, 0.0); save(name, 'D_dip', wD); res['D_dip'] = em.all_metrics(a.wav, wD); res['dip_report'] = rD.get('dip')
    wE, rE = p.postprocess(a, baseC, True, 1.0); save(name, 'E_colour', wE); res['E_colour'] = em.all_metrics(a.wav, wE); res['colour_report'] = rE.get('colour')
    save(name, 'vocoder_only', baseC['vocoder']); res['vocoder_only'] = em.all_metrics(a.wav, baseC['vocoder'])
    if official is not None:
        o = load22(official); res['official'] = {**em.voice_quality(o), 'colour': em.colour_shift(a.wav, o, same_timeline=False), 'fast_f0_cents': em.fast_f0_cents(em.praat_pitch(o))}
    if v2a2p is not None:
        o = load22(v2a2p); n = min(len(o), len(a.wav)); res['v002_a2p'] = em.all_metrics(a.wav[:n], o[:n]); save(name, 'v002_a2p', o)
    sf.write(OUT / 'audio' / f'{name}__input.wav', a.wav.astype(np.float32), 22050, subtype='FLOAT')
    f.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8'); print(name, 'done', round(time.time() - t0, 1), 's', flush=True)
print('DONE', flush=True)
