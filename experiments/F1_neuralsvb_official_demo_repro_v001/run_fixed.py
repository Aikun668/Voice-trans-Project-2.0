"""Pair flow with ONE change vs run_repro.py: pitch inputs follow the shipped config (normalize_pitch: false) -> raw F0, no standardisation to the proxy mean/std.
Everything else (weights, seed, DTW alignment, vocoder) identical."""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'neuralsvb_f1'))
import neuralsvb_infer as n
OFF = HERE / 'official'; OUT = HERE / 'ours_fixed'; OUT.mkdir(exist_ok=True)
n._bootstrap(None); eng = n.NeuralSVB()
def raw_pitch_inputs(f0):
    from utils.pitch_utils import f0_to_coarse, norm_interp_f0, denorm_f0
    hp = eng.hp; f0 = f0.copy()
    pitch = f0_to_coarse(f0)                      # config normalize_pitch=false: stored raw pitch ids
    f0n, uv = norm_interp_f0(f0, hp)              # vocoder branch unchanged (interp + denorm round trip)
    return pitch, denorm_f0(torch.FloatTensor(f0n), torch.FloatTensor(uv), hp).numpy()
eng.hp['f0_mean'], eng.hp['f0_std'] = 300.0, 100.0   # any fixed pair: norm->denorm round trip is identity for the vocoder branch
eng._pitch_inputs = raw_pitch_inputs
eng.f0_norm = n.F0Norm(0.0, 1.0)                  # only to satisfy run_pair's assert; unused by raw_pitch_inputs
for tag, lang, ids in [('en', 'English', [1, 2, 3]), ('zh', 'Chinese', [1, 2, 3, 4, 5])]:
    for i in ids:
        if (OUT / f'{tag}{i}_P_a2p_pair.wav').exists(): continue
        wavs, res, gt, al = eng.run_pair(str(OFF / f'{lang}_gt_a_{i}.wav'), str(OFF / f'{lang}_gt_p_{i}.wav'))
        for k, nm in [('a2p', 'P_a2p_pair'), ('a2a', 'P_a2a'), ('gt_a', 'P_vocoder_only_gt_a')]:
            sf.write(OUT / f'{tag}{i}_{nm}.wav', wavs[k], n.SAMPLE_RATE, subtype='PCM_16')
        print('done', tag, i, flush=True)
print('FINISHED')
