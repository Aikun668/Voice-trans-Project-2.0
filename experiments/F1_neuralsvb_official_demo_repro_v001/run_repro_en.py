"""Reproduce official Chinese demo clips with the local weights: (P) official pair path (amateur + professional pitch target),
(S) our F1 single-WAV path (self pitch). Inputs are the demo page's GT Amateur / GT Professional files (untrusted downloads, read-only)."""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'neuralsvb_f1'))
import neuralsvb_infer as n
OFF = HERE / 'official'; OUT = HERE / 'ours_en'; OUT.mkdir(exist_ok=False)
ids = [1, 2, 3]
paths = {i: (str(OFF / f'English_gt_a_{i}.wav'), str(OFF / f'English_gt_p_{i}.wav')) for i in ids}
n._bootstrap(None); probe = n.NeuralSVB()
pooled = np.concatenate([probe.features(p)[2] for pair in paths.values() for p in pair]); pooled = pooled[pooled > 0]
f0n = n.F0Norm(float(pooled.mean()), float(pooled.std()))   # proxy for unreleased train_f0s_mean_std.npy
print('f0 norm proxy', f0n, flush=True)
pair_engine = probe; pair_engine.f0_norm = f0n; pair_engine.hp['f0_mean'], pair_engine.hp['f0_std'] = f0n.mean, f0n.std
rep = {'f0_norm_proxy': [f0n.mean, f0n.std], 'clips': []}
for i in ids:
    a, p = paths[i]
    wavs, res, gt, align = pair_engine.run_pair(a, p)
    sf.write(OUT / f'en{i}_P_a2p_pair.wav', wavs['a2p'], n.SAMPLE_RATE, subtype='PCM_16')
    sf.write(OUT / f'en{i}_P_a2a.wav', wavs['a2a'], n.SAMPLE_RATE, subtype='PCM_16')
    sf.write(OUT / f'en{i}_P_vocoder_only_gt_a.wav', wavs['gt_a'], n.SAMPLE_RATE, subtype='PCM_16')
    # F1 single-WAV path: self pitch, no professional reference, identity F0 standardisation
    pair_engine.f0_norm = None
    s_out, f0_in, _ = pair_engine.run(a, mode='a2p')
    sf.write(OUT / f'en{i}_S_a2p_single.wav', s_out, n.SAMPLE_RATE, subtype='PCM_16')
    pair_engine.f0_norm = f0n; pair_engine.hp['f0_mean'], pair_engine.hp['f0_std'] = f0n.mean, f0n.std
    rep['clips'].append({'id': i, 'frames_a': len(gt['a']), 'frames_p': len(gt['p'])}); print('done', i, flush=True)
json.dump(rep, open(HERE / 'repro_report_en.json', 'w'), indent=1)
print('FINISHED')
