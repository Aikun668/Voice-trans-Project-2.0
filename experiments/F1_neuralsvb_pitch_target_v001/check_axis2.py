"""Voicing-boundary lag test. For lag k (frames), agreement between (track shifted by k) voiced mask and pYIN voiced mask.
pYIN frame i is centred at i*hop, same as the mel (center=True). Peak lag tells the frame offset of each F0 track."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'F1_neuralsvb_contract_v001'))
import numpy as np, librosa, common as c
import feature_contract as fc
tot = {'legacy': np.zeros(17), 'aligned': np.zeros(17)}
for tag, cid in [('zh', 1), ('zh', 4), ('zh', 9), ('zh', 12), ('en', 1), ('en', 8), ('en', 12), ('en', 5)]:
    d = c.LANGS[tag]; wav, mel, f_leg = c.features(c.clip_path(d, 'gt_a', cid)); f_al, _ = fc.aligned_f0(wav, len(mel))
    yp, vf, _ = librosa.pyin(wav, fmin=75, fmax=800, sr=22050, frame_length=2048, hop_length=128)
    n = min(len(mel), len(yp)); ref = vf[:n]
    for name, f in [('legacy', f_leg), ('aligned', f_al)]:
        m = (f[:n] > 0)
        tot[name] += np.array([np.mean(np.roll(m, k) == ref) for k in range(-8, 9)])
for name, v in tot.items():
    v = v / 8; k = int(np.argmax(v)) - 8
    print(name, 'best shift', k, 'frames | agreement at 0: %.3f, at best: %.3f | by shift:' % (v[8], v.max()), ' '.join(f'{x:.3f}' for x in v[4:13]))
