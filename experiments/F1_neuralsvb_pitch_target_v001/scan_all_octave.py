"""Which clips change under the evidence-based octave repair? (analysis only, no synthesis)"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf, librosa
import common as c, pipeline as p
b = p.Beautifier.__new__(p.Beautifier); b.n = c.feature_engine()
def load22(path):
    y, s = sf.read(str(path), dtype='float32'); y = y if y.ndim == 1 else y.mean(1)
    return librosa.resample(y, orig_sr=s, target_sr=22050) if s != 22050 else y
class _Stub:  # analyse() only needs the speaker encoder for 'spk'; skip it here
    def embed_utterance(self, w): return np.zeros(256)
b.eng = type('E', (), {'voice_encoder': _Stub()})()
jobs = [(f'{t}{i}', c.clip_path(c.LANGS[t], 'gt_a', i)) for t, i in c.all_clips()] + [(f'my_{n}', c.ROOT / f'test_dataset/my_singing/{n}_normal.wav') for n in ('molihua', 'shige', 'zuixuan')]
out = {}
for name, path in jobs:
    a = p.Beautifier.analyse(b, load22(path), 'aligned'); tgt, rep = p.build_target(a, True); o = rep['octave_evidence']
    out[name] = {'changed': o['frames_changed'], 'stretches': o['stretches'], 'octave_fixed_frames_local': rep['octave_fixed_frames'], 'gap': rep['gap_filled_frames']}
    if o['stretches']: print(name, 'frames_changed', o['frames_changed'], [(s['start_frame'], s['end_frame'], s['tracker_hz'], s['alt_hz'], s['comb_tracker'], s['comb_alt'], s['applied']) for s in o['stretches']], flush=True)
json.dump(out, open('octave_all.json', 'w'), indent=1)
print('clips with changes:', [k for k, v in out.items() if v['changed']]); print('DONE')
