"""zuixuan with the evidence-based octave repair: synth with repaired target, then dip repair (all frames) + colour match 1.0 / 0.5."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf
import common as c, pipeline as p, eval_metrics as em
from dip_repair import repair_dips
from colour_match import ColourConfig, colour_match
E = c.HERE / 'eval_out' / 'audio'; sr = 22050
b = p.Beautifier(); y = sf.read(str(c.ROOT / 'test_dataset/my_singing/zuixuan_normal.wav'), dtype='float32')[0]
import librosa
y = librosa.resample(y, orig_sr=44100, target_sr=22050)
a = b.analyse(y, 'aligned'); tgt, rep = p.build_target(a, True); print('octave frames changed', rep['octave_evidence']['frames_changed'], flush=True)
base = b.synthesise(a, tgt)
sf.write(E / 'my_zuixuan__C_target_oct.wav', base['a2p'].astype(np.float32), sr, subtype='FLOAT')
sf.write(E / 'my_zuixuan__vocoder_only_oct.wav', base['vocoder'].astype(np.float32), sr, subtype='FLOAT')
np.save(c.HERE / 'zuixuan_f0_vocoder_oct.npy', base['f0_vocoder'])
print('DONE')
