"""One variable: input level. Scale the 22.05 kHz input by -10 dB (about the level of the official clips, -22 dBFS RMS) before the identical whole-file pipeline;
output is compared at matched RMS. Official amateur takes: median RMS -22 dBFS; user recordings: about -12 dBFS."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf, librosa
import common as c, pipeline as p, eval_metrics as em
sr = 22050; E = c.HERE / 'eval_out' / 'audio'; b = p.Beautifier(); rep_all = {}
GAIN_DB = -10.0
for nm in ('shige', 'zuixuan'):
    y = librosa.resample(sf.read(str(c.ROOT / f'test_dataset/my_singing/{nm}_normal.wav'), dtype='float32')[0], orig_sr=44100, target_sr=22050)
    yq = y * 10 ** (GAIN_DB / 20); print(nm, 'input rms dBFS %.1f -> %.1f' % (20*np.log10(np.sqrt(np.mean(y**2))), 20*np.log10(np.sqrt(np.mean(yq**2)))), flush=True)
    a = b.analyse(yq, 'aligned'); tgt, rep = p.build_target(a, True); base = b.synthesise(a, tgt)
    out, _ = p.postprocess(a, base, True, 1.0)
    a0 = b.analyse(y, 'aligned'); out = out * 10 ** (-GAIN_DB / 20)       # back to the original level for a fair loudness match later
    sf.write(E / f'my_{nm}__T_quiet.wav', out.astype(np.float32), sr, subtype='FLOAT')
    whole = sf.read(str(E / f'my_{nm}__Q_full.wav'), dtype='float64')[0]; k = min(len(a0.wav), len(out), len(whole))
    rep_all[nm] = {'quiet': em.all_metrics(a0.wav[:k], out[:k]), 'whole': em.all_metrics(a0.wav[:k], whole[:k])}
    for kk in ('whole', 'quiet'): print(' ', kk, json.dumps({q: rep_all[nm][kk][q] for q in ('worst_drop_db', 'hnr_median_db', 'shimmer', 'p95_dev_cents')}), 'colour', rep_all[nm][kk]['colour']['nasal_norm_db'], flush=True)
json.dump(rep_all, open('quiet_report.json', 'w'), indent=1); print('DONE')
