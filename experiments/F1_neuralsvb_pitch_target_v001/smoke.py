import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf
import common as c, pipeline as p, eval_metrics as em
t0 = time.time(); b = p.Beautifier(); print('model loaded', round(time.time() - t0, 1), 's', flush=True)
y = sf.read(str(c.clip_path('English', 'gt_a', 8)), dtype='float32')[0]
a = b.analyse(y); print('analysed: frames', a.count, 'wav', len(a.wav), 'voiced', int((a.f0 > 0).sum()), 'hint', int(a.hint.sum()), flush=True)
tgt, rep = p.build_target(a, True); print('target report', rep, flush=True)
base = b.synthesise(a, tgt); print('synth lens', {k: len(v) for k, v in base.items()}, round(time.time() - t0, 1), 's', flush=True)
for name, (dip, col) in {'C_target': (False, 0), 'D_dip': (True, 0), 'E_colour': (True, 1.0)}.items():
    w, r = p.postprocess(a, base, dip, col); m = em.all_metrics(a.wav, w); print(name, json.dumps(m, ensure_ascii=False), '| post', json.dumps(r, ensure_ascii=False), flush=True)
print('input', json.dumps(em.all_metrics(a.wav, a.wav), ensure_ascii=False)); print('DONE')
