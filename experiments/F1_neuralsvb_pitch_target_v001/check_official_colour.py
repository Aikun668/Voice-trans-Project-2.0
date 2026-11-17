"""Does the authors' own output also carry the spectral tilt? NSVB vs its amateur input, in the same bands (clip-level LTAS, different timeline allowed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf, common as c
from colour_match import ColourConfig, ltas_db
B = [(0, 500), (500, 1500), (1500, 4000), (4000, 8000)]
def ltas(y): 
    f, d, _ = ltas_db(y, None, 22050, ColourConfig(active_db=40)); return f, d
rows = []
for tag, cid in c.all_clips():
    d = c.LANGS[tag]; r = lambda k: sf.read(str(c.clip_path(d, k, cid)), dtype='float64')[0]
    fa, da = ltas(r('gt_a')); _, dn = ltas(r('NSVB')); _, dp = ltas(r('gt_p'))
    band = lambda x: [float(x[(fa >= a) & (fa < b)].mean()) for a, b in B]
    rows.append((np.array(band(dn)) - np.array(band(da)), np.array(band(dp)) - np.array(band(da))))
m = np.mean([x[0] for x in rows], 0); s = np.std([x[0] for x in rows], 0); mp = np.mean([x[1] for x in rows], 0)
print('authors NSVB minus amateur input, mean dB per band', [round(float(v), 2) for v in m], ' std', [round(float(v), 2) for v in s])
print('professional minus amateur (target), mean dB per band', [round(float(v), 2) for v in mp])
