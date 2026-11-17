"""Fast amplitude flutter (roughness proxy): std (dB) of the 5.8 ms frame level minus its 46 ms moving average, over voiced active frames; plus run lengths of >=8 dB dips. Measurements only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf
from dip_repair import frame_db
import common as c
E = Path('eval_out/audio'); rd = lambda n, k: sf.read(str(E / f'{n}__{k}.wav'), dtype='float64')[0]
def flutter(y, ref_active):
    L = frame_db(y, win=256); k = 9; sm = np.convolve(np.pad(L, k // 2, mode='edge'), np.ones(k) / k, 'valid')[:len(L)]
    d = (L - sm); n = min(len(d), len(ref_active)); return float(np.std(d[:n][ref_active[:n]]))
def pct_lowdips(y, ref, ref_active):
    a, b = frame_db(y)[:len(ref_active)], frame_db(ref)[:len(ref_active)]; n = min(len(a), len(b), len(ref_active)); return float(np.mean(((b[:n] - a[:n]) > 8)[ref_active[:n]]))
for n in ('my_molihua', 'my_shige', 'my_zuixuan'):
    x = rd(n, 'input'); L = frame_db(x); act = L > np.percentile(L, 99) - 25
    row = {}
    for k in ('input', 'vocoder_only', 'C_target', 'G_dip_all', 'H_full'):
        y = rd(n, k)[:len(x)]; row[k] = (round(flutter(y, act), 2), round(pct_lowdips(y, x[:len(y)], act), 3))
    print(n, 'flutter dB / share of active frames >8 dB below input:', row)
