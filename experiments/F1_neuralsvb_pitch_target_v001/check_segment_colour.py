"""Per-2s-window tilt of output vs input (dB): (1.5-4 kHz) minus (<0.5 kHz) change, active frames only. Shows whether a clip-wide colour match leaves time-varying nasal residue."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf
from colour_match import ColourConfig, ltas_db
E = Path('eval_out/audio'); sr = 22050; cfg = ColourConfig(active_db=40)
rd = lambda n, k: sf.read(str(E / f'{n}__{k}.wav'), dtype='float64')[0]
for n in ('my_shige', 'my_zuixuan'):
    sfx = '_oct' if n == 'my_zuixuan' else ''
    x = rd(n, 'input'); res = {}
    for k in ('vocoder_only' + sfx, 'C_target' + sfx, 'N_level_full'):
        y = rd(n, k); m = min(len(x), len(y)); row = []
        for t in range(0, int(m / sr) - 1, 2):
            a, b = x[t * sr:(t + 2) * sr], y[t * sr:(t + 2) * sr]
            f, da, act = ltas_db(a, None, sr, ColourConfig(active_db=30)); _, db, _ = ltas_db(b, act, sr, cfg)
            d = db - da; row.append(round(float(d[(f >= 1500) & (f < 4000)].mean() - d[(f >= 100) & (f < 500)].mean()), 1))
        res[k] = row
    print(n, 'window starts (s):', list(range(0, 2 * len(next(iter(res.values()))), 2)))
    for k, row in res.items(): print(f'   {k:18s}', row, ' mean %.1f  max %.1f' % (np.mean(row), np.max(row)))
