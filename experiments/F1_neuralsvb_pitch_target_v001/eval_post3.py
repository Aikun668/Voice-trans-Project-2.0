"""Variants on saved model outputs: M_level = level_match | N_level_full = M + colour 1.0 | O_level_half = M + colour 0.5 | P_dip_level_full = offset-aware dip repair, then level_match, then colour 1.0.
Then export through the project's own checks (delivery_v5)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf
import common as c, eval_metrics as em
from colour_match import ColourConfig, colour_match
from dip_repair import repair_dips
from level_match import level_match
E = c.HERE / 'eval_out'; sr = 22050
rd = lambda n, k: sf.read(str(E / 'audio' / f'{n}__{k}.wav'), dtype='float64')[0]
for n in ('my_shige', 'my_zuixuan', 'my_molihua'):
    sfx = '_oct' if n == 'my_zuixuan' else ''
    x, c_t, v = rd(n, 'input'), rd(n, 'C_target' + sfx), rd(n, 'vocoder_only' + sfx); k = min(map(len, (x, c_t, v))); x, c_t, v = x[:k], c_t[:k], v[:k]
    m, rep = level_match(x, c_t, sr); out = {'M_level': m}
    out['N_level_full'], _ = colour_match(x, m, sr, ColourConfig(strength=1.0)); out['O_level_half'], _ = colour_match(x, m, sr, ColourConfig(strength=0.5))
    d, _ = repair_dips(c_t, v, np.zeros(k // 128 + 1), sr); d, _ = level_match(x, d, sr); out['P_dip_level_full'], _ = colour_match(x, d, sr, ColourConfig(strength=1.0))
    for kk, w in out.items(): sf.write(E / 'audio' / f'{n}__{kk}.wav', w.astype(np.float32), sr, subtype='FLOAT')
    print(n, 'level_match', rep, flush=True)
    for kk, w in out.items(): print('   ', kk, json.dumps({q: em.all_metrics(x, w)[q] for q in ('hnr_median_db', 'jitter', 'shimmer', 'median_dev_cents', 'p95_dev_cents', 'fast_f0_cents')}), 'colour', em.colour_shift(x, w)['nasal_norm_db'], flush=True)
print('DONE')
