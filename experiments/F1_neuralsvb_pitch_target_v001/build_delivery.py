"""Export the three user recordings: original, old v002 a2p (what Jade already heard), candidate A (dip repair + full colour match),
candidate B (dip repair + half colour match). Reuses the contract_v001 exporter and engineering checks (thresholds unchanged)."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, soundfile as sf
import common as c
c.feature_engine()
sys.path.insert(0, str(HERE.parent / 'F1_neuralsvb_contract_v001'))
import run_pipeline as rp
rp.HERE = HERE                                   # keep rp.log() inside this experiment
OUT = HERE / 'delivery_v6'; E = HERE / 'eval_out' / 'audio'
hp = c._hp()
for nm in ('shige', 'zuixuan', 'molihua'):
    src = c.ROOT / 'test_dataset/my_singing' / f'{nm}_normal.wav'
    rd = lambda k: sf.read(str(E / f'my_{nm}__{k}.wav'), dtype='float64')[0]
    n_src = len(sf.read(str(src), dtype='float64')[0])
    need = (n_src + 1) // 2 + 2
    fit = lambda w: np.pad(w, (0, max(0, need - len(w)))) if len(w) < need else w
    raws = {'old_v002': fit(rd('v002_a2p')), 'candA_full': fit(rd('Q_full')), 'candB_half': fit(rd('R_half'))}
    if (OUT / src.stem).exists(): print('skip existing', nm); continue
    rep = rp.checked_exports(src, raws, OUT / src.stem, hp)
    print(nm, {k: (v['checks']['technical_pass'], v['checks']['errors'], round(v['checks']['active_gain']['min_db'], 1)) for k, v in rep['outputs'].items() if 'checks' in v}, flush=True)
print('DONE', flush=True)
