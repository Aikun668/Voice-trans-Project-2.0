import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, soundfile as sf
import common as c
c.feature_engine()
sys.path.insert(0, str(HERE.parent / 'F1_neuralsvb_contract_v001'))
import run_pipeline as rp
rp.HERE = HERE
OUT = HERE / 'delivery_v7'; E = HERE / 'eval_out' / 'audio'; hp = c._hp()
for nm in ('shige', 'zuixuan'):
    src = c.ROOT / 'test_dataset/my_singing' / f'{nm}_normal.wav'; n_src = len(sf.read(str(src), dtype='float64')[0]); need = (n_src + 1) // 2 + 2
    fit = lambda w: np.pad(w, (0, max(0, need - len(w))))
    rd = lambda k: fit(sf.read(str(E / f'my_{nm}__{k}.wav'), dtype='float64')[0])
    if (OUT / src.stem).exists(): continue
    rep = rp.checked_exports(src, {'whole': rd('Q_full'), 'chunked': rd('S_chunked')}, OUT / src.stem, hp)
    print(nm, {k: (v['checks']['technical_pass'], v['checks']['errors']) for k, v in rep['outputs'].items() if 'checks' in v}, flush=True)
print('DONE')
