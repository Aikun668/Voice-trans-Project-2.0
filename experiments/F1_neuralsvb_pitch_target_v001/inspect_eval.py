import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf
import eval_metrics as em
A = Path('eval_out/audio'); sr = 22050
rd = lambda n, c: sf.read(str(A / f'{n}__{c}.wav'), dtype='float64')[0]
print('--- fast_f0 (Praat, cents) input vs conditions')
for n in ('my_molihua', 'my_shige', 'my_zuixuan'):
    x = rd(n, 'input'); print(n, {c: em.fast_f0_cents(em.praat_pitch(rd(n, c) if c != 'input' else x)) for c in ('input', 'v002_a2p', 'A_legacy', 'B_aligned', 'vocoder_only', 'E_colour')})
print('--- remaining dips (< -10 dB, 100 ms blocks) after E_colour: time, input dB, f0 voiced in input?')
for n in ('my_shige', 'my_zuixuan'):
    x, y = rd(n, 'input'), rd(n, 'E_colour'); k = min(len(x), len(y)); li, lo = em.block_db(x[:k]), em.block_db(y[:k]); act = li > li.max() - 45
    f0 = em.praat_pitch(x); out = []
    for i in np.where(act & ((lo - li) < -10))[0]:
        j = int(i * 2205 / 128); out.append((round(i * 0.1, 1), round(float(li[i]), 0), round(float(lo[i] - li[i]), 1), bool((f0[j:j + 17] > 0).mean() > .5)))
    print(n, 'blocks', len(out), out[:20])
    a = rd(n, 'D_dip'); lo2 = em.block_db(a[:k]); bad = np.where(act & ((lo2 - li) < -10))[0]
    print('   after D_dip only:', len(bad), 'blocks; voiced share', round(float(np.mean([(f0[int(i*2205/128):int(i*2205/128)+17] > 0).mean() > .5 for i in bad])), 2) if len(bad) else '-')
