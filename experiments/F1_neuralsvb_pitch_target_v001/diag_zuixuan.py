"""Why does zuixuan candA still trip the gain-spread (8.4>8) and pitch-P95 (48.6>35) checks? Uses the project's own measurement functions on the exported files."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import f1_poc as qa
R = Path('delivery_v3/zuixuan_normal')
orig, sr = qa.read_pcm16(R / 'zuixuan_normal__original.wav')
for name in ('candA_full', 'old_v002'):
    out, _ = qa.read_pcm16(R / f'zuixuan_normal__{name}.wav')
    b, a = qa.frame_rms(orig, sr), qa.frame_rms(out, sr); m = min(len(a), len(b)); b, a = b[:m], a[:m]
    act = b > max(b.max() * 0.08, 1e-4); g = 20 * np.log10(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)); g[~act] = np.nan
    med = np.nanmedian(g[act]); gg = g - med; hop = 0.02
    hi, lo = np.nanpercentile(gg[act], [95, 5])
    print(name, 'median gain %.1f dB | p5 %.1f p95 %.1f spread %.1f' % (med, lo, hi, hi - lo))
    idx = np.where(act)[0]; order = idx[np.argsort(gg[idx])]
    print('  lowest-gain frames (t, gain-med dB):', [(round(i * hop, 2), round(float(gg[i]), 1)) for i in sorted(order[:10])])
    print('  highest-gain frames:', [(round(i * hop, 2), round(float(gg[i]), 1)) for i in sorted(order[-10:])])
    if name == 'candA_full':
        po, pn = qa.estimate_pitch(orig, sr), qa.estimate_pitch(out, sr); n = min(len(po), len(pn)); po, pn = po[:n], pn[:n]
        ok = ~np.isnan(po) & ~np.isnan(pn); c_ = 1200 * np.abs(np.log2(pn[ok] / po[ok])); t = np.arange(n)[ok] * 0.02
        big = np.where(c_ > 35)[0]; print('  pitch frames', ok.sum(), 'over 35 c:', len(big), '| times', [round(float(t[i]), 2) for i in big][:30], '| cents', [int(c_[i]) for i in big][:30])
