import sys, json, importlib.util
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / '.venv-f2/Lib/site-packages'))
p = ROOT / 'experiments/WOMENDE_FOCUS_2026-11-04_v005/run.py'
spec = importlib.util.spec_from_file_location('old', p)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
base = ROOT / 'experiments/WOMENDE_FOCUS_2026-11-04_v005/audio'
x, sr = sf.read(str(ROOT / 'experiments/WOMENDE_FOCUS_2026-11-03_v001/work/c2_keyup_tilt_raw.wav'))
mask, rel = old.decay_mask(x, sr)
gain = uniform_filter1d(np.where(mask, -10., 0.), size=6)
ts = np.arange(len(gain)) * old.HOP_S
sel = (ts >= 11) & (ts < 12)
changes = np.abs(np.diff(gain[sel]))
print('11–12s v005 tail gain dB min/max:', np.min(gain[sel]), np.max(gain[sel]))
print('11–12s tail transitions >= 1 dB per 5ms:', int(np.sum(changes >= 1)))
print('11–12s selected gain every 50ms:')
print([(round(float(t), 2), round(float(g), 2)) for t,g in zip(ts[sel][::10], gain[sel][::10])])
for name in ['op_replay', 'shelf', 'tail']:
    y, _ = sf.read(str(base / (name + '.wav')))
    t, f = old.f0_of(y, sr)
    idx = (t >= 11) & (t < 12) & (f > 0)
    ff = f[idx]
    print(name, '11–12 voiced frames:', len(ff), 'F0 Hz median/range:', round(float(np.median(ff)), 2) if len(ff) else None, [round(float(np.min(ff)),2),round(float(np.max(ff)),2)] if len(ff) else None)

