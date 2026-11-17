"""Which F0 time axis matches the mel frames? legacy = official _get_pitch (8-frame left pad); aligned = Praat frame centres mapped to mel frame centres.
Reference: pYIN on the same waveform (librosa frame i is centred at i*hop, like the center=True mel)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'F1_neuralsvb_contract_v001'))
import numpy as np, librosa, parselmouth, common as c
import feature_contract as fc
rows = []
for tag, cid in [('zh', 1), ('zh', 4), ('zh', 9), ('en', 1), ('en', 8), ('en', 12)]:
    d = c.LANGS[tag]; wav, mel, f_leg = c.features(c.clip_path(d, 'gt_a', cid))
    f_al, info = fc.aligned_f0(wav, len(mel))
    yp, vf, _ = librosa.pyin(wav, fmin=75, fmax=800, sr=22050, frame_length=2048, hop_length=128)
    n = min(len(mel), len(yp)); ref = np.where(vf[:n], yp[:n], 0)
    def err(f):
        m = (f[:n] > 0) & (ref > 0); return float(np.median(np.abs(1200 * np.log2(f[:n][m] / ref[m])))) if m.sum() > 20 else float('nan')
    # best integer lag for the legacy track
    lags = {k: np.nanmedian(np.abs(1200*np.log2(np.roll(f_leg, k)[:n][(np.roll(f_leg, k)[:n] > 0) & (ref > 0)] / ref[(np.roll(f_leg, k)[:n] > 0) & (ref > 0)]))) for k in range(-8, 9)}
    best = min(lags, key=lags.get)
    rows.append((tag, cid, err(f_leg), err(f_al), best, info['first_praat_center_seconds']))
    print(f'{tag}{cid:>2} legacy vs pYIN {err(f_leg):6.1f} c | aligned vs pYIN {err(f_al):6.1f} c | best lag of legacy {best:+d} frames | praat x1 {info["first_praat_center_seconds"]*1000:.1f} ms (legacy assumes {8*128/22050*1000:.1f} ms)', flush=True)
