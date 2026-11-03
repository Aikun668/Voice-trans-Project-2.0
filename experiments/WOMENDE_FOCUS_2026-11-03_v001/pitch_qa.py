import sys
from pathlib import Path
import numpy as np, parselmouth
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "f2"))
from poc_phase1 import read_wav_pcm16
def f0(p):
    x, sr = read_wav_pcm16(Path(p)); return parselmouth.Sound(x, sr).to_pitch(0.01, 75, 600).selected_array["frequency"]
for a, b in [("thin", "c1_tilt"), ("keyup_only", "c2_keyup_tilt"), ("thin", "keyup_only")]:
    fa, fb = f0(f"audio/{a}.wav"), f0(f"audio/{b}.wav"); n = min(len(fa), len(fb)); m = (fa[:n] > 0) & (fb[:n] > 0)
    c = 1200 * np.log2(fb[:n][m] / fa[:n][m])
    print(f"{b} vs {a}: voiced overlap {m.sum() / max((fa[:n] > 0).sum(), 1):.3f}, median {np.median(c):+.1f} cents, p95|.| {np.percentile(np.abs(c), 95):.1f}")
