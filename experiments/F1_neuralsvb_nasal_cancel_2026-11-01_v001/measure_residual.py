"""Measure residual spectral tilt of the A01 reverse-calibration outputs (read-only)."""
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
AUDIO = ROOT / "experiments" / "F1_neuralsvb_nasal_cancel_2026-11-01_v001" / "run-001" / "listening" / "audio"
OUT = AUDIO.parent.parent / "residual.json"
SR = 22050
N, H = 2048, 512
BANDS = {"low_0_500": (0, 500), "mid1_500_1500": (500, 1500),
         "mid2_1500_4000": (1500, 4000), "high_4000_8000": (4000, 8000)}


def read(name):
    x, sr = sf.read(str(AUDIO / name), dtype="float64")
    if sr != SR:
        g = math.gcd(sr, SR)
        x = signal.resample_poly(x, SR // g, sr // g)
    return x


def power(x):
    w = np.hanning(N)
    n = 1 + max(0, (len(x) - N) // H)
    fr = np.stack([x[i * H:i * H + N] * w for i in range(n)])
    return np.abs(np.fft.rfft(fr, axis=1)) ** 2, np.fft.rfftfreq(N, 1.0 / SR)


def band(p, f, lo, hi, sel):
    b = (f >= lo) & (f < hi)
    return float(p[np.ix_(sel, b)].sum())


def rel(p, f, sel):
    vals = {k: band(p, f, lo, hi, sel) for k, (lo, hi) in BANDS.items()}
    ref = max(vals["mid2_1500_4000"], 1e-20)
    return {k: float(10 * np.log10(max(v, 1e-20) / ref)) for k, v in vals.items()}


o = read("A01_original.wav")
p_o, f = power(o)
lvl = np.array([band(p_o, f, 100, 4000, np.array([i])) for i in range(len(p_o))])
lvl_db = 10 * np.log10(np.maximum(lvl, 1e-20))
active = lvl_db > lvl_db.max() - 30
rel_o = rel(p_o, f, active)
out = {"active_frames": int(active.sum()), "original": {k: round(v, 2) for k, v in rel_o.items()}}
for key in ("a2p", "eq100", "eq50", "vocoder_only"):
    x = read(f"A01_{key}.wav")
    m = min(len(x), len(o))
    p_x, _ = power(x[:m])
    p_o_m, _ = power(o[:m])
    # same active mask length may differ by one; rebuild on the common length
    lvl = np.array([band(p_o_m, f, 100, 4000, np.array([i])) for i in range(len(p_o_m))])
    lvl_db = 10 * np.log10(np.maximum(lvl, 1e-20))
    act = lvl_db > lvl_db.max() - 30
    r = rel(p_x, f, act)
    d = {k: round(r[k] - rel_o[k], 2) for k in BANDS}
    out[key] = {"rel": {k: round(v, 2) for k, v in r.items()}, "delta_vs_original": d,
                "mid_emphasis_db": round(-(d["low_0_500"] + d["high_4000_8000"]) / 2, 2)}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
