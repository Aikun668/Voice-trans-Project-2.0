"""Why does the reverse EQ still sound nasal? (A01 only, read-only)

1) Per-2s window residual mid-emphasis after eq100/eq50 (is the correction global-only?).
2) Formant F1/F2/F3 before and after EQ (does the EQ move the vocal-tract resonances?).
"""
import json
import math
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
AUDIO = ROOT / "experiments" / "F1_neuralsvb_nasal_cancel_2026-11-01_v001" / "run-001" / "listening" / "audio"
SR = 22050
N, H = 2048, 512
STEP = 0.01
BANDS = {"low_0_500": (0, 500), "mid1_500_1500": (500, 1500),
         "mid2_1500_4000": (1500, 4000), "high_4000_8000": (4000, 8000)}
FILES = ["original", "a2p", "eq100", "eq50", "vocoder_only"]


def read(name):
    x, sr = sf.read(str(AUDIO / f"A01_{name}.wav"), dtype="float64")
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
    return float(p[np.ix_(sel, b)].sum()) if sel.any() else 0.0


def rel(p, f, sel):
    vals = {k: band(p, f, lo, hi, sel) for k, (lo, hi) in BANDS.items()}
    ref = max(vals["mid2_1500_4000"], 1e-20)
    return {k: float(10 * np.log10(max(v, 1e-20) / ref)) for k, v in vals.items()}


def active_mask(p, f):
    lvl = np.array([band(p, f, 100, 4000, np.array([i])) for i in range(len(p))])
    ldb = 10 * np.log10(np.maximum(lvl, 1e-20))
    return ldb > ldb.max() - 30


def windows(x, ref_p, ref_act, f):
    p, _ = power(x)
    m = min(len(p), len(ref_p))
    step, win = int(round(0.5 * SR / H)), int(round(2.0 * SR / H))
    rows = []
    for start in range(0, max(1, m - win + 1), step):
        idx = np.arange(start, min(start + win, m))
        sel = idx[ref_act[idx]]
        if len(sel) < 10:
            continue
        d = {k: rel(p, f, sel)[k] - rel(ref_p, f, sel)[k] for k in BANDS}
        rows.append({"start_s": round(start * H / SR, 2), "end_s": round(min(start + win, m) * H / SR, 2),
                     "mid_emphasis_db": round(-(d["low_0_500"] + d["high_4000_8000"]) / 2, 2),
                     "low": round(d["low_0_500"], 2), "high": round(d["high_4000_8000"], 2)})
    return rows


def formants(x):
    snd = parselmouth.Sound(x, sampling_frequency=SR)
    form = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                               maximum_formant=5500, window_length=0.025)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=80, pitch_ceiling=750, voicing_threshold=0.6)
    vals = {1: [], 2: [], 3: []}
    for t in np.arange(0.02, snd.duration - 0.02, 0.01):
        f0 = pitch.get_value_at_time(t)
        if not np.isfinite(f0) or f0 <= 0:
            continue
        for k, lo, hi in ((1, 200, 900), (2, 700, 2500), (3, 1500, 4000)):
            v = form.get_value_at_time(k, t)
            if np.isfinite(v) and lo <= v <= hi:
                vals[k].append(v)
    return {k: (round(float(np.median(v)), 1) if len(v) >= 30 else None) for k, v in vals.items()}


orig = read("original")
p0, f = power(orig)
act = active_mask(p0, f)
out = {"files": list(FILES), "windows": {}, "formants": {}}
for name in FILES:
    x = read(name)
    out["formants"][name] = formants(x)
    if name == "original":
        continue
    ws = windows(x, p0, act, f)
    scores = [w["mid_emphasis_db"] for w in ws]
    out["windows"][name] = {
        "mean": round(float(np.mean(scores)), 2), "std": round(float(np.std(scores)), 2),
        "min": round(float(np.min(scores)), 2), "max": round(float(np.max(scores)), 2),
        "top": sorted(ws, key=lambda w: w["mid_emphasis_db"], reverse=True)[:3],
    }

(AUDIO.parent.parent / "residual_windows.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
