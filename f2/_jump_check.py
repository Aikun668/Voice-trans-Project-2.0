"""量化跳变: 对比修正前后在音符边界的帧间音高跳变。"""
import csv
import math
from pathlib import Path

import numpy as np
import parselmouth

from poc_phase1 import read_wav_pcm16

D = Path("f2/outputs/phase3c_gyjyn_qingchang_v001")
before, sr = read_wav_pcm16(D / "01_target_original.wav")
after, _ = read_wav_pcm16(D / "02_corrected.wav")
rows = [r for r in csv.DictReader(open(D / "pairs.csv", encoding="utf-8")) if float(r["applied"]) != 0.0]


def track(x, t0, t1, step=0.005):
    p = parselmouth.Sound(x, sr).to_pitch(step, 110, 600)
    f = p.selected_array["frequency"]
    ts = p.xs()
    return np.array([t for t in ts if t0 <= t <= t1]),
    

def contour(x, t0, t1, step=0.005):
    p = parselmouth.Sound(x, sr).to_pitch(step, 110, 600)
    f = p.selected_array["frequency"]
    ts = p.xs()
    m = (ts >= t0) & (ts <= t1)
    v = np.where(f[m] > 0, 1200 * np.log2(np.maximum(f[m], 1e-9) / 440.0), np.nan)
    return ts[m], v


print(f"{'音符':>12} {'修前最大帧跳':>12} {'修后最大帧跳':>12} {'边界处前后差':>12}")
for r in rows:
    t0, t1 = float(r["t0"]), float(r["t1"])
    a0, a1 = t0 - 0.10, t1 + 0.10
    tsa, va = contour(before, a0, a1)
    tsb, vb = contour(after, a0, a1)
    def maxjump(v):
        d = []
        for i in range(1, len(v)):
            if not math.isnan(v[i]) and not math.isnan(v[i - 1]):
                d.append(abs(v[i] - v[i - 1]))
        return max(d) if d else float("nan")
    # 边界前后各 15ms 的中位差
    def edge_step(ts, v, tb):
        pre = v[(ts >= tb - 0.03) & (ts < tb)]
        post = v[(ts >= tb) & (ts <= tb + 0.03)]
        pre = pre[~np.isnan(pre)]
        post = post[~np.isnan(post)]
        if len(pre) == 0 or len(post) == 0:
            return float("nan")
        return float(np.median(post) - np.median(pre))
    print(f"{t0:5.2f}-{t1:5.2f} {maxjump(va):12.1f} {maxjump(vb):12.1f} "
          f"{edge_step(tsb, vb, t0):12.1f}")
