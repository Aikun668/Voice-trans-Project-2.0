"""核验修正效果: 只看音符中段(避开首尾渐变), 对比修正前后音高。"""
import csv
import math
import sys

import numpy as np
import parselmouth

from poc_phase1 import read_wav_pcm16

FLOOR, CEIL = 75.0, 600.0


def med_cents(x, sr, t0, t1):
    i0, i1 = int(t0 * sr), int(t1 * sr)
    if i1 - i0 < 64:
        return math.nan
    s = parselmouth.Sound(x[i0:i1], sr)
    p = parselmouth.praat.call(s, "To Pitch", 0.01, FLOOR, CEIL)
    v = parselmouth.praat.call(p, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
    return 1200 * math.log2(v / 440.0) if v and v > 0 else math.nan


d = sys.argv[1]
before, sr = read_wav_pcm16(f"{d}/01_target_original.wav")
after, _ = read_wav_pcm16(f"{d}/02_corrected.wav")

rows = list(csv.DictReader(open(f"{d}/deviations.csv", encoding="utf-8")))
print(f"{'时间':>10} {'音':>5} {'意图修正':>9} {'修前':>7} {'修后':>7} {'实际变化':>9} {'误差':>7}")
errs = []
for r in rows:
    if float(r["applied"]) == 0.0:
        continue
    t0, t1 = float(r["t0"]), float(r["t1"])
    dur = t1 - t0
    ramp = min(0.07, dur / 2.5)
    c0 = med_cents(before, sr, t0 + ramp, t1 - ramp)
    c1 = med_cents(after, sr, t0 + ramp, t1 - ramp)
    if math.isnan(c0) or math.isnan(c1):
        continue
    delta = c1 - c0
    err = delta - float(r["applied"])
    errs.append(abs(err))
    print(f"{t0:5.2f}-{t1:5.2f} {r['note']:>5} {float(r['applied']):+9.1f} "
          f"{c0:7.1f} {c1:7.1f} {delta:+9.1f} {err:+7.1f}")

if errs:
    print(f"\n中段实测误差: mean {np.mean(errs):.1f}  max {np.max(errs):.1f} 音分 (n={len(errs)})")
