"""Phase-3c 收尾: 核验修正 + 生成试听材料 + ABA 拼接。"""
import csv
import math
import shutil
from pathlib import Path

import numpy as np
import parselmouth

from poc_phase1 import clip_count, peak_db, read_wav_pcm16, rms_db, write_wav_pcm16

D = Path("f2/outputs/phase3c_shige_origref_v001")
LST = D / "listening" / "audio"
LST.mkdir(parents=True, exist_ok=True)


def med_cents(x, sr, t0, t1):
    i0, i1 = int(t0 * sr), int(t1 * sr)
    if i1 - i0 < 64:
        return math.nan
    s = parselmouth.Sound(x[i0:i1], sr)
    p = parselmouth.praat.call(s, "To Pitch", 0.01, 110, 600)
    v = parselmouth.praat.call(p, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
    return 1200 * math.log2(v / 440.0) if v and v > 0 else math.nan


before, sr = read_wav_pcm16(D / "01_target_original.wav")
after, _ = read_wav_pcm16(D / "02_corrected.wav")
rows = list(csv.DictReader(open(D / "pairs.csv", encoding="utf-8")))

print(f"{'时间':>11} {'意图':>8} {'修前':>8} {'修后':>8} {'实际':>8} {'误差':>7}")
errs = []
for r in rows:
    if float(r["applied"]) == 0.0:
        continue
    t0, t1 = float(r["t0"]), float(r["t1"])
    ramp = min(0.07, (t1 - t0) / 2.5)
    c0 = med_cents(before, sr, t0 + ramp, t1 - ramp)
    c1 = med_cents(after, sr, t0 + ramp, t1 - ramp)
    if math.isnan(c0) or math.isnan(c1):
        continue
    delta = c1 - c0
    errs.append(abs(delta - float(r["applied"])))
    print(f"{t0:5.2f}-{t1:5.2f} {float(r['applied']):+8.1f} {c0:8.1f} {c1:8.1f} "
          f"{delta:+8.1f} {delta - float(r['applied']):+7.1f}")
if errs:
    print(f"\n中段执行误差: mean {np.mean(errs):.1f}  max {np.max(errs):.1f} 音分")

for src, dst in [("01_target_original.wav", "before.wav"), ("02_corrected.wav", "after.wav")]:
    shutil.copyfile(D / src, LST / dst)
    x, _ = read_wav_pcm16(LST / dst)
    print(f"{dst:12s} RMS {rms_db(x):+.2f} Peak {peak_db(x):+.2f} 削波 {clip_count(x)}")

# ABA: 19.4-21.2s (含 +250 的最大修正)
t0, t1 = int(19.4 * sr), int(21.2 * sr)
gap = np.zeros(int(0.4 * sr))
aba = np.concatenate([before[t0:t1], gap, after[t0:t1], gap, before[t0:t1]])
write_wav_pcm16(LST / "aba_19.4-21.2.wav", aba, sr)
print(f"aba_19.4-21.2.wav {len(aba)/sr:.2f}s")
