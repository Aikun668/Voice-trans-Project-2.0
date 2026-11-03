"""清唱版收尾: 核验 + 显著性 + 试听材料。"""
import csv
import math
import shutil
from pathlib import Path

import numpy as np
import parselmouth

from poc_phase1 import clip_count, peak_db, read_wav_pcm16, rms_db, write_wav_pcm16

D = Path("f2/outputs/phase3c_gyjyn_qingchang_v001")
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

print(f"{'时间':>12} {'时长':>5} {'音':>5} {'原唱':>5} {'修正':>8} {'实测':>8} {'误差':>6} {'显著性':>7}")
errs = []
for r in rows:
    app = float(r["applied"])
    if app == 0.0:
        continue
    t0, t1 = float(r["t0"]), float(r["t1"])
    dur = t1 - t0
    ramp = min(0.07, dur / 2.5)
    c0 = med_cents(before, sr, t0 + ramp, t1 - ramp)
    c1 = med_cents(after, sr, t0 + ramp, t1 - ramp)
    if math.isnan(c0) or math.isnan(c1):
        continue
    d = c1 - c0
    errs.append(abs(d - app))
    print(f"{t0:5.2f}-{t1:5.2f} {dur:5.2f} {r['note']:>5} {r['ref_note']:>5} "
          f"{app:+8.1f} {d:+8.1f} {d - app:+6.1f} {dur * abs(app):7.0f}")
if errs:
    print(f"\n执行误差 mean {np.mean(errs):.1f} max {np.max(errs):.1f}")

common = 10 ** ((-16.0 - rms_db(before)) / 20.0)
for src, dst, arr in [("01_target_original.wav", "before.wav", before),
                      ("02_corrected.wav", "after.wav", after)]:
    g = common
    if peak_db(arr * g) > -3.0:
        g *= 10 ** ((-3.0 - peak_db(arr * g)) / 20.0)
    write_wav_pcm16(LST / dst, arr * g, sr)
    x, _ = read_wav_pcm16(LST / dst)
    print(f"{dst:12s} RMS {rms_db(x):+.2f} Peak {peak_db(x):+.2f} 削波 {clip_count(x)}")
t0, t1 = int(2.2 * sr), int(4.2 * sr)
gap = np.zeros(int(0.45 * sr))
aba = np.concatenate([before[t0:t1] * common, gap, after[t0:t1] * common, gap, before[t0:t1] * common])
write_wav_pcm16(LST / "aba_2.2-4.2.wav", aba, sr)
print(f"aba_2.2-4.2.wav {len(aba)/sr:.2f}s")
