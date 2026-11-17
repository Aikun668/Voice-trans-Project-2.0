# 定位工程拦截原因：找出处理版相对原音在演唱段内增益异常的时间区间。
# 用法：python diagnose_gain.py <clip_dir>
import sys
import wave
from pathlib import Path

import numpy as np

clip_dir = Path(sys.argv[1])
files = {}
rate = None
for name in ("original_matched", "smule_matched"):
    matches = sorted(clip_dir.glob(f"*__{name}.wav"))
    if len(matches) != 1:
        raise SystemExit(f"找不到唯一的 {name}: {matches}")
    with wave.open(str(matches[0]), "rb") as s:
        assert s.getnchannels() == 1 and s.getsampwidth() == 2
        rate = s.getframerate()
        files[name] = np.frombuffer(s.readframes(s.getnframes()), dtype="<i2").astype(np.float64) / 32768

b = files["original_matched"]
a = files["smule_matched"]

size = int(rate * 0.02)
n = min(len(b), len(a)) // size
br = np.sqrt(np.mean((b[: n * size].reshape(n, size)) ** 2, axis=1))
ar = np.sqrt(np.mean((a[: n * size].reshape(n, size)) ** 2, axis=1))
active = br > max(np.max(br) * 0.08, 1e-4)
gain = np.full(n, np.nan)
idx = np.where(active & (ar > 1e-9))[0]
gain[idx] = 20 * np.log10(ar[idx] / br[idx])
gain -= np.nanmedian(gain)

bad = np.where(np.isfinite(gain) & (gain < -12))[0]
print(f"演唱活跃帧 {int(active.sum())}/{n}；中位归一后增益 < -12 dB 的帧数 {len(bad)}")
if len(bad):
    # 合并相邻帧为区间
    spans, start = [], bad[0]
    for prev, cur in zip(bad[:-1], bad[1:]):
        if cur != prev + 1:
            spans.append((start, prev))
            start = cur
    spans.append((start, bad[-1]))
    for s, e in spans:
        t0, t1 = s * 0.02, (e + 1) * 0.02
        g = float(np.nanmin(gain[s:e + 1]))
        seg = slice(int(t0 * rate), int(t1 * rate))
        rms_a = 20 * np.log10(max(np.sqrt(np.mean(a[seg] ** 2)), 1e-12))
        rms_b = 20 * np.log10(max(np.sqrt(np.mean(b[seg] ** 2)), 1e-12))
        print(f"  {t0:6.2f}s - {t1:6.2f}s  增益最低 {g:6.1f} dB  "
              f"原音段RMS {rms_b:6.1f} dBFS -> 处理版 {rms_a:6.1f} dBFS")
else:
    print("  无 < -12 dB 的帧（异常在别处）")
