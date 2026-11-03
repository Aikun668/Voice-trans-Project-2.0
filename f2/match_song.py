"""用旋律比对确认用户录音对应的是哪一首原唱（v2）。

改进:
1. 局部中值做八度错误修正(偏离局部中值 >600 音分则 ±1200 校正)
2. 用音级(pitch class, mod 1200 音分)做比对, 允许整首歌不同调
3. 子序列 DTW, 允许用户片段落在原曲任意位置
"""

import math
import sys
from pathlib import Path

import numpy as np
import parselmouth

FLOOR, CEIL = 75.0, 600.0
HOP = 0.05
WIN = 21  # ±10 帧, 约 ±0.5s


def raw_cents(path):
    x = parselmouth.Sound(str(path))
    p = parselmouth.praat.call(x, "To Pitch", 0.01, FLOOR, CEIL)
    n = parselmouth.praat.call(p, "Get number of frames")
    v = np.array([(lambda q: q if (q == q and q > 0) else math.nan)(
        parselmouth.praat.call(p, "Get value in frame", i, "Hertz")) for i in range(1, n + 1)])
    c = np.where(np.isnan(v), np.nan, 1200 * np.log2(v / 440.0))
    k = int(round(HOP / 0.01))
    out = [np.nanmedian(c[i:i + k]) if np.any(~np.isnan(c[i:i + k])) else math.nan
           for i in range(0, len(c), k)]
    return np.array(out)


def fix_octaves(c):
    c = c.copy()
    idx = np.where(~np.isnan(c))[0]
    vals = c[idx]
    corr = vals.copy()
    for j in range(len(vals)):
        lo, hi = max(0, j - WIN // 2), min(len(vals), j + WIN // 2 + 1)
        med = np.median(corr[lo:hi])
        d = corr[j] - med
        if abs(d) > 600:
            corr[j] -= np.round(d / 1200.0) * 1200.0
    c[idx] = corr
    return c


def to_pitch_class(c):
    return np.where(np.isnan(c), np.nan, ((c % 1200) + 1200) % 1200)


def pc_distance(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 1200 - d)


def subsequence_dtw(q, r):
    n, m = len(q), len(r)
    INF = 1e12
    D = np.full((n, m), INF)
    D[0] = pc_distance(q[0], r)
    for i in range(1, n):
        c = pc_distance(q[i], r)
        prev = D[i - 1]
        m1 = np.empty(m)
        m1[0] = prev[0]
        m1[1:] = np.minimum(prev[:-1], prev[1:])
        m2 = m1.copy()
        if m > 2:
            m2[2:] = np.minimum(m2[2:], prev[:-2])
        D[i] = c + m2
    end = int(np.argmin(D[-1]))
    return float(D[-1][end] / n), end


def main():
    user_path = Path(sys.argv[1])
    refs = [Path(p) for p in sys.argv[2:]]

    q = to_pitch_class(fix_octaves(raw_cents(user_path)))
    q_filled = q.copy()
    nans = np.isnan(q_filled)
    if nans.all():
        raise SystemExit("用户录音全部无声帧")
    q_filled[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], q_filled[~nans])
    print(f"用户: {user_path.name} 帧={len(q)} 有声={int((~nans).sum())}\n")

    for rp in refs:
        r = to_pitch_class(fix_octaves(raw_cents(rp)))
        bad = np.isnan(r)
        if bad.all():
            print(f"{rp.name}: 全无声, 跳过")
            continue
        r_filled = r.copy()
        r_filled[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], r_filled[~bad])
        cost, end = subsequence_dtw(q_filled, r_filled)
        start = max(0, end - len(q_filled) + 1) * HOP
        print(f"{rp.name}: 有声比例={1 - bad.mean():.2f} 匹配代价={cost:.0f} 音分/帧 "
              f"原曲位置≈{start:.1f}s (共 {len(r) * HOP:.0f}s)")


if __name__ == "__main__":
    main()
