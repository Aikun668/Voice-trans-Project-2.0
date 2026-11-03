"""粗定位: 把用户音级序列在原曲上滑窗比对, 找最佳位置; 并对多个音高范围做对比。"""
import math
import sys

import numpy as np
import parselmouth


def cents_of(path, floor, ceil, hop=0.05):
    s = parselmouth.Sound(str(path))
    p = s.to_pitch(hop, floor, ceil)
    f = p.selected_array["frequency"]
    return np.where(f > 0, 1200 * np.log2(np.maximum(f, 1e-9) / 440.0), np.nan)


def fix_oct(c):
    c = c.copy()
    idx = np.where(~np.isnan(c))[0]
    vals = c[idx].copy()
    W = 11
    for j in range(len(vals)):
        lo, hi = max(0, j - W), min(len(vals), j + W + 1)
        med = np.median(vals[lo:hi])
        d = vals[j] - med
        if abs(d) > 600:
            vals[j] -= np.round(d / 1200.0) * 1200.0
    c[idx] = vals
    return c


def pc(c):
    return ((c % 1200) + 1200) % 1200


def fill(a):
    a = a.copy()
    b = np.isnan(a)
    if b.all():
        return None
    a[b] = np.interp(np.where(b)[0], np.where(~b)[0], a[~b])
    return a


def scan(q, r):
    n, m = len(q), len(r)
    D = np.abs(q[:, None] - r[None, :])
    D = np.minimum(D, 1200 - D)
    best = (1e9, 0)
    costs = np.full(m - n + 1, 1e9)
    idx = np.arange(n)
    for s in range(m - n + 1):
        c = D[idx, s + idx].mean()
        costs[s] = c
        if c < best[0]:
            best = (c, s)
    return best, costs


user = sys.argv[1]
ref = sys.argv[2]
for floor, ceil in [(75, 600), (130, 450), (150, 500)]:
    q = fill(pc(fix_oct(cents_of(user, floor, ceil))))
    r = fill(pc(fix_oct(cents_of(ref, floor, ceil))))
    if q is None or r is None:
        print("skip", floor, ceil)
        continue
    (cost, s), costs = scan(q, r)
    print(f"floor={floor} ceil={ceil}: 最佳位置 {s * 0.05:.1f}s  代价 {cost:.0f} 音分/帧  "
          f"vs 随机基准 ~300  中位代价 {np.median(costs):.0f}")
