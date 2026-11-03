"""鲁棒滑窗匹配 v2: 只用双方都有声的帧, 先减整体调差, 再用中位绝对差。"""
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
    v = c[idx].copy()
    W = 11
    for j in range(len(v)):
        lo, hi = max(0, j - W), min(len(v), j + W + 1)
        med = np.median(v[lo:hi])
        d = v[j] - med
        if abs(d) > 600:
            v[j] -= np.round(d / 1200.0) * 1200.0
    c[idx] = v
    return c


def robust_scan(q, r):
    n, m = len(q), len(r)
    best = (1e9, 0)
    costs = np.full(m - n + 1, 1e9)
    for s in range(m - n + 1):
        rs = r[s:s + n]
        both = ~np.isnan(q) & ~np.isnan(rs)
        if both.sum() < n * 0.4:
            continue
        d = q[both] - rs[both]
        d = d - np.median(d)
        c = float(np.median(np.abs(d)))
        costs[s] = c
        if c < best[0]:
            best = (c, s)
    return best, costs


if __name__ == "__main__":
    pairs = [
        ("平凡之路(已知正确)",
         r"test_dataset\f1_dataset_v2\A_thin\self\wav\平凡之路正常版.wav",
         r"test_dataset\reference\vocals\平凡之路_朴树_vocals.wav"),
        ("shige vs 自由飞翔",
         r"test_dataset\my_singing\shige_medium.wav",
         r"test_dataset\reference\vocals\自由飞翔_凤凰传奇_vocals.wav"),
    ]
    for label, user, ref in pairs:
        q = fix_oct(cents_of(user, 130, 450))
        r = fix_oct(cents_of(ref, 130, 450))
        (cost, s), costs = robust_scan(q, r)
        valid = costs[costs < 1e8]
        print(f"{label}: 最佳位置 {s * 0.05:.1f}s  中位绝对差 {cost:.0f} 音分  "
              f"全曲中位 {np.median(valid):.0f}")
