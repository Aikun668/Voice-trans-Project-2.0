"""自身参考对齐: 用"唱得好"的版本作参考, 分析"唱得差"版本的逐音偏差。

同为无伴奏人声, 音高提取干净; 两版语速不同, 用带斜率约束的全局 DTW 对齐(首尾锚定)。
输出逐音偏差表 + 偏差分布, 用于判断"是否存在真正唱错的音"。
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import parselmouth

from poc_phase1 import read_wav_pcm16, rms_db, write_wav_pcm16

HOP = 0.05
FLOOR, CEIL = 75.0, 600.0
NOTE_NAMES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def cents_track(path):
    s = parselmouth.Sound(str(path))
    p = s.to_pitch(HOP, FLOOR, CEIL)
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


def fill(a):
    a = a.copy()
    b = np.isnan(a)
    a[b] = np.interp(np.where(b)[0], np.where(~b)[0], a[~b])
    return a


def global_dtw(q, r):
    """带斜率约束的全局 DTW, 步长仅允许 (1,1)/(1,2)/(2,1), 首尾锚定。

    不允许 (1,0) 原地不动, 避免路径退化成"前半段 1:1、后半段压缩"的坏对齐。
    """
    n, m = len(q), len(r)
    INF = 1e15
    D = np.full((n, m), INF)
    bp = np.zeros((n, m), dtype=np.int8)
    D[0, 0] = abs(q[0] - r[0])
    costmat = np.abs(q[:, None] - r[None, :])

    def shift_right(a, k):
        out = np.full(len(a), INF)
        if k < len(a):
            out[k:] = a[:len(a) - k]
        return out

    for i in range(1, n):
        p1 = D[i - 1]
        p2 = D[i - 2] if i >= 2 else np.full(m, INF)
        a = shift_right(p1, 1)   # (1,1)
        b = shift_right(p1, 2)   # (1,2)
        c = shift_right(p2, 1)   # (2,1)
        stack = np.vstack([a, b, c])
        idx = np.argmin(stack, axis=0)
        D[i] = costmat[i] + stack[idx, np.arange(m)]
        bp[i] = idx.astype(np.int8)

    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        k = bp[i, j]
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i, j = i - 1, j - 2
        else:
            i, j = i - 2, j - 1
        path.append((i, j))
    path.reverse()
    return path, float(D[-1, -1] / max(n, 1))


def note_name(c):
    n = round(c / 100.0)
    return f"{NOTE_NAMES[n % 12]}{4 + (n + 9) // 12}"


def notes_of(path):
    from poc_phase2 import analyze_notes, hz_to_cents, MIN_NOTE_DUR, MAX_SPREAD_CENTS
    x, sr = read_wav_pcm16(Path(path))
    out = []
    for t0, t1, f0s in analyze_notes(x, sr):
        c = np.array([hz_to_cents(f) for f in f0s])
        spread = float(np.percentile(c, 90) - np.percentile(c, 10))
        if t1 - t0 >= MIN_NOTE_DUR and spread <= MAX_SPREAD_CENTS:
            out.append({"t0": t0, "t1": t1, "cents": float(np.median(c)),
                        "hz": float(np.median(f0s))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="要修的版本(唱得差)")
    ap.add_argument("--reference", required=True, help="参考版本(唱得好)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {out}")

    qc = fix_oct(cents_track(a.target))
    rc = fix_oct(cents_track(a.reference))
    q = fill(qc)
    r = fill(rc)
    path, cost = global_dtw(q, r)
    map_r = np.zeros(len(q))
    for i, j in path:
        map_r[min(i, len(map_r) - 1)] = j * HOP

    print(f"DTW 代价 {cost:.0f} 音分/帧; 目标 {len(q) * HOP:.2f}s -> 参考 {len(r) * HOP:.2f}s")

    tnotes = notes_of(a.target)
    rnotes = notes_of(a.reference)
    rows = []
    for n in tnotes:
        i0 = int(round(n["t0"] / HOP))
        i1 = min(int(round(n["t1"] / HOP)), len(map_r) - 1)
        rt0, rt1 = map_r[i0], map_r[i1]
        seg = [m["cents"] for m in rnotes if m["t1"] >= rt0 and m["t0"] <= rt1]
        if not seg:
            continue
        ref_c = float(np.median(seg))
        dev = n["cents"] - ref_c
        rows.append({"t0": round(n["t0"], 2), "t1": round(n["t1"], 2), "hz": round(n["hz"], 1),
                     "note": note_name(n["cents"]), "target_cents": round(n["cents"], 1),
                     "ref_time": round(rt0, 2), "ref_cents": round(ref_c, 1),
                     "ref_note": note_name(ref_c), "deviation_cents": round(dev, 1)})

    if not rows:
        raise SystemExit("没有可对比的音符")
    global_off = float(np.median([x["deviation_cents"] for x in rows]))
    for x in rows:
        rel = x["deviation_cents"] - global_off
        x["relative_cents"] = round(rel, 1)
        x["bucket"] = ("≤50" if abs(rel) <= 50 else "50-100" if abs(rel) <= 100
                       else "100-200" if abs(rel) <= 200 else ">200")

    out.mkdir(parents=True)
    with open(out / "deviations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    abs_rel = np.array([abs(x["relative_cents"]) for x in rows])
    from collections import Counter
    cnt = Counter(x["bucket"] for x in rows)
    summary = {
        "target": a.target, "reference": a.reference, "dtw_cost_cents": round(cost, 1),
        "target_notes": len(rows), "global_offset_cents": round(global_off, 1),
        "abs_relative_deviation": {"median": round(float(np.median(abs_rel)), 1),
                                   "p90": round(float(np.percentile(abs_rel, 90)), 1),
                                   "max": round(float(abs_rel.max()), 1)},
        "buckets": dict(cnt),
        "notes_over_100c": int(sum(1 for v in abs_rel if v > 100)),
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n目标音符 {len(rows)} 个; 整体调差 {global_off:+.1f} 音分")
    print(f"相对偏差 |中位| {np.median(abs_rel):.0f}  |p90| {np.percentile(abs_rel, 90):.0f}  "
          f"|最大| {abs_rel.max():.0f} 音分")
    print("分布:", dict(cnt))
    print(f"超过 100 音分的音: {summary['notes_over_100c']} 个")
    for x in sorted(rows, key=lambda z: -abs(z["relative_cents"]))[:10]:
        print(f"  {x['t0']:5.2f}s 你 {x['hz']:6.1f}Hz({x['note']:4s}) vs 参考 {x['ref_note']:4s} "
              f"相对偏差 {x['relative_cents']:+7.1f}c {x['bucket']}")
    print(f"\n输出: {out}")


if __name__ == "__main__":
    main()
