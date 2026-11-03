"""F2 Phase-3 第一步: 参考旋律对齐 + 逐音偏差对比表(不改音)。

流程:
1. 用户录音与原唱全曲分别做 F0 提取(parselmouth 批量 API, 快)
2. 八度错误修正
3. 音级(pitch class)子序列 DTW -> 找到用户片段在原曲的位置 + 时间映射
4. 按用户音符逐音统计: 该音对应原曲哪些帧 -> 原曲中位音高 -> 偏差
5. 导出对齐后的原曲片段(供人耳核对是不是同一段)
"""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import parselmouth

from poc_phase1 import read_wav_pcm16, rms_db, write_wav_pcm16

HOP = 0.05
FLOOR, CEIL = 75.0, 600.0
WIN = 21
NOTE_NAMES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def f0_cents(path):
    """整段 F0 -> 音分, 50ms 一帧。用 parselmouth 批量接口, 避免逐帧调用。"""
    s = parselmouth.Sound(str(path))
    p = s.to_pitch(HOP, FLOOR, CEIL)
    f = p.selected_array["frequency"]
    c = np.where(f > 0, 1200 * np.log2(np.maximum(f, 1e-9) / 440.0), np.nan)
    return c


def fix_octaves(c):
    c = c.copy()
    idx = np.where(~np.isnan(c))[0]
    vals = c[idx].copy()
    for j in range(len(vals)):
        lo, hi = max(0, j - WIN // 2), min(len(vals), j + WIN // 2 + 1)
        med = np.median(vals[lo:hi])
        d = vals[j] - med
        if abs(d) > 600:
            vals[j] -= np.round(d / 1200.0) * 1200.0
    c[idx] = vals
    return c


def to_pc(c):
    return ((c % 1200) + 1200) % 1200


def pc_dist(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 1200 - d)


def subseq_dtw(q, r):
    n, m = len(q), len(r)
    INF = 1e15
    D = np.full((n, m), INF)
    bp = np.zeros((n, m), dtype=np.int8)
    D[0] = pc_dist(q[0], r)
    for i in range(1, n):
        c = pc_dist(q[i], r)
        stay = D[i - 1]
        step1 = np.concatenate(([INF], D[i - 1][:-1]))
        step2 = np.concatenate(([INF, INF], D[i - 1][:-2])) if m > 2 else np.full(m, INF)
        stack = np.vstack([stay, step1, step2])
        idx = np.argmin(stack, axis=0)
        D[i] = c + stack[idx, np.arange(m)]
        bp[i] = idx.astype(np.int8)
    end = int(np.argmin(D[-1]))
    cost = float(D[-1][end] / n)
    path = [(n - 1, end)]
    i, j = n - 1, end
    while i > 0:
        d = bp[i, j]
        i -= 1
        if d == 1:
            j -= 1
        elif d == 2:
            j -= 2
        path.append((i, j))
    path.reverse()
    return cost, path


def note_name(cents):
    n = round(cents / 100.0)
    return f"{NOTE_NAMES[n % 12]}{4 + (n + 9) // 12}"


def user_notes(path):
    """复用 Phase-2 的分音符逻辑, 返回稳定音符列表(时间+中位音分)。"""
    from poc_phase2 import analyze_notes, hz_to_cents, MIN_NOTE_DUR, MAX_SPREAD_CENTS
    x, sr = read_wav_pcm16(Path(path))
    out = []
    for t0, t1, f0s in analyze_notes(x, sr):
        cents = np.array([hz_to_cents(f) for f in f0s])
        spread = float(np.percentile(cents, 90) - np.percentile(cents, 10))
        if t1 - t0 >= MIN_NOTE_DUR and spread <= MAX_SPREAD_CENTS:
            out.append({"t0": t0, "t1": t1, "cents": float(np.median(cents)),
                        "hz": float(np.median(f0s)), "spread": spread})
    return out, x, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {out}")

    q_raw = f0_cents(args.user)
    r_raw = f0_cents(args.ref)
    q = fix_octaves(q_raw)
    r = fix_octaves(r_raw)

    q_pc = to_pc(q)
    r_pc = to_pc(r)
    badq = np.isnan(q_pc)
    badr = np.isnan(r_pc)
    qf = q_pc.copy()
    qf[badq] = np.interp(np.where(badq)[0], np.where(~badq)[0], q_pc[~badq])
    rf = r_pc.copy()
    rf[badr] = np.interp(np.where(badr)[0], np.where(~badr)[0], r_pc[~badr])

    cost, path = subseq_dtw(qf, rf)
    map_time = np.full(len(qf), np.nan)
    for i, j in path:
        map_time[i] = j * HOP
    # 补全映射
    ok = ~np.isnan(map_time)
    map_time[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], map_time[ok])

    ref_start, ref_end = float(map_time[0]), float(map_time[-1])
    print(f"对齐: 代价 {cost:.0f} 音分/帧; 原曲对应区间 {ref_start:.2f}-{ref_end:.2f}s "
          f"(时长 {ref_end - ref_start:.2f}s, 用户 {len(q) * HOP:.2f}s)")

    notes, x_user, sr = user_notes(args.user)
    ref_x, ref_sr = read_wav_pcm16(Path(args.ref))

    rows = []
    for n in notes:
        i0 = int(round(n["t0"] / HOP))
        i1 = min(int(round(n["t1"] / HOP)), len(map_time) - 1)
        j0 = int(round(np.nanmin(map_time[i0:i1 + 1]) / HOP))
        j1 = int(round(np.nanmax(map_time[i0:i1 + 1]) / HOP))
        j0, j1 = max(0, j0), min(len(r) - 1, j1)
        seg = r[j0:j1 + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) == 0:
            continue
        ref_c = float(np.median(seg))
        dev = n["cents"] - ref_c
        rows.append({
            "user_t0_s": round(n["t0"], 2), "user_t1_s": round(n["t1"], 2),
            "user_hz": round(n["hz"], 1), "user_note": note_name(n["cents"]),
            "user_cents": round(n["cents"], 1),
            "ref_time_s": round(j0 * HOP, 2), "ref_cents": round(ref_c, 1),
            "ref_note": note_name(ref_c), "deviation_cents": round(dev, 1),
        })

    global_off = float(np.median([r["deviation_cents"] for r in rows])) if rows else 0.0
    for r in rows:
        r["relative_cents"] = round(r["deviation_cents"] - global_off, 1)
        r["flag"] = "偏差大" if abs(r["deviation_cents"] - global_off) > 60 else "接近"

    out.mkdir(parents=True)
    shutil.copyfile(args.user, out / "00_user_original.wav")

    # 导出对齐后的原曲片段(同区间, 响度匹配到用户录音)
    a, b = int(ref_start * ref_sr), int(ref_end * ref_sr)
    frag = ref_x[a:b].copy()
    tgt = rms_db(x_user)
    g = 10 ** ((tgt - rms_db(frag)) / 20.0)
    peak = np.max(np.abs(frag * g))
    if peak > 0.7079:
        g *= 0.7079 / peak
    write_wav_pcm16(out / "01_ref_fragment_aligned.wav", frag * g, ref_sr)

    with open(out / "notes_vs_ref.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "user": args.user, "ref": args.ref, "align_cost_cents": round(cost, 1),
        "ref_region_s": [round(ref_start, 2), round(ref_end, 2)],
        "global_key_offset_cents": round(global_off, 1),
        "notes": len(rows),
        "big_deviation_notes": sum(1 for r in rows if r["flag"] == "偏差大"),
    }
    with open(out / "alignment.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n整体调差(用户比原唱高) {global_off:+.1f} 音分; 音符 {len(rows)} 个, "
          f"相对偏差>60 音分的有 {summary['big_deviation_notes']} 个")
    for r in rows[:12]:
        print(f"  {r['user_t0_s']:5.2f}s 你 {r['user_hz']:6.1f}Hz/{r['user_note']:4s} "
              f"v.s. 原唱 {r['ref_cents']:8.1f}c/{r['ref_note']:4s}  偏差 {r['deviation_cents']:+7.1f}c "
              f"(相对 {r['relative_cents']:+6.1f}) {r['flag']}")
    print(f"\n输出: {out}")


if __name__ == "__main__":
    main()
