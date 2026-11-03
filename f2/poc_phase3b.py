"""F2 Phase-3b: 音符序列对齐版参考修音。

与帧级 DTW 的区别: 直接在音符层面配对(音高差 + 相对位置差), 避免帧级路径在语速差异
较大时漂移。配对后逐音修正, 只修相对偏差超过阈值的音。
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import parselmouth

from poc_phase1 import peak_db, read_wav_pcm16, rms_db, write_wav_pcm16
from self_ref_align import note_name, notes_of

RAMP = 0.070
SEGMENTS = 9
FIX_THRESHOLD = 100.0
MAX_SHIFT = 400.0
PITCH_FLOOR, PITCH_CEIL = 75.0, 600.0
GAP_PENALTY = 220.0       # 音分等价
POS_WEIGHT = 350.0        # 相对位置差权重


def align_notes(t, r):
    """Needleman-Wunsch: 代价 = |音高差| + 位置差权重, 允许插入/删除。"""
    n, m = len(t), len(r)
    t_dur = max(t[-1]["t1"], 1e-6)
    r_dur = max(r[-1]["t1"], 1e-6)
    tp = [(x["t0"] + x["t1"]) / 2 / t_dur for x in t]
    rp = [(x["t0"] + x["t1"]) / 2 / r_dur for x in r]
    INF = 1e15
    D = np.full((n + 1, m + 1), INF)
    D[:, 0] = np.arange(n + 1) * GAP_PENALTY
    D[0, :] = np.arange(m + 1) * GAP_PENALTY
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = abs(t[i - 1]["cents"] - r[j - 1]["cents"]) + POS_WEIGHT * abs(tp[i - 1] - rp[j - 1])
            D[i, j] = min(D[i - 1, j - 1] + sub,
                          D[i - 1, j] + GAP_PENALTY,
                          D[i, j - 1] + GAP_PENALTY)
    i, j = n, m
    pairs, unmatched = [], []
    while i > 0 and j > 0:
        sub = abs(t[i - 1]["cents"] - r[j - 1]["cents"]) + POS_WEIGHT * abs(tp[i - 1] - rp[j - 1])
        if abs(D[i, j] - (D[i - 1, j - 1] + sub)) < 1e-6:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif abs(D[i, j] - (D[i - 1, j] + GAP_PENALTY)) < 1e-6:
            unmatched.append(i - 1)
            i -= 1
        else:
            j -= 1
    while i > 0:
        unmatched.append(i - 1)
        i -= 1
    pairs.reverse()
    return pairs, unmatched


def apply_tapered(sound, shifts):
    manip = parselmouth.praat.call(sound, "To Manipulation", 0.01, PITCH_FLOOR, PITCH_CEIL)
    tier = parselmouth.praat.call(manip, "Extract pitch tier")
    for t0, t1, cents in shifts:
        dur = t1 - t0
        ramp = min(RAMP, dur / 2.5)
        target = 2.0 ** (cents / 1200.0)
        for s in range(SEGMENTS):
            a, b = t0 + dur * s / SEGMENTS, t0 + dur * (s + 1) / SEGMENTS
            mid = (a + b) / 2
            u = min(mid - t0, t1 - mid)
            w = min(1.0, u / ramp) if ramp > 0 else 1.0
            parselmouth.praat.call(tier, "Multiply frequencies", a, b, 1.0 + (target - 1.0) * w)
    parselmouth.praat.call([manip, tier], "Replace pitch tier")
    out = parselmouth.praat.call(manip, "Get resynthesis (overlap-add)")
    return np.asarray(out.values[0], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {out}")

    t = notes_of(a.target)
    r = notes_of(a.reference)
    pairs, unmatched = align_notes(t, r)
    print(f"目标 {len(t)} 音, 参考 {len(r)} 音, 配对 {len(pairs)} 对, 未配对 {len(unmatched)}")

    rows = []
    for i, j in pairs:
        x = t[i]
        ref_c = r[j]["cents"]
        rows.append({"t0": x["t0"], "t1": x["t1"], "hz": x["hz"], "note": note_name(x["cents"]),
                     "cents": x["cents"], "ref_note": note_name(ref_c), "ref_cents": ref_c,
                     "ref_time": r[j]["t0"], "deviation": x["cents"] - ref_c})
    if not rows:
        raise SystemExit("无配对音符")
    off = float(np.median([x["deviation"] for x in rows]))
    print(f"整体调差 {off:+.1f} 音分")
    for x in rows:
        x["relative"] = x["deviation"] - off
        x["applied"] = round(max(-MAX_SHIFT, min(MAX_SHIFT, -x["relative"]))
                             if abs(x["relative"]) > FIX_THRESHOLD else 0.0, 1)
    fix = [x for x in rows if x["applied"] != 0]
    print(f"待修 {len(fix)}/{len(rows)}")
    for x in fix:
        print(f"  {x['t0']:5.2f}s {x['note']:>4} -> {x['ref_note']:>4} 相对 {x['relative']:+7.1f} "
              f"修正 {x['applied']:+7.1f}")

    x_in, sr = read_wav_pcm16(Path(a.target))
    x_corr = apply_tapered(parselmouth.Sound(x_in, sr),
                           [(x["t0"], x["t1"], x["applied"]) for x in fix])
    out.mkdir(parents=True)
    tgt = rms_db(x_in)
    for name, arr in [("01_target_original.wav", x_in), ("02_corrected.wav", x_corr)]:
        g = 10 ** ((tgt - rms_db(arr)) / 20.0)
        pk = peak_db(arr * g)
        if pk > -3.0:
            g *= 10 ** ((-3.0 - pk) / 20.0)
        write_wav_pcm16(out / name, arr * g, sr)
    x_ref, ref_sr = read_wav_pcm16(Path(a.reference))
    g = 10 ** ((tgt - rms_db(x_ref)) / 20.0)
    if peak_db(x_ref * g) > -3.0:
        g *= 10 ** ((-3.0 - peak_db(x_ref * g)) / 20.0)
    write_wav_pcm16(out / "03_reference_take.wav", x_ref * g, ref_sr)

    with open(out / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["t0", "t1", "hz", "note", "cents", "ref_note", "ref_cents", "ref_time",
                  "deviation", "relative", "applied"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(out / "qa_report.json", "w", encoding="utf-8") as f:
        json.dump({"target": a.target, "reference": a.reference,
                   "paired": len(rows), "unmatched": len(unmatched),
                   "global_offset_cents": round(off, 1),
                   "corrected": len(fix),
                   "corrections": [x["applied"] for x in fix]}, f, ensure_ascii=False, indent=2)
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
